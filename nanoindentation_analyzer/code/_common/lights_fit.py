"""Curve preparation, the pipeline estimator, and three cross-check methods.

Estimators
----------
``fit_pipeline``   the production estimator: one regression of s on F^{2/3}
                   over the fixed indentation window, in two passes.
``fit_nonlinear``  nonlinear Hertz with the contact point free, fitted in
                   force, with a stretch of pre-contact baseline kept inside
                   the window so it constrains the contact point from below.
``fit_stiffness``  contact-point-free: (dF/ds)^2 is linear in s for a
                   half-space. The derivative's own variance is subtracted,
                   because squaring rectifies noise and the bias is largest
                   where the stiffness is smallest - the end that sets the
                   intercept - and the regression is weighted, because the
                   variance of a squared quantity grows with its mean.
``fit_film``       ``fit_nonlinear`` with the Garcia bonded-film correction,
                   thickness an input rather than a fitted parameter.

All four report E* in kPa and the contact coordinate in um, in the same
coordinate as ``s``, so they can be compared curve by curve. The last three
are cross-checks on the first; ``fit_pipeline`` produces the reported number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, signal

__all__ = [
    "DEFAULT_THICKNESS_UM",
    "FIT_MAX_DEPTH_UM",
    "FIT_MIN_DEPTH_UM",
    "FitOut",
    "Prepared",
    "depth_diagnostics",
    "e_star_from_slope",
    "fit_film",
    "fit_nonlinear",
    "fit_pipeline",
    "fit_stiffness",
    "garcia_bec",
    "linearised_line",
    "prepare",
    "seed_window",
]

#: Indentation cap for every estimator, in micrometres.
FIT_MAX_DEPTH_UM = 3.5

#: Lower edge of the fit window, in indentation. Below ~0.5 um the Hertz force
#: is too noisy
FIT_MIN_DEPTH_UM = 0.5

#: Stop before the commanded ramp ends, so no estimator sees the
#: deceleration into the hold.
RAMP_GUARD_S = 0.02

#: Gap left between the end of the baseline window and the detected force
#: onset, so that the baseline never contains contact.
PRE_GAP_S = 0.30

#: Derivative smoothing for the stiffness method, in seconds, so 1 kHz and
#: 100 Hz are filtered identically.
DERIVATIVE_WINDOW_S = 0.05

#: Assumed bonded-film thickness when none is supplied, in micrometres.
DEFAULT_THICKNESS_UM = 50.0


@dataclass
class Prepared:
    """Loading branch, baseline-removed."""

    time_s: np.ndarray
    force_nN: np.ndarray
    s_nm: np.ndarray
    dt_s: float
    radius_um: float
    spring_constant_n_m: float
    peak_force_nN: float
    peak_index: int
    onset_index: int
    sigma_nN: float
    baseline_slope_nN_per_s: float
    baseline_intercept_nN: float
    baseline_model: str
    pre_n: int
    pre_duration_s: float
    pre_resid_rms_nN: float
    pre_span_nN: float
    ramp_start_s: float
    ramp_end_s: float
    notes: list = field(default_factory=list)
    extras: dict = field(default_factory=dict)

    @property
    def s_onset_nm(self) -> float:
        if self.onset_index < 0:
            return math.nan
        return float(self.s_nm[self.onset_index])


@dataclass
class FitOut:
    """Estimator output."""

    method: str
    ok: bool = False
    e_star_kPa: float = math.nan
    contact_s0_um: float = math.nan
    r_squared: float = math.nan
    rmse_nN: float = math.nan
    n_points: int = 0
    depth_lo_um: float = math.nan
    depth_hi_um: float = math.nan
    reason: str = ""
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def e_star_from_slope(slope_nN_per_nm32: float, radius_um: float) -> float:
    """F = slope * delta^{3/2} with F in nN and delta in nm, to E* in kPa."""
    if not np.isfinite(slope_nN_per_nm32) or radius_um <= 0:
        return math.nan
    return (
        0.75 * (10.0**7.5) * (slope_nN_per_nm32 * 1e-3) / math.sqrt(radius_um)
    )


def _r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    resid = observed - predicted
    total = observed - observed.mean()
    denom = float(total @ total)
    if denom <= 0:
        return math.nan
    return 1.0 - float(resid @ resid) / denom


def garcia_bec(depth_nm, radius_um: float, thickness_um: float):
    """Bonded-film stiffening factor, in chi = a/h.
    """
    chi = np.sqrt(radius_um * np.clip(depth_nm, 0.0, None) / 1000.0) / max(
        thickness_um, 1e-6
    )
    return 1.0 + 1.133 * chi + 1.497 * chi**2 + 1.469 * chi**3 + 0.755 * chi**4


def ramp_stop_index(
    time_s: np.ndarray,
    s_nm: np.ndarray,
    upto: int,
    *,
    guard_s: float = RAMP_GUARD_S,
    velocity_fraction: float = 0.85,
) -> int:
    """Last index still on the constant-velocity ramp
    """
    n = int(min(upto + 1, time_s.size))
    if n < 20:
        return max(n - 1, 0)
    dt = float(np.median(np.diff(time_s)))
    win = max(5, round(0.03 / dt) | 1)
    if win >= n:
        win = max(5, (n // 2) * 2 - 1)
    try:
        v = signal.savgol_filter(s_nm[:n], win, 2, deriv=1, delta=dt)
    except ValueError:
        v = np.gradient(s_nm[:n], time_s[:n])
    moving = v > 1000.0
    if int(moving.sum()) < 10:
        return n - 1
    v0 = float(np.median(v[moving]))
    on_ramp = np.flatnonzero(v >= velocity_fraction * v0)
    if on_ramp.size == 0:
        return n - 1
    stop = int(on_ramp[-1])
    stop -= max(1, round(guard_s / dt))
    return int(np.clip(stop, 20, n - 1))


def _last_baseline_index(
    force: np.ndarray, threshold: float, upto: int, smooth_pts: int
) -> int:
    """Index just after the last sample still sitting at the baseline
    """
    n = int(min(upto + 1, force.size))
    if n < 5:
        return -1
    win = int(np.clip(smooth_pts | 1, 3, max(3, (n // 4) * 2 - 1)))
    kernel = np.ones(win) / win
    smooth = np.convolve(force[:n], kernel, mode="same")
    below = np.flatnonzero(smooth < threshold)
    if below.size == 0:
        return -1
    idx = int(below[-1]) + 1
    return idx if idx < n else -1


def _stiffness_onset(
    time_s: np.ndarray,
    force_nN: np.ndarray,
    s_nm: np.ndarray,
    dt_s: float,
    peak_i: int,
    *,
    window_s: float = 0.06,
    reference_s: float = 0.30,
    n_sigma: float = 6.0,
    peak_fraction: float = 0.05,
) -> int:
    """Find contact from the contact stiffness
    """
    n = int(min(peak_i + 1, time_s.size))
    if n < 40:
        return -1
    win = max(5, round(window_s / dt_s) | 1)
    if win >= n:
        win = max(5, (n // 2) * 2 - 1)
    if win < 5:
        return -1
    try:
        df = signal.savgol_filter(force_nN[:n], win, 2, deriv=1, delta=dt_s)
        ds = signal.savgol_filter(s_nm[:n], win, 2, deriv=1, delta=dt_s)
    except ValueError:
        return -1
    moving = ds > 1000.0
    if int(moving.sum()) < 30:
        return -1
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(moving, df / ds, np.nan)
    t_move = time_s[:n][moving][0]
    ref = moving & (time_s[:n] <= t_move + reference_s)
    if int(ref.sum()) < 20:
        ref = moving & (time_s[:n] <= t_move + 0.5 * (time_s[n - 1] - t_move))
    scatter = float(1.4826 * np.median(np.abs(k[ref] - np.median(k[ref]))))
    if not np.isfinite(scatter) or scatter <= 0:
        scatter = float(np.nanstd(k[ref])) or 1e-9
    k_peak = float(np.nanpercentile(k[moving], 95))
    if not np.isfinite(k_peak) or k_peak <= 0:
        return -1
    threshold = max(n_sigma * scatter, peak_fraction * k_peak)
    below = np.flatnonzero(moving & np.isfinite(k) & (k < threshold))
    if below.size == 0:
        return -1
    idx = int(below[-1]) + 1
    return idx if 0 < idx < n else -1


def baseline_extrapolation_error(
    time_s: np.ndarray,
    force_nN: np.ndarray,
    pre_mask: np.ndarray,
    degree: int,
) -> float:
    """Baseline model extrapolation error from contact fitting
    """
    idx = np.flatnonzero(pre_mask)
    if idx.size < 60:
        return math.nan
    cut = int(idx.size * 2 // 3)
    train, holdout = idx[:cut], idx[cut:]
    if train.size < 30 or holdout.size < 10:
        return math.nan
    try:
        coef = np.polyfit(time_s[train], force_nN[train], degree)
    except (np.linalg.LinAlgError, ValueError):
        return math.nan
    resid = force_nN[holdout] - np.polyval(coef, time_s[holdout])
    return float(np.median(np.abs(resid)))


def oscillation_metrics(
    time_s: np.ndarray,
    resid_nN: np.ndarray,
    dt_s: float,
    fit_duration_s: float,
) -> dict:
    """Amplitude, frequency and size of the ferrule-top wobble
    """
    out = {
        "osc_amplitude_nN": math.nan,
        "osc_frequency_hz": math.nan,
        "osc_effective_nN": math.nan,
    }
    n = int(resid_nN.size)
    if n < 64 or not np.isfinite(dt_s) or dt_s <= 0:
        return out
    taper = np.hanning(n)
    x = (resid_nN - np.mean(resid_nN)) * taper
    spec = np.abs(np.fft.rfft(x)) * 2.0 / np.sum(taper)
    freq = np.fft.rfftfreq(n, dt_s)
    keep = (freq > 0.5) & (freq < 0.45 / dt_s)
    if not keep.any():
        return out
    k = int(np.argmax(spec[keep]))
    amp = float(spec[keep][k])
    f_osc = float(freq[keep][k])
    cycles = max(1.0, f_osc * max(fit_duration_s, dt_s))
    out["osc_amplitude_nN"] = amp
    out["osc_frequency_hz"] = f_osc
    out["osc_effective_nN"] = amp / math.sqrt(cycles)
    return out


def prepare(
    rc,
    *,
    ramp_guard_s: float = RAMP_GUARD_S,
    pre_gap_s: float = PRE_GAP_S,
    baseline_model: str = "linear",
) -> Prepared | None:
    """Cut the loading branch and remove drift-following baseline
    """
    t = np.asarray(rc.time_s, dtype=float)
    f = np.asarray(rc.load_uN, dtype=float) * 1000.0
    s = np.asarray(rc.s_nm, dtype=float)
    if t.size < 50:
        return None
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return None
    radius = float(rc.radius_um)
    if not np.isfinite(radius) or radius <= 0:
        return None
    spring_constant = float(rc.spring_constant_n_m)
    if not np.isfinite(spring_constant) or spring_constant <= 0:
        return None

    r0, r1 = rc.ramp_window_s
    mask = np.isfinite(t) & np.isfinite(f) & np.isfinite(s)
    notes: list = []
    if not np.isfinite(r1):
        notes.append("no_ramp_window_in_header")
    # Header end-of-ramp as a first bound, then the one from the data.
    coarse = np.flatnonzero(mask)
    if coarse.size < 50:
        return None
    upper = int(coarse[-1])
    if np.isfinite(r1):
        within = np.flatnonzero(mask & (t <= r1 + 0.05))
        if within.size >= 50:
            upper = int(within[-1])
    stop = ramp_stop_index(t, s, upper, guard_s=ramp_guard_s)
    mask &= np.arange(t.size) <= stop
    if int(mask.sum()) < 50:
        return None
    peak_i = int(np.nanargmax(np.where(mask, f, -np.inf)))
    mask &= np.arange(t.size) <= peak_i

    early = t <= t[0] + 0.10
    if int(early.sum()) < 5:
        early = np.zeros(t.size, bool)
        early[: min(5, t.size)] = True
    b0 = float(np.median(f[early]))
    sigma0 = float(1.4826 * np.median(np.abs(f[early] - b0)))
    if not np.isfinite(sigma0) or sigma0 <= 0:
        sigma0 = float(np.std(f[early])) or 1e-3
    peak0 = float(f[peak_i] - b0)
    if not np.isfinite(peak0) or peak0 <= 0:
        return None

    smooth_pts = max(3, round(0.02 / dt))
    onset = _stiffness_onset(t, f, s, dt, peak_i)
    if onset <= 0:
        thr = b0 + max(6.0 * sigma0, 0.01 * peak0)
        onset = _last_baseline_index(f, thr, peak_i, smooth_pts)
        notes.append("onset_from_force_threshold")

    # --- baseline over the pre-contact stretch, then refinement ---
    slope, intercept, used = 0.0, b0, "constant"
    sigma = sigma0
    base = np.full_like(f, b0)
    fc = f - base
    deg = {
        "quadratic": 2,
        "linear": 1,
        "constant": 0,
        "initial_constant": 0,
    }.get(baseline_model, 1)
    for _ in range(2):
        cut = t[onset] - pre_gap_s if onset > 0 else t[0] + 0.10
        pre = mask & (t <= cut)
        n_pre = int(pre.sum())
        if n_pre >= 60 and deg >= 1:
            coef = np.polyfit(t[pre], f[pre], deg)
            base = np.polyval(coef, t)
            slope = float(coef[-2])
            intercept = float(coef[-1])
            used = baseline_model
        elif baseline_model == "initial_constant":
            # The median of the first 0.10 s of the record
            intercept = b0
            base = np.full_like(f, b0)
            slope = 0.0
            used = "initial_constant"
        elif n_pre >= 10:
            intercept = float(np.median(f[pre]))
            base = np.full_like(f, intercept)
            slope = 0.0
            used = "constant"
        else:
            base = np.full_like(f, b0)
            intercept, slope, used = b0, 0.0, "constant_fallback"
            if "short_pre_contact" not in notes:
                notes.append("short_pre_contact")
        resid = f[pre] - base[pre] if n_pre >= 10 else f[early] - b0
        sigma = float(1.4826 * np.median(np.abs(resid - np.median(resid))))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = sigma0
        fc = f - base
        peak_now = float(fc[peak_i])
        if not np.isfinite(peak_now) or peak_now <= 0:
            return None

        if "onset_from_force_threshold" not in notes:
            break
        thr = max(6.0 * sigma, 0.01 * peak_now)
        new_onset = _last_baseline_index(fc, thr, peak_i, smooth_pts)
        if new_onset <= 0:
            break
        onset = new_onset

    cut = t[onset] - pre_gap_s if onset > 0 else t[0] + 0.10
    pre = mask & (t <= cut)
    n_pre = int(pre.sum())

    s = s + base / spring_constant
    pre_rms = float(np.std(fc[pre])) if n_pre >= 10 else math.nan
    pre_span = (
        float(np.nanmax(fc[pre]) - np.nanmin(fc[pre]))
        if n_pre >= 10
        else math.nan
    )
    fit_duration = (
        float(t[peak_i] - t[onset]) if onset > 0 else float(t[peak_i] - t[0])
    )
    extras = {
        "baseline_extrap_nN": baseline_extrapolation_error(
            t, f, pre, max(deg, 1)
        ),
        "fit_duration_s": fit_duration,
        "ramp_stop_s": float(t[stop]) + ramp_guard_s,
    }
    if n_pre >= 64:
        extras.update(oscillation_metrics(t[pre], fc[pre], dt, fit_duration))
    prepared = Prepared(
        time_s=t[mask],
        force_nN=fc[mask],
        s_nm=s[mask],
        dt_s=dt,
        radius_um=radius,
        spring_constant_n_m=spring_constant,
        peak_force_nN=float(fc[peak_i]),
        peak_index=int(mask[: peak_i + 1].sum()) - 1,
        onset_index=int(mask[: onset + 1].sum()) - 1 if onset > 0 else -1,
        sigma_nN=sigma,
        baseline_slope_nN_per_s=slope,
        baseline_intercept_nN=intercept,
        baseline_model=used,
        pre_n=n_pre,
        pre_duration_s=(
            float(t[pre][-1] - t[pre][0]) if n_pre >= 2 else math.nan
        ),
        pre_resid_rms_nN=pre_rms,
        pre_span_nN=pre_span,
        ramp_start_s=float(r0),
        ramp_end_s=float(r1),
        notes=notes,
        extras=extras,
    )
    return prepared


# ----------------------------------------------------------------------
# linearized Hertz plot
# ----------------------------------------------------------------------
def linearised_line(p: Prepared, lo: float, hi: float):
    """Least squares s = s0 + beta F^{2/3} over [lo, hi] * peak force."""
    f, s = p.force_nN, p.s_nm
    win = (f >= lo * p.peak_force_nN) & (f <= hi * p.peak_force_nN) & (f > 0)
    if int(win.sum()) < 20:
        return None
    x = np.power(f[win], 2.0 / 3.0)
    design = np.column_stack([x, np.ones(x.size)])
    beta, *_ = np.linalg.lstsq(design, s[win], rcond=None)
    return float(beta[0]), float(beta[1]), win, design


# ----------------------------------------------------------------------
# pipeline estimator
# ----------------------------------------------------------------------
def fit_pipeline(
    p: Prepared,
    *,
    depth_lo_um: float = FIT_MIN_DEPTH_UM,
    depth_hi_um: float = FIT_MAX_DEPTH_UM,
    seed_force_lo: float = 0.20,
    seed_force_hi: float = 1.00,
    slope_from: str = "linearised",
) -> FitOut:
    """ Linearized Hertz over fixed indentation window
    """
    out = FitOut("pipeline")
    f, s = p.force_nN, p.s_nm
    seed = linearised_line(p, seed_force_lo, seed_force_hi)
    if seed is None:
        out.reason = "no_seed_window"
        return out
    beta_seed, s0_seed, _w, _d = seed
    if beta_seed <= 0 or not np.isfinite(s0_seed):
        out.reason = "bad_seed"
        return out

    win = (
        (s >= s0_seed + depth_lo_um * 1000.0)
        & (s <= s0_seed + depth_hi_um * 1000.0)
        & (f > 0)
        & np.isfinite(f)
        & np.isfinite(s)
    )
    if int(win.sum()) < 25:
        out.reason = "window_collapsed"
        out.contact_s0_um = s0_seed / 1000.0
        out.extra = {"seed_s0_um": s0_seed / 1000.0}
        return out
    x = np.power(f[win], 2.0 / 3.0)
    design = np.column_stack([x, np.ones(x.size)])
    coef, *_ = np.linalg.lstsq(design, s[win], rcond=None)
    beta, s0 = float(coef[0]), float(coef[1])
    if beta <= 0:
        out.reason = "nonpositive_beta"
        out.extra = {"seed_s0_um": s0_seed / 1000.0}
        return out

    depth = s - s0
    if slope_from == "through_origin":
        keep = win & (depth > 0)
        xx = np.power(depth[keep], 1.5)
        slope = float((xx @ f[keep]) / (xx @ xx))
    else:
        slope = beta ** (-1.5)
    pred_f = np.where(
        depth[win] > 0,
        slope * np.power(np.clip(depth[win], 0, None), 1.5),
        0.0,
    )
    predicted_s = s0 + beta * x
    out.ok = True
    out.e_star_kPa = e_star_from_slope(slope, p.radius_um)
    out.contact_s0_um = s0 / 1000.0
    out.r_squared = _r2(s[win], predicted_s)
    out.rmse_nN = float(np.sqrt(np.mean((f[win] - pred_f) ** 2)))
    out.n_points = int(win.sum())
    out.depth_lo_um = float(np.nanmin(depth[win])) / 1000.0
    out.depth_hi_um = float(np.nanmax(depth[win])) / 1000.0
    out.extra = {
        "seed_s0_um": s0_seed / 1000.0,
        "beta_nm_per_nN23": beta,
        "hertz_slope_nN_per_nm32": slope,
        "depth_reached_um": float(np.nanmax(s) - s0_seed) / 1000.0,
        "force_lo_frac": float(np.nanmin(f[win]) / p.peak_force_nN),
        "force_hi_frac": float(np.nanmax(f[win]) / p.peak_force_nN),
        "nrmse_percent": float(
            100.0 * out.rmse_nN / max(np.nanmax(f[win]), 1e-9)
        ),
    }
    return out


# ----------------------------------------------------------------------
# analysis window
# ----------------------------------------------------------------------
def seed_window(
    p: Prepared,
    seed_s0_nm: float,
    depth_lo_um: float,
    depth_hi_um: float,
    *,
    include_precontact_um: float = 0.0,
) -> np.ndarray:
    """The points feeded to the estimators
    """
    s = p.s_nm
    finite = np.isfinite(s) & np.isfinite(p.force_nN)
    inside = s >= seed_s0_nm + depth_lo_um * 1000.0
    if include_precontact_um > 0:
        before = (s >= seed_s0_nm - include_precontact_um * 1000.0) & (
            s < seed_s0_nm
        )
        inside = inside | before
    return finite & inside & (s <= seed_s0_nm + depth_hi_um * 1000.0)


# ----------------------------------------------------------------------
# nonlinear contact point free Hertz model 
# ----------------------------------------------------------------------
def _hertz(sep, f0, s0, slope):
    return f0 + slope * np.power(np.clip(sep - s0, 0.0, None), 1.5)


def fit_nonlinear(
    p: Prepared,
    *,
    seed_s0_nm: float | None = None,
    seed_slope: float | None = None,
    depth_lo_um: float = FIT_MIN_DEPTH_UM,
    depth_hi_um: float = FIT_MAX_DEPTH_UM,
    include_precontact_um: float = 1.0,
) -> FitOut:
    """Joint least squares for (offset, contact point, modulus).

    ``include_precontact_um`` keeps a stretch of baseline below the contact
    point inside the fit. The model is defined on both sides of contact, so
    baseline points constrain s0 from below instead of being discarded allowing for contact free model
    """
    out = FitOut("nonlinear")
    f, s = p.force_nN, p.s_nm
    good = np.isfinite(f) & np.isfinite(s)
    if int(good.sum()) < 40:
        out.reason = "too_few_points"
        return out
    if seed_s0_nm is None or not np.isfinite(seed_s0_nm):
        line = linearised_line(p, 0.20, 1.00)
        seed_s0_nm = line[1] if line else float(np.percentile(s[good], 30))
        if line and seed_slope is None:
            seed_slope = line[0] ** (-1.5)
    if seed_slope is None or not np.isfinite(seed_slope) or seed_slope <= 0:
        seed_slope = max(p.peak_force_nN, 1.0) / (4000.0**1.5)
    params = np.array([0.0, float(seed_s0_nm), float(seed_slope)])
    span = float(np.nanmax(s) - np.nanmin(s))
    lo_b = [-abs(p.peak_force_nN), float(np.nanmin(s)) - span, 1e-12]
    hi_b = [abs(p.peak_force_nN), float(np.nanmax(s)), 1e6]
    win = seed_window(
        p,
        seed_s0_nm,
        depth_lo_um,
        depth_hi_um,
        include_precontact_um=include_precontact_um,
    )
    if int(win.sum()) < 30:
        out.reason = "window_collapsed"
        out.contact_s0_um = seed_s0_nm / 1000.0
        return out
    try:
        params, _cov = optimize.curve_fit(
            _hertz,
            s[win],
            f[win],
            p0=params,
            bounds=(lo_b, hi_b),
            maxfev=60000,
        )
    except (RuntimeError, ValueError) as exc:
        out.reason = f"nls_failed:{type(exc).__name__}"
        return out
    f0, s0, slope = (float(v) for v in params)
    if slope <= 0:
        out.reason = "nonpositive_slope"
        return out
    depth = s - s0
    predicted = _hertz(s[win], f0, s0, slope)
    out.ok = True
    out.e_star_kPa = e_star_from_slope(slope, p.radius_um)
    out.contact_s0_um = s0 / 1000.0
    out.r_squared = _r2(f[win], predicted)
    out.rmse_nN = float(np.sqrt(np.mean((f[win] - predicted) ** 2)))
    out.n_points = int(win.sum())
    out.depth_lo_um = float(np.nanmin(depth[win])) / 1000.0
    out.depth_hi_um = float(np.nanmax(depth[win])) / 1000.0
    out.extra = {
        "force_offset_nN": f0,
        "hertz_slope_nN_per_nm32": slope,
    }
    return out


# ----------------------------------------------------------------------
# contact-point-free stiffness method
# ----------------------------------------------------------------------
def fit_stiffness(
    p: Prepared,
    *,
    seed_s0_nm: float | None = None,
    depth_lo_um: float = FIT_MIN_DEPTH_UM,
    depth_hi_um: float = FIT_MAX_DEPTH_UM,
    window_s: float = DERIVATIVE_WINDOW_S,
    subtract_noise: bool = True,
    weighted: bool = True,
) -> FitOut:
    """(dF/ds)^2 linear in the raw displacement; E* is derived from the slope
    """
    out = FitOut("stiffness")
    f, s, dt = p.force_nN, p.s_nm, p.dt_s
    good = np.isfinite(f) & np.isfinite(s)
    n_good = int(good.sum())
    if n_good < 60:
        out.reason = "too_few_points"
        return out
    win_pts = max(5, round(window_s / dt))
    if win_pts % 2 == 0:
        win_pts += 1
    if win_pts >= n_good:
        win_pts = max(5, (n_good // 2) * 2 - 1)
    if win_pts < 5 or n_good <= win_pts:
        out.reason = "derivative_window_too_large"
        return out
    try:
        df = signal.savgol_filter(f, win_pts, 2, deriv=1, delta=dt)
        ds = signal.savgol_filter(s, win_pts, 2, deriv=1, delta=dt)
    except ValueError as exc:
        out.reason = f"savgol_failed:{type(exc).__name__}"
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        k = df / ds
    k2 = np.square(k)
    moving = ds > 1000.0  # nm/s; the stage is descending

    floor = 0.0
    if subtract_noise and p.onset_index > 0:
        pre = moving & (np.arange(f.size) < max(p.onset_index - 20, 0))
        if int(pre.sum()) > 40:
            floor = float(np.median(k2[pre]))

    if seed_s0_nm is None or not np.isfinite(seed_s0_nm):
        line = linearised_line(p, 0.20, 1.00)
        if line is None:
            out.reason = "no_seed_window"
            return out
        seed_s0_nm = line[1]
    band = (
        moving
        & seed_window(p, seed_s0_nm, depth_lo_um, depth_hi_um)
        & np.isfinite(k2)
    )
    if int(band.sum()) < 30:
        out.reason = "insufficient_stiffness_points"
        return out
    y = k2[band] - floor
    x = s[band]
    keep = np.isfinite(y) & (y > 0)
    if int(keep.sum()) < 30:
        out.reason = "nonpositive_stiffness"
        return out
    x, y = x[keep], y[keep]
    weights = np.ones_like(y)
    if weighted:
        order = np.argsort(x)
        smooth = np.interp(
            x,
            x[order],
            np.maximum(
                signal.medfilt(y[order], kernel_size=min(51, len(y) | 1)),
                np.nanmedian(y) * 1e-3,
            ),
        )
        weights = 1.0 / np.clip(smooth, 1e-30, None)
    sw = np.sqrt(weights)
    design = np.column_stack([x * sw, sw])
    coef, *_ = np.linalg.lstsq(design, y * sw, rcond=None)
    slope, intercept = float(coef[0]), float(coef[1])
    if slope <= 0:
        out.reason = "nonpositive_stiffness_slope"
        return out
    # (dF/ds)^2 = 4 E*^2 R (s - s0); F in nN, s in nm, R in nm gives E* in
    # nN/nm^2 = 1e6 kPa.
    e_star = math.sqrt(slope / (4.0 * p.radius_um * 1000.0)) * 1.0e6
    s0_nm = -intercept / slope
    predicted = slope * x + intercept
    out.ok = True
    out.e_star_kPa = e_star
    out.contact_s0_um = s0_nm / 1000.0
    out.r_squared = _r2(y, predicted)
    out.rmse_nN = math.nan
    out.n_points = int(x.size)
    out.depth_lo_um = float(x.min() - s0_nm) / 1000.0
    out.depth_hi_um = float(x.max() - s0_nm) / 1000.0
    out.extra = {
        "stiffness_slope": slope,
        "k2_noise_floor": floor,
        "savgol_window_points": win_pts,
        "savgol_window_s": win_pts * dt,
    }
    return out


# ----------------------------------------------------------------------
# bonded-film-corrected Hertz
# ----------------------------------------------------------------------
def fit_film(
    p: Prepared,
    *,
    thickness_um: float = DEFAULT_THICKNESS_UM,
    seed_s0_nm: float | None = None,
    seed_slope: float | None = None,
    depth_lo_um: float = FIT_MIN_DEPTH_UM,
    depth_hi_um: float = FIT_MAX_DEPTH_UM,
    include_precontact_um: float = 1.0,
) -> FitOut:
    """``fit_nonlinear`` with the Garcia film factor applied pointwise."""
    out = FitOut("film")
    f, s = p.force_nN, p.s_nm
    good = np.isfinite(f) & np.isfinite(s)
    if int(good.sum()) < 40:
        out.reason = "too_few_points"
        return out

    def model(sep, f0, s0, slope):
        depth = np.clip(sep - s0, 0.0, None)
        return f0 + slope * np.power(depth, 1.5) * garcia_bec(
            depth, p.radius_um, thickness_um
        )

    if seed_s0_nm is None or not np.isfinite(seed_s0_nm):
        line = linearised_line(p, 0.20, 1.00)
        seed_s0_nm = line[1] if line else float(np.percentile(s[good], 30))
        if line and seed_slope is None:
            seed_slope = line[0] ** (-1.5)
    if seed_slope is None or not np.isfinite(seed_slope) or seed_slope <= 0:
        seed_slope = max(p.peak_force_nN, 1.0) / (4000.0**1.5)
    params = np.array([0.0, float(seed_s0_nm), float(seed_slope)])
    span = float(np.nanmax(s) - np.nanmin(s))
    lo_b = [-abs(p.peak_force_nN), float(np.nanmin(s)) - span, 1e-12]
    hi_b = [abs(p.peak_force_nN), float(np.nanmax(s)), 1e6]
    win = seed_window(
        p,
        seed_s0_nm,
        depth_lo_um,
        depth_hi_um,
        include_precontact_um=include_precontact_um,
    )
    if int(win.sum()) < 30:
        out.reason = "window_collapsed"
        out.contact_s0_um = seed_s0_nm / 1000.0
        return out
    try:
        params, _cov = optimize.curve_fit(
            model,
            s[win],
            f[win],
            p0=params,
            bounds=(lo_b, hi_b),
            maxfev=60000,
        )
    except (RuntimeError, ValueError) as exc:
        out.reason = f"nls_failed:{type(exc).__name__}"
        return out
    f0, s0, slope = (float(v) for v in params)
    if slope <= 0:
        out.reason = "nonpositive_slope"
        return out
    depth = s - s0
    depth_max = float(np.nanmax(depth[win]))
    if not np.isfinite(depth_max) or depth_max <= 0:
        out.reason = "contact_point_beyond_fit_window"
        out.contact_s0_um = s0 / 1000.0
        return out
    predicted = model(s[win], f0, s0, slope)
    out.ok = True
    out.e_star_kPa = e_star_from_slope(slope, p.radius_um)
    out.contact_s0_um = s0 / 1000.0
    out.r_squared = _r2(f[win], predicted)
    out.rmse_nN = float(np.sqrt(np.mean((f[win] - predicted) ** 2)))
    out.n_points = int(win.sum())
    out.depth_lo_um = float(np.nanmin(depth[win])) / 1000.0
    out.depth_hi_um = depth_max / 1000.0
    out.extra = {
        "force_offset_nN": f0,
        "thickness_um": thickness_um,
        "chi_at_fit_max": float(
            math.sqrt(p.radius_um * depth_max / 1000.0) / thickness_um
        ),
        "bec_at_fit_max": float(
            garcia_bec(np.array([depth_max]), p.radius_um, thickness_um)[0]
        ),
    }
    return out


# ----------------------------------------------------------------------
# depth dependence
# ----------------------------------------------------------------------
#: Lower edges of the linearised-plot window ladder, as fractions of peak
#: force. The upper edge is always the peak.
LADDER_LO = (0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70)

#: Indentation bins, in micrometres, at which the pointwise modulus is
#: reported.
POINTWISE_UM = (0.75, 1.50, 2.50, 3.50, 4.00)


def depth_diagnostics(
    p: Prepared,
    *,
    ladder_lo=LADDER_LO,
    pointwise_um=POINTWISE_UM,
    bin_width_um: float = 0.25,
) -> dict:
    """Three measurements of whether the apparent modulus depends on depth
    """
    out: dict = {}
    f, s, dt = p.force_nN, p.s_nm, p.dt_s
    for lo in ladder_lo:
        line = linearised_line(p, lo, 1.00)
        key = f"ladder_E_{lo:.2f}_kPa"
        if line is None or line[0] <= 0:
            out[key] = math.nan
            continue
        out[key] = e_star_from_slope(line[0] ** (-1.5), p.radius_um)

    ref = linearised_line(p, 0.30, 1.00)
    win_pts = max(5, round(0.05 / dt) | 1)
    if ref is None or win_pts >= f.size:
        out.update({f"Ept_{d:.2f}_kPa": math.nan for d in pointwise_um})
        out["k2_slope_ratio"] = math.nan
        return out
    s0_nm = ref[1]
    try:
        df = signal.savgol_filter(f, win_pts, 2, deriv=1, delta=dt)
        ds = signal.savgol_filter(s, win_pts, 2, deriv=1, delta=dt)
    except ValueError:
        out.update({f"Ept_{d:.2f}_kPa": math.nan for d in pointwise_um})
        out["k2_slope_ratio"] = math.nan
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(ds > 1000.0, df / ds, np.nan)
        depth_um = (s - s0_nm) / 1000.0
        # 2a, twice the Hertz contact radius a = sqrt(R delta), in um
        two_a_um = 2.0 * np.sqrt(p.radius_um * np.clip(depth_um, 1e-9, None))
        e_pt = 1.0e6 * k / two_a_um
    for d in pointwise_um:
        m = (depth_um >= d) & (depth_um < d + bin_width_um) & np.isfinite(e_pt)
        out[f"Ept_{d:.2f}_kPa"] = (
            float(np.median(e_pt[m])) if int(m.sum()) >= 5 else math.nan
        )

    # The stiffness plot is fitted in two halves of the same contact region.
    # The noise floor cancels between the two slopes, so it is not removed.
    band = np.isfinite(k) & (depth_um > 0.0) & (depth_um <= FIT_MAX_DEPTH_UM)
    out["k2_slope_ratio"] = math.nan
    if int(band.sum()) >= 60:
        half = 0.5 * FIT_MAX_DEPTH_UM
        lo_m = band & (depth_um <= half)
        hi_m = band & (depth_um > half)
        if int(lo_m.sum()) >= 25 and int(hi_m.sum()) >= 25:
            a_lo = np.polyfit(s[lo_m], np.square(k[lo_m]), 1)[0]
            a_hi = np.polyfit(s[hi_m], np.square(k[hi_m]), 1)[0]
            if a_lo > 0 and a_hi > 0:
                out["k2_slope_ratio"] = float(a_hi / a_lo)
    return out
