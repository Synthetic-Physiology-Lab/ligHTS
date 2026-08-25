"""Per-curve mechanical quality control, with a reason code for every failure.

This module decides, before any modulus is compared with any other, whether a
given force-displacement curve is a measurement of a gel at all. A curve that
never reached the sample, that snapped into it, that had no baseline to speak
of, or that pushed 20 uN into something rigid is not a soft-gel indentation,
and averaging it with ones that are inflates the within-gel scatter that the
same table then reports as a property of the material.

QC failure modes
----------------
``no_loading_branch``      the file has no usable loading segment
``fit_failed``             the estimator did not converge
``no_contact``             force never rises convincingly above the baseline
``starts_in_contact``      already loaded when recording starts: no baseline
``short_pre_contact``      too little pre-contact record to fit a baseline
``contact_outside_travel`` fitted contact point is not inside recorded travel
``shallow_indentation``    the curve does not reach the analysis depth cap
``overshoot``              indentation far beyond what the profile commands
``implausible_modulus``    outside 0.1-100 kPa: not a hydrated GelMA gel
``discontinuity``          a step in force far larger than the local increment
``nonmonotonic``           the loading branch genuinely reverses in the window
``plateau``                a stretch of near-zero stiffness inside contact
``baseline_unstable``      the baseline model does not extrapolate to contact
``baseline_drift``         gross drift: the measurement was not settled
``oscillation``            ferrule-top wobble too slow, too large to average
``poor_fit``               residual so large the curve is not a contact
"""

from __future__ import annotations

import math

import numpy as np
from scipy import signal

__all__ = [
    "QC_ORDER",
    "THRESHOLDS",
    "contact_equivalent_um",
    "qc_curve",
]


THRESHOLDS = {
    # baseline
    "min_pre_duration_s": 0.20,
    "min_pre_points": 20,
    "max_start_force_over_sigma": 10.0,
    "max_baseline_extrap_contact_um": 1.00,
    "max_drift_contact_um": 2.00,
    "max_oscillation_contact_um": 1.00,
    # geometry
    "min_depth_um": 3.5,
    "max_depth_um": 10.0,
    "contact_margin_um": 0.0,
    # shape
    "max_backstep_frac": 0.10,
    "max_backstep_over_peak": 0.05,
    "max_step_over_sigma": 8.0,
    "max_plateau_span_um": 0.50,
    # physics and fit
    "min_e_star_kPa": 0.1,
    "max_e_star_kPa": 100.0,
    "max_nrmse_percent": 8.0,
    "min_peak_snr": 20.0,
}

QC_ORDER = (
    "no_loading_branch",
    "fit_failed",
    "no_contact",
    "starts_in_contact",
    "short_pre_contact",
    "contact_outside_travel",
    "shallow_indentation",
    "overshoot",
    "implausible_modulus",
    "discontinuity",
    "nonmonotonic",
    "plateau",
    "baseline_unstable",
    "baseline_drift",
    "oscillation",
    "poor_fit",
)


def contact_equivalent_um(
    force_nN: float, e_star_kPa: float, radius_um: float
) -> float:
    """Indentation at which Hertz would produce ``force_nN``.
    """
    if (
        not np.isfinite(force_nN)
        or force_nN <= 0
        or not np.isfinite(e_star_kPa)
        or e_star_kPa <= 0
        or radius_um <= 0
    ):
        return math.nan
    f_n = force_nN * 1e-9
    e_pa = e_star_kPa * 1e3
    r_m = radius_um * 1e-6
    delta_m = (3.0 * f_n / (4.0 * e_pa * math.sqrt(r_m))) ** (2.0 / 3.0)
    return delta_m * 1e6


def _shape_metrics(
    p, s0_nm: float, cap_um: float, hertz_slope: float = math.nan
) -> dict:
    """Monotonicity, steps and plateaux, measured inside the fit window.
    """
    out = {
        "frac_backsteps": math.nan,
        "max_backstep_nN": math.nan,
        "max_step_over_sigma": math.nan,
        "max_plateau_span_um": math.nan,
        "n_fit_window": 0,
    }
    f, s, dt = p.force_nN, p.s_nm, p.dt_s
    depth = s - s0_nm
    # Start 0.3 um below the fitted contact point

    win = (depth > -300.0) & (depth <= cap_um * 1000.0) & np.isfinite(f)
    n = int(win.sum())
    out["n_fit_window"] = n
    if n < 40:
        return out
    fw, sw, dw = f[win], s[win], depth[win]
    order = np.argsort(sw)
    fw, sw, dw = fw[order], sw[order], dw[order]
    span = max(5, round(0.02 / dt) | 1)
    if span >= fw.size:
        span = max(5, (fw.size // 2) * 2 - 1)
    fs = signal.savgol_filter(fw, span, 2)
    diff = np.diff(fs)
    out["frac_backsteps"] = float(np.mean(diff < 0))
    out["max_backstep_nN"] = float(max(0.0, -np.min(diff)))

    # step detector
    smooth_span = max(7, round(0.04 / dt) | 1)
    if smooth_span < fw.size:
        local = signal.savgol_filter(fw, smooth_span, 2)
        edge = smooth_span
        resid = np.abs(fw - local)[edge:-edge] if fw.size > 2 * edge else None
        if resid is not None and resid.size > 10 and p.sigma_nN > 0:
            out["max_step_over_sigma"] = float(np.nanmax(resid) / p.sigma_nN)

    # plateau: stiffness below the expected Hertz stiffness
    if not np.isfinite(hertz_slope) or hertz_slope <= 0:
        positive = dw > 0
        if int(positive.sum()) > 10:
            hertz_slope = float(
                np.nanmedian(fw[positive] / np.power(dw[positive], 1.5))
            )
    if np.isfinite(hertz_slope) and hertz_slope > 0:
        bin_nm = 100.0
        edges = np.arange(sw[0], sw[-1] + bin_nm, bin_nm)
        idx = np.digitize(sw, edges) - 1
        best, run = 0.0, 0.0
        for b in range(edges.size - 1):
            m = idx == b
            if int(m.sum()) < 4:
                run = 0.0
                continue
            width = float(sw[m][-1] - sw[m][0])
            if width <= 0:
                run = 0.0
                continue
            measured = float(fs[m][-1] - fs[m][0]) / width
            d_mid = max(float(np.mean(dw[m])), 1.0)
            expected = 1.5 * hertz_slope * math.sqrt(d_mid)
            if measured < 0.25 * expected:
                run += width
                best = max(best, run)
            else:
                run = 0.0
        out["max_plateau_span_um"] = float(best / 1000.0)
    return out


def qc_curve(
    p,
    fit,
    *,
    thresholds: dict | None = None,
    cap_um: float = 3.5,
) -> dict:
    """Run every criterion on one prepared curve and its pipeline fit.

    ``p`` is a ``lights_fit.Prepared`` (or None) and ``fit`` the ``FitOut``
    from ``lights_fit.fit_pipeline``. Returns a flat dictionary of metrics
    plus ``qc_pass`` and ``qc_reasons``
    """
    th = dict(THRESHOLDS)
    if thresholds:
        th.update(thresholds)
    m: dict = {"qc_pass": False, "qc_reasons": ""}
    fail: set = set()

    if p is None:
        m["qc_reasons"] = "no_loading_branch"
        return m

    e_star = fit.e_star_kPa if fit.ok else math.nan
    radius = p.radius_um

    e_price = e_star
    if not np.isfinite(e_price) or e_price <= 0:
        e_price = max(
            0.75
            * (10.0**7.5)
            * (p.peak_force_nN * 1e-3 / 4000.0**1.5)
            / math.sqrt(radius),
            0.2,
        )

    extras = getattr(p, "extras", {}) or {}
    m["e_star_kPa"] = e_star
    m["peak_force_nN"] = p.peak_force_nN
    m["sigma_nN"] = p.sigma_nN
    m["peak_snr"] = (
        p.peak_force_nN / p.sigma_nN if p.sigma_nN > 0 else math.nan
    )
    m["baseline_model"] = p.baseline_model
    m["pre_duration_s"] = p.pre_duration_s
    m["pre_n"] = p.pre_n
    m["pre_resid_rms_nN"] = p.pre_resid_rms_nN
    m["pre_span_nN"] = p.pre_span_nN
    m["drift_nN_per_s"] = p.baseline_slope_nN_per_s
    m["fit_duration_s"] = extras.get("fit_duration_s", math.nan)
    m["baseline_extrap_nN"] = extras.get("baseline_extrap_nN", math.nan)
    m["osc_amplitude_nN"] = extras.get("osc_amplitude_nN", math.nan)
    m["osc_frequency_hz"] = extras.get("osc_frequency_hz", math.nan)
    m["osc_effective_nN"] = extras.get("osc_effective_nN", math.nan)
    m["drift_over_loading_nN"] = abs(p.baseline_slope_nN_per_s) * (
        m["fit_duration_s"] if np.isfinite(m["fit_duration_s"]) else 0.0
    )
    m["noise_contact_um"] = contact_equivalent_um(
        3.0 * p.sigma_nN, e_price, radius
    )
    m["baseline_extrap_contact_um"] = contact_equivalent_um(
        m["baseline_extrap_nN"], e_price, radius
    )
    m["drift_contact_um"] = contact_equivalent_um(
        m["drift_over_loading_nN"], e_price, radius
    )
    m["oscillation_contact_um"] = contact_equivalent_um(
        m["osc_effective_nN"], e_price, radius
    )
    # Propagate rather than exclude: for Hertz at fixed force,
    # d(ln E*)/d(s0) = 1.5/delta, so the contact-point uncertainty converts
    # directly into a relative modulus uncertainty at the analysis cap.
    parts = [
        m["baseline_extrap_contact_um"],
        m["oscillation_contact_um"],
    ]
    good = [v for v in parts if np.isfinite(v)]
    m["s0_uncertainty_um"] = (
        float(np.sqrt(np.sum(np.square(good)))) if good else math.nan
    )
    m["e_star_rel_sigma"] = (
        1.5 * m["s0_uncertainty_um"] / cap_um
        if np.isfinite(m["s0_uncertainty_um"])
        else math.nan
    )

    if p.pre_n < th["min_pre_points"] or not (
        p.pre_duration_s >= th["min_pre_duration_s"]
    ):
        fail.add("short_pre_contact")
    head = min(max(5, round(0.05 / p.dt_s)), p.force_nN.size)
    start_force = float(np.median(p.force_nN[:head]))
    m["start_force_over_sigma"] = (
        start_force / p.sigma_nN if p.sigma_nN > 0 else math.nan
    )
    if (
        np.isfinite(m["start_force_over_sigma"])
        and m["start_force_over_sigma"] > th["max_start_force_over_sigma"]
    ):
        fail.add("starts_in_contact")
    if p.onset_index < 0:
        fail.add("no_contact")
    if np.isfinite(m["peak_snr"]) and m["peak_snr"] < th["min_peak_snr"]:
        fail.add("no_contact")

    if not fit.ok:
        fail.add("fit_failed")
        m["contact_s0_um"] = fit.contact_s0_um
        m["depth_max_um"] = fit.extra.get("depth_reached_um", math.nan)
        m["nrmse_percent"] = math.nan
    else:
        s0_nm = fit.contact_s0_um * 1000.0
        m["contact_s0_um"] = fit.contact_s0_um
        m["seed_s0_um"] = fit.extra.get("seed_s0_um", math.nan)
        m["depth_max_um"] = fit.extra.get(
            "depth_reached_um", float(np.nanmax(p.s_nm) - s0_nm) / 1000.0
        )
        m["nrmse_percent"] = fit.extra.get("nrmse_percent", math.nan)
        lo = float(np.nanmin(p.s_nm)) + th["contact_margin_um"] * 1000.0
        hi = float(np.nanmax(p.s_nm))
        if not (lo <= s0_nm <= hi):
            fail.add("contact_outside_travel")
        if not np.isfinite(m["depth_max_um"]) or m["depth_max_um"] <= 0:
            fail.add("no_contact")
        elif m["depth_max_um"] < th["min_depth_um"]:
            fail.add("shallow_indentation")
        elif m["depth_max_um"] > th["max_depth_um"]:
            fail.add("overshoot")
        if not np.isfinite(e_star) or not (
            th["min_e_star_kPa"] <= e_star <= th["max_e_star_kPa"]
        ):
            fail.add("implausible_modulus")
        if (
            np.isfinite(m["nrmse_percent"])
            and m["nrmse_percent"] > th["max_nrmse_percent"]
        ):
            fail.add("poor_fit")
        shape = _shape_metrics(
            p,
            s0_nm,
            cap_um,
            fit.extra.get("hertz_slope_nN_per_nm32", math.nan),
        )
        m.update(shape)
        if (
            np.isfinite(shape["frac_backsteps"])
            and shape["frac_backsteps"] > th["max_backstep_frac"]
            and np.isfinite(shape["max_backstep_nN"])
            and shape["max_backstep_nN"]
            > th["max_backstep_over_peak"] * p.peak_force_nN
        ):
            fail.add("nonmonotonic")
        if (
            np.isfinite(shape["max_step_over_sigma"])
            and shape["max_step_over_sigma"] > th["max_step_over_sigma"]
        ):
            fail.add("discontinuity")
        if (
            np.isfinite(shape["max_plateau_span_um"])
            and shape["max_plateau_span_um"] > th["max_plateau_span_um"]
        ):
            fail.add("plateau")

    if (
        np.isfinite(m["baseline_extrap_contact_um"])
        and m["baseline_extrap_contact_um"]
        > th["max_baseline_extrap_contact_um"]
    ):
        fail.add("baseline_unstable")
    if (
        np.isfinite(m["drift_contact_um"])
        and m["drift_contact_um"] > th["max_drift_contact_um"]
    ):
        fail.add("baseline_drift")
    if (
        np.isfinite(m["oscillation_contact_um"])
        and m["oscillation_contact_um"] > th["max_oscillation_contact_um"]
    ):
        fail.add("oscillation")

    m["qc_reasons"] = ";".join(r for r in QC_ORDER if r in fail)
    m["qc_pass"] = not fail
    return m
