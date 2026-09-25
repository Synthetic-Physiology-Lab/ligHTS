# -*- coding: utf-8 -*-
"""Shared measurement kernel for the LigHTS tri-modal groove calibration.

One estimator is used for every modality and for every dataset, so that the
constants derived on the calibration sample can be applied to the plates
without an estimator mismatch.  The chain is:

  1. orient   - find the grating normal by FFT, rotate so the grooves are
                vertical, then refine the angle by maximising the fundamental
                amplitude of the column-collapsed profile (sub-0.05 deg)
  2. collapse - median across the groove direction (phase-locked once the
                residual tilt is below the refinement step)
  3. detrend  - remove a quadratic in the groove-normal coordinate
  4. estimate - least-squares fit of a 6-harmonic periodic waveform at the
                refined pitch; the depth is the peak-to-trough of the fitted
                waveform

Step 4 is unbiased under additive noise (the noise projects onto the residual,
not onto the harmonic coefficients), unlike a raw peak-to-trough, which is
inflated by roughly 2.5 sigma / sqrt(n_independent) and therefore depends on
the noise level of the instrument rather than on the specimen.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import rotate as ndrotate

K_HARMONICS = 6
PITCH_SEARCH_SPAN = 0.15
PITCH_SEARCH_N = 241


# ------------------------------------------------------------------ harmonics
def design(coord: np.ndarray, pitch: float, K: int = K_HARMONICS) -> np.ndarray:
    cols = [np.ones_like(coord)]
    for k in range(1, K + 1):
        cols += [np.cos(2 * np.pi * k * coord / pitch),
                 np.sin(2 * np.pi * k * coord / pitch)]
    return np.column_stack(cols)


def fit_harmonic(coord, value, pitch, K=K_HARMONICS, extra=None):
    ok = np.isfinite(value) & np.isfinite(coord)
    A = design(coord[ok], pitch, K)
    if extra is not None:
        A = np.hstack([A, extra[ok]])
    beta, *_ = np.linalg.lstsq(A, value[ok], rcond=None)
    return beta


def waveform(beta, pitch, K=K_HARMONICS, n=1024):
    t = np.linspace(0.0, pitch, n, endpoint=False)
    y = np.full(n, beta[0])
    for k in range(1, K + 1):
        y += beta[2 * k - 1] * np.cos(2 * np.pi * k * t / pitch)
        y += beta[2 * k] * np.sin(2 * np.pi * k * t / pitch)
    return t, y


def fundamental(beta) -> float:
    return float(np.hypot(beta[1], beta[2]))


def refine_pitch(coord, value, p0, K=K_HARMONICS, extra=None,
                 span=PITCH_SEARCH_SPAN, n=PITCH_SEARCH_N):
    ok = np.isfinite(value) & np.isfinite(coord)
    c, v = coord[ok], value[ok]
    e = None if extra is None else extra[ok]
    best_p, best_r = p0, np.inf
    for p in np.linspace(p0 * (1 - span), p0 * (1 + span), n):
        A = design(c, p, K)
        if e is not None:
            A = np.hstack([A, e])
        beta, *_ = np.linalg.lstsq(A, v, rcond=None)
        r = float(np.sum((v - A @ beta) ** 2))
        if r < best_r:
            best_p, best_r = p, r
    return best_p


# -------------------------------------------------------------- orientation
def grating_normal_deg(img: np.ndarray, px_um: float = 1.0,
                       pitch_lo: float = 20.0, pitch_hi: float = 120.0) -> float:
    """Angle of the grating normal from the dominant 2-D FFT peak.

    The radial-power version used by the historical converter integrates over
    all spatial frequencies and is pulled several degrees off by low-frequency
    illumination structure.  Locating the single fundamental peak instead, and
    refining it with an intensity centroid over its 3x3 neighbourhood, is
    accurate to a few tenths of a degree.
    """
    img = np.nan_to_num(img.astype(np.float64), nan=float(np.nanmean(img)))
    h, w = img.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    F = np.fft.fftshift(np.fft.fft2((img - img.mean()) * win))
    P = np.abs(F)
    cy, cx = h // 2, w // 2
    fy = (np.arange(h) - cy) / (h * px_um)
    fx = (np.arange(w) - cx) / (w * px_um)
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    R = np.hypot(FY, FX)
    with np.errstate(divide="ignore"):
        period = np.where(R > 0, 1.0 / R, np.inf)
    mask = (period >= pitch_lo) & (period <= pitch_hi)
    if not mask.any():
        return 0.0
    Pm = np.where(mask, P, 0.0)
    iy, ix = np.unravel_index(int(np.argmax(Pm)), Pm.shape)
    # centroid refinement over the 3x3 neighbourhood of the peak
    y0, y1 = max(iy - 1, 0), min(iy + 2, h)
    x0, x1 = max(ix - 1, 0), min(ix + 2, w)
    blk = P[y0:y1, x0:x1]
    wy = np.arange(y0, y1)[:, None] * np.ones((1, x1 - x0))
    wx = np.ones((y1 - y0, 1)) * np.arange(x0, x1)[None, :]
    sy = float((blk * wy).sum() / blk.sum())
    sx = float((blk * wx).sum() / blk.sum())
    ang = np.degrees(np.arctan2(sy - cy, sx - cx))
    return float(((ang + 90.0) % 180.0) - 90.0)      # fold to (-90, 90]


def collapse(img, trim_frac=0.10):
    """Collapse along the groove direction with a symmetric trimmed mean.

    A plain median is robust but, for a square-wave-like relief, almost blind to
    residual tilt: the majority sign survives a large phase smear, so it cannot
    be used as the sharpness metric for the angle search.  A trimmed mean keeps
    most of the outlier resistance while staying linear enough to lose amplitude
    when the grooves are not vertical.
    """
    a = np.asarray(img, float)
    n = a.shape[0]
    k = int(np.floor(trim_frac * n))
    a = np.sort(a, axis=0)                      # NaNs sort to the end
    valid = np.sum(np.isfinite(a), axis=0)
    out = np.full(a.shape[1], np.nan)
    for j in range(a.shape[1]):
        v = a[:valid[j], j]
        if v.size < 8:
            continue
        kk = int(np.floor(trim_frac * v.size))
        out[j] = np.mean(v[kk:v.size - kk]) if v.size - 2 * kk > 0 else np.mean(v)
    return out


def _profile_contrast(img, px, p0, n_pitch=21, span=0.04):
    """Fundamental amplitude of the column-median profile - the sharpness metric.

    The pitch is re-optimised over a narrow window at every trial angle, because
    a residual tilt and a pitch error are partly degenerate; leaving the pitch
    fixed would let the fit absorb the tilt and flatten the maximum.
    """
    prof = collapse(img)
    ok = np.isfinite(prof)
    if ok.sum() < 32:
        return -np.inf
    x = np.arange(prof.size) * px
    extra = np.column_stack([x, x ** 2])
    best = -np.inf
    for p in np.linspace(p0 * (1 - span), p0 * (1 + span), n_pitch):
        beta = fit_harmonic(x[ok], prof[ok], p, K=1, extra=extra[ok])
        best = max(best, fundamental(beta))
    return best


def _downsample(img, factor):
    if factor <= 1:
        return img, 1
    h = (img.shape[0] // factor) * factor
    w = (img.shape[1] // factor) * factor
    blk = img[:h, :w].reshape(h // factor, factor, w // factor, factor)
    return np.nanmean(blk, axis=(1, 3)), factor


def _trim_frac(angle_deg, base=0.03, k=1.2):
    """Border to discard after rotating by `angle_deg`.

    `mode="nearest"` fills the wedges left by the rotation with replicated edge
    values, which are not data.  The invalid width scales with |sin(angle)|, so
    the trim is tied to the angle actually applied rather than fixed: a field
    that needed no rotation keeps almost all of its periods, which matters at
    the 80 um pitch where a fixed 15 % trim leaves fewer than five.
    """
    return float(base + k * abs(np.sin(np.radians(angle_deg))))


def orient(img: np.ndarray, px_um: float, pitch_guess: float = 40.0,
           fine_range: float = 3.0, fine_step: float = 0.05, trim: float = None,
           search_px: int = 280):
    """Rotate so the grooves are vertical; refine on profile contrast.

    The angle search runs on a copy binned down to about `search_px` pixels,
    which leaves >10 samples per groove period and makes the scan ~20x cheaper;
    the chosen angle is then applied once to the full-resolution field.

    Returns (rotated_and_trimmed_image, total_angle_deg, refined_pitch_um).
    """
    coarse = grating_normal_deg(img, px_um)

    fill = float(np.nanmedian(img))
    factor = max(1, int(min(img.shape) // search_px))
    small, factor = _downsample(np.nan_to_num(img, nan=fill), factor)
    small_px = px_um * factor

    def rot(src, a, tf):
        """Rotate and crop.  `tf` is held constant across a scan so that every
        candidate angle is scored on the same region of the field."""
        r = ndrotate(src, -a, reshape=False, order=1, mode="nearest")
        c = int(tf * r.shape[0])
        return r[c:r.shape[0] - c, c:r.shape[1] - c] if c else r

    # coarse-to-fine on the contrast metric, guarding against an edge maximum
    angles = np.arange(coarse - fine_range, coarse + fine_range + 1e-9, fine_step)
    for _ in range(5):
        tf = trim if trim is not None else _trim_frac(np.abs(angles).max())
        scores = np.array([_profile_contrast(rot(small, a, tf), small_px, pitch_guess)
                           for a in angles])
        best = float(angles[int(np.argmax(scores))])
        if abs(best - angles[0]) > 1e-9 and abs(best - angles[-1]) > 1e-9:
            break                                    # interior maximum, done
        angles = np.arange(best - fine_range, best + fine_range + 1e-9, fine_step)

    out = rot(np.nan_to_num(img, nan=fill), best,
              trim if trim is not None else _trim_frac(best))
    x = np.arange(out.shape[1]) * px_um
    prof = collapse(out)
    pitch = refine_pitch(x, prof, pitch_guess, K=1,
                         extra=np.column_stack([x, x ** 2]))
    return out, best, pitch


# ------------------------------------------------------------------ measure
def measure_field(img: np.ndarray, px_um: float, pitch_guess: float = 40.0,
                  do_orient: bool = True, residual_scan: bool = True):
    """Full chain on one oriented or unoriented field.

    Returns a dict with the refined pitch, the harmonic depth (peak-to-trough of
    the fitted waveform), the fundamental peak-to-trough, the raw peak-to-trough
    of the detrended profile, the applied angle and the profile itself.
    """
    angle = 0.0
    if do_orient:
        img, angle, pitch_guess = orient(img, px_um, pitch_guess)
    elif residual_scan:
        # the field is already nominally vertical: scan only a narrow residual
        best_s, best_a, best_img = -np.inf, 0.0, img
        for a in np.arange(-0.6, 0.601, 0.05):
            r = ndrotate(np.nan_to_num(img, nan=float(np.nanmedian(img))),
                         -a, reshape=False, order=1, mode="nearest")
            c = int(0.08 * r.shape[0])
            r = r[c:r.shape[0] - c, c:r.shape[1] - c]
            s = _profile_contrast(r, px_um, pitch_guess)
            if s > best_s:
                best_s, best_a, best_img = s, float(a), r
        img, angle = best_img, best_a

    rows = img - np.nanmedian(img, axis=1, keepdims=True)   # per-row offset
    prof = collapse(rows)                                    # phase-locked collapse
    x = np.arange(prof.size) * px_um
    extra = np.column_stack([x, x ** 2])
    pitch = refine_pitch(x, prof, pitch_guess, K=1, extra=extra)
    b1 = fit_harmonic(x, prof, pitch, K=1, extra=extra)
    bK = fit_harmonic(x, prof, pitch, K=K_HARMONICS, extra=extra)
    t, w = waveform(bK, pitch)
    flat = prof - np.polyval(np.polyfit(x, prof, 2), x)
    resid = flat - np.interp(np.mod(x, pitch), t, w - w.mean())
    return dict(pitch_um=float(pitch),
                depth=float(np.ptp(w)),
                depth_fundamental=float(2 * fundamental(b1)),
                depth_rawptp=float(np.ptp(flat)),
                angle_deg=float(angle),
                residual_rms=float(np.std(resid)),
                n_periods=float(x[-1] / pitch),
                x=x, profile=flat, wave_t=t, wave=w, image=img)


def measure_scatter(coord_normal, coord_along, value, pitch_guess=40.0,
                    shear_range=0.12, shear_step=0.004):
    """Same estimator for scattered points (the nanoindentation grid).

    A shear y' = y + s*x replaces the image rotation, so the grooves become
    exactly perpendicular to the collapse axis.  Per-column offsets are carried
    as nuisance regressors instead of a per-row median.
    """
    cn = np.asarray(coord_normal, float)
    ca = np.asarray(coord_along, float)
    v = np.asarray(value, float)
    cols = np.unique(ca)
    E = np.zeros((v.size, cols.size - 1))
    for j, c in enumerate(cols[1:]):
        E[ca == c, j] = 1.0

    best = (0.0, np.inf, pitch_guess, None)
    for s in np.arange(-shear_range, shear_range + 1e-9, shear_step):
        u = cn + s * ca
        p = refine_pitch(u, v, pitch_guess, K=1, extra=E, span=0.12, n=121)
        A = np.hstack([design(u, p, 1), E])
        beta, *_ = np.linalg.lstsq(A, v, rcond=None)
        r = float(np.sum((v - A @ beta) ** 2))
        if r < best[1]:
            best = (float(s), r, float(p), u)
    shear, _, pitch, u = best
    pitch = refine_pitch(u, v, pitch, K=K_HARMONICS, extra=E, span=0.06, n=121)
    bK = fit_harmonic(u, v, pitch, K=K_HARMONICS, extra=E)
    b1 = fit_harmonic(u, v, pitch, K=1, extra=E)
    t, w = waveform(bK, pitch)
    A = np.hstack([design(u, pitch, K_HARMONICS), E])
    noise = float(np.std(v - A @ bK))
    return dict(pitch_um=pitch, depth=float(np.ptp(w)),
                depth_fundamental=float(2 * fundamental(b1)),
                shear=shear, tilt_deg=float(np.degrees(np.arctan(shear))),
                noise_sd=noise, u=u, wave_t=t, wave=w, beta=bK, nuisance=E)


def bootstrap_depth(coord, value, pitch, groups, n_boot=300, seed=0):
    """Depth uncertainty by resampling independent groups (columns or fields)."""
    rng = np.random.default_rng(seed)
    g = np.asarray(groups)
    uniq = np.unique(g)
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, uniq.size, replace=True)
        idx = np.concatenate([np.where(g == u)[0] for u in pick])
        beta = fit_harmonic(coord[idx], value[idx], pitch)
        out.append(np.ptp(waveform(beta, pitch)[1]))
    return float(np.std(out)), (float(np.percentile(out, 2.5)),
                                float(np.percentile(out, 97.5)))
