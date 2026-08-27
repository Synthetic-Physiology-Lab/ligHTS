"""Groove Analyzer
Analyzes sinusoidal groove geometry from 3D TIFF volumes using ISO-inspired
Gaussian filtering and per-groove pitch/depth extraction with GUM uncertainty.
"""
from __future__ import annotations

import argparse
import hashlib
import csv
import datetime as dt
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from numpy.linalg import lstsq
from scipy.ndimage import gaussian_filter1d, generic_filter, median_filter, rotate
from scipy.signal import find_peaks

try:
    import tifffile
except ImportError:  # pragma: no cover
    tifffile = None

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog
except ImportError:  # pragma: no cover
    tk = None
    filedialog = None
    messagebox = None
    simpledialog = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

__version__ = "7.3.1"
ANALYSIS_NAME = "grooves_geometry"
ANALYSIS_VERSION = "2026.01.27-v7.3.1"
LOGGER = logging.getLogger("groove_analyzer")

""" Constants fot correction of bias on groove depth due to PSF blurring and Gaussian smoothing
Values estimated on a synthethic dataset of N = 240 samples"""
_DEPTH_BIAS_PITCH_UM = np.asarray([20.0, 40.0, 60.0, 80.0], dtype=float)
_DEPTH_BIAS_MULT = np.asarray([1.1635, 1.0442, 1.0221, 1.0171], dtype=float)

@dataclass(frozen=True)
class AnalyzerConfig:
    """Analysis configuration (units are µm unless stated otherwise)."""

    pitch_range_um: tuple[float, float] = (15.0, 100.0)

    min_slices: int = 5
    confidence_frac: float = 0.02

    lambda_c_factor: float = 0.8
    pad_mode: str = "mirror"

    prominence_frac: float = 0.03
    min_peak_distance_frac: float = 0.45

    tukey_k: float = 1.5
    u_rel_calibration: float = 0.012

    qc_min_valid_frac: float = 0.50
    qc_min_grooves: int = 6

    dc_exclusion_radius_px: int = 5
    fft_use_hann: bool = True

    generate_plots: bool = True
    plot_dpi: int = 300

CONFIG = AnalyzerConfig()


@dataclass(frozen=True)
class MeasurementResult:
    """Scalar estimate with GUM-style uncertainties and sampling metadata."""
    value: float
    u_a: float = float("nan")
    u_b: float = float("nan")
    n: int = 0
    dof: int = 0
    std: float = float("nan")

    @property
    def u_combined(self) -> float:
        """Return combined standard uncertainty sqrt(u_a² + u_b²)."""
        if not (math.isfinite(self.u_a) and math.isfinite(self.u_b)):
            return float("nan")
        return math.hypot(self.u_a, self.u_b)


def depth_correction_multiplier(pitch_um: float) -> float:
    """Return depth correction multiplier from pitch (µm).
    Uses linear interpolation across the empirical table and clamps outside
    [20, 80] µm. If pitch is non-finite, returns 1"""

    if not math.isfinite(pitch_um):
        return 1.0
    return float(
        np.interp(
            float(pitch_um),
            _DEPTH_BIAS_PITCH_UM,
            _DEPTH_BIAS_MULT,
            left=float(_DEPTH_BIAS_MULT[0]),
            right=float(_DEPTH_BIAS_MULT[-1]),
        )
    )


def scale_measurement(m: MeasurementResult, factor: float) -> MeasurementResult:
    """Scale uncertainties and std by the bias correction factor."""
    if not math.isfinite(factor):
        factor = 1.0
    a = abs(float(factor))
    return MeasurementResult(
        value=float(m.value) * float(factor),
        u_a=float(m.u_a) * a,
        u_b=float(m.u_b) * a,
        n=int(m.n),
        dof=int(m.dof),
        std=float(m.std) * a,
    )


def now_utc() -> str:
    """Return an ISO-8601 UTC timestamp like 'YYYY-MM-DDTHH:MM:SSZ'."""
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def sha256_file(path: str) -> str:
    """Compute SHA-256 hex digest for a file at `path` (streamed, constant memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def round_for_csv(value: Any, decimals: int = 2) -> Any:
    """Round finite floats for CSV output; return non-floats unchanged."""
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return value
        return round(float(value), decimals)
    return value


def type_b_resolution(delta: float) -> float:
    """Return Type-B uncertainty for a rectangular distribution (Δ/√12)."""
    return abs(delta) / math.sqrt(12.0)


def iso16610_sigma_um(lambda_c_um: float) -> float:
    """Return ISO 16610-21 equivalent Gaussian sigma for a cutoff λc."""
    return lambda_c_um * math.sqrt(math.log(2.0)) / (math.pi * math.sqrt(2.0))


def iso16610_sigma_px(lambda_c_um: float, xy_um: float) -> float:
    """Return ISO 16610-21 sigma in pixels for cutoff λc and pixel size."""
    return iso16610_sigma_um(lambda_c_um) / xy_um


def parabolic_subpixel_offset_3(values: np.ndarray) -> float:
    """Return sub-pixel offset from a 3-point parabola fit centered at 0."""
    if values.shape[0] != 3:
        return 0.0
    denom = float(values[0] - 2.0 * values[1] + values[2])
    if denom == 0.0:
        return 0.0
    return 0.5 * float(values[0] - values[2]) / denom


def nanmedian_window(values: np.ndarray) -> float:
    """Return the median of finite values in a generic_filter window."""
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def interpolate_finite_1d(arr: np.ndarray) -> np.ndarray:
    """Return a copy with NaNs filled by linear interpolation across finite points."""
    out = np.asarray(arr, dtype=float).copy()
    x = np.arange(out.size, dtype=float)
    good = np.isfinite(out)
    if not np.any(good):
        return out
    if np.all(good):
        return out
    out[~good] = np.interp(x[~good], x[good], out[good])
    return out


def baseline_to_zero(profile: np.ndarray) -> np.ndarray:
    """Return profile shifted so its finite minimum is 0 (or unchanged if none)."""
    out = np.asarray(profile, dtype=float).copy()
    finite = np.isfinite(out)
    if not np.any(finite):
        return out
    out -= float(np.nanmin(out[finite]))
    return out


def columnwise_nanmedian(arr2d: np.ndarray) -> np.ndarray:
    """Return nanmedian along axis=0 without warnings from all-NaN columns."""
    arr = np.asarray(arr2d, dtype=float)
    finite_any = np.isfinite(arr).any(axis=0)
    out = np.full(arr.shape[1], np.nan, dtype=float)
    if not np.any(finite_any):
        return out
    out[finite_any] = np.nanmedian(arr[:, finite_any], axis=0)
    return out


def nanmean_axis0_no_warn(arr2d: np.ndarray) -> np.ndarray:
    """Return nanmean along axis=0 without "empty slice" warnings."""
    a = np.asarray(arr2d, dtype=float)
    mask = np.isfinite(a)
    num = np.where(mask, a, 0.0).sum(axis=0)
    den = mask.sum(axis=0)
    out = np.full(a.shape[1], np.nan, dtype=float)
    good = den > 0
    out[good] = num[good] / den[good]
    return out

def tukey_fence_mask(arr: np.ndarray, k: float = 1.5) -> np.ndarray:
    """Return mask selecting values within Tukey fences based on finite entries."""
    a = np.asarray(arr, dtype=float)
    valid = np.isfinite(a)
    if not np.any(valid):
        return np.zeros_like(a, dtype=bool)
    q1, q3 = np.nanpercentile(a[valid], [25.0, 75.0])
    iqr = float(q3 - q1)
    if not (math.isfinite(iqr) and iqr > 0.0):
        return valid
    lo, hi = float(q1 - k * iqr), float(q3 + k * iqr)
    return valid & (a >= lo) & (a <= hi)


def mad_clean_mask(arr: np.ndarray, k: float = 3.5) -> np.ndarray:
    """Return mask selecting values within k scaled MAD of the median."""
    a = np.asarray(arr, dtype=float)
    valid = np.isfinite(a)
    if not np.any(valid):
        return np.zeros_like(a, dtype=bool)
    med = float(np.nanmedian(a[valid]))
    mad = float(np.nanmedian(np.abs(a[valid] - med)))
    if mad <= 0.0:
        return valid
    threshold = k * mad
    return valid & (np.abs(a - med) <= threshold)


def load_volume(path: Path) -> np.ndarray:
    """Load a 3D TIFF stack and return float32 volume of shape (z, y, x)."""
    if tifffile is None:
        raise ImportError("tifffile is required: pip install tifffile")
    vol = tifffile.imread(str(path))
    if vol.ndim == 2:
        vol = vol[np.newaxis, ...]
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D TIFF (z,y,x), got shape {vol.shape}")
    return vol.astype(np.float32, copy=False)


def reconstruct_height_map(
    volume: np.ndarray,
    dz_um: float,
    z0_um: float = 0.0,
    config: AnalyzerConfig = CONFIG,
) -> tuple[np.ndarray, dict[str, float]]:
    """Reconstruct height map via quadratic sub-slice peak localization."""
    nz, ny, nx = volume.shape
    if nz < config.min_slices:
        raise ValueError(f"Need >= {config.min_slices} slices, got {nz}")

    vol = volume.astype(np.float32, copy=False)
    z_int = np.argmax(vol, axis=0).astype(np.int32)
    z0_idx = np.clip(z_int, 1, nz - 2)

    yy, xx = np.indices((ny, nx))
    a = vol[z0_idx - 1, yy, xx]
    b = vol[z0_idx, yy, xx]
    c = vol[z0_idx + 1, yy, xx]

    denom = 2.0 * (a - 2.0 * b + c)
    is_peak = (np.abs(denom) > 1e-12) & (denom < 0) & (b >= a) & (b >= c)

    delta = np.zeros_like(b, dtype=np.float32)
    delta[is_peak] = (a[is_peak] - c[is_peak]) / denom[is_peak]
    delta = np.clip(delta, -0.5, 0.5)
    z_sub = z0_idx.astype(np.float32) + delta

    vol_med = np.median(vol, axis=0)
    vol_min = np.min(vol, axis=0)
    vol_max = np.max(vol, axis=0)
    local_range = np.maximum(vol_max - vol_min, 1e-12)

    peak_vs_median = b - vol_med
    peakness = b - np.maximum(a, c)
    edge = (z_int <= 0) | (z_int >= nz - 1)
    low_conf = peak_vs_median < config.confidence_frac * local_range
    flat_peak = peakness < config.confidence_frac * local_range

    bad = edge | low_conf | flat_peak | (~is_peak)
    z_sub[bad] = np.nan
    h_um = z_sub * float(dz_um) + float(z0_um)

    n_tot = float(h_um.size) if h_um.size else float("nan")
    qc = {
        "recon_nz_slices": float(nz),
        "recon_valid_frac": float(np.isfinite(h_um).sum() / n_tot)
        if h_um.size
        else float("nan"),
        "recon_edge_frac": float(edge.sum() / n_tot) if h_um.size else float("nan"),
    }
    return h_um, qc


def nan_aware_gaussian_1d(
    arr: np.ndarray, sigma: float, mode: str = "mirror"
) -> np.ndarray:
    """Gaussian-filter a 1D array while treating NaNs as missing data."""
    a = np.asarray(arr, dtype=float)
    if sigma <= 0:
        return a.copy()
    mask = np.isfinite(a).astype(float)
    a0 = np.where(mask > 0, a, 0.0)
    num = gaussian_filter1d(a0, sigma=float(sigma), mode=mode)
    den = gaussian_filter1d(mask, sigma=float(sigma), mode=mode)
    out = np.full_like(a, np.nan, dtype=float)
    good = den > 1e-12
    out[good] = num[good] / den[good]
    return out


def nan_aware_median_filter(
    arr2d: np.ndarray, size: int = 3, mode: str = "mirror"
) -> np.ndarray:
    """Median-filter a 2D array, ignoring NaNs while preserving NaN-only regions."""
    a = np.asarray(arr2d, dtype=float)
    if not np.isnan(a).any():
        return median_filter(a, size=size, mode=mode)
    return generic_filter(
        a, nanmedian_window, size=(size, size), mode=mode, cval=np.nan
    )


def estimate_pitch_angle_fft(
    h_um: np.ndarray,
    xy_um: float,
    pitch_hint_um: Optional[float] = None,
    config: AnalyzerConfig = CONFIG,
) -> tuple[float, float, dict[str, float]]:
    """Estimate pitch and groove orientation from the dominant 2D FFT peak."""
    h = np.asarray(h_um, dtype=np.float32)
    mask = np.isfinite(h)
    if not np.any(mask):
        return float("nan"), float("nan"), {"fft_valid": False}

    h0 = h.copy()
    h0[~mask] = float(np.nanmedian(h0[mask]))
    h0 -= float(np.mean(h0))

    ny, nx = h0.shape
    if config.fft_use_hann:
        h0 *= np.outer(np.hanning(ny), np.hanning(nx)).astype(np.float32)

    f2 = np.fft.fftshift(np.fft.fft2(h0))
    power = np.abs(f2) ** 2

    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=float(xy_um)))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=float(xy_um)))
    fx_grid, fy_grid = np.meshgrid(fx, fy)
    fmag = np.hypot(fx_grid, fy_grid)

    cy, cx = ny // 2, nx // 2
    yy, xx = np.ogrid[:ny, :nx]

    lo_um, hi_um = config.pitch_range_um
    dc_r = max(1, int(config.dc_exclusion_radius_px))
    ring = np.hypot(xx - cx, yy - cy) >= dc_r
    band = (fmag >= 1.0 / hi_um) & (fmag <= 1.0 / lo_um)
    freq_mask = ring & band

    if pitch_hint_um and math.isfinite(pitch_hint_um) and pitch_hint_um > 0.0:
        f0, tol = 1.0 / float(pitch_hint_um), 0.35
        freq_mask &= (fmag >= f0 * (1 - tol)) & (fmag <= f0 * (1 + tol))

    if not np.any(freq_mask):
        return float("nan"), float("nan"), {"fft_valid": False}

    masked_power = np.where(freq_mask, power, 0.0)
    peak_row, peak_col = np.unravel_index(int(np.argmax(masked_power)), power.shape)

    peak_row_f = float(peak_row)
    peak_col_f = float(peak_col)
    if 1 <= peak_row < ny - 1 and 1 <= peak_col < nx - 1:
        col_vals = power[peak_row - 1 : peak_row + 2, peak_col]
        row_vals = power[peak_row, peak_col - 1 : peak_col + 2]
        peak_row_f += float(np.clip(parabolic_subpixel_offset_3(col_vals), -0.5, 0.5))
        peak_col_f += float(np.clip(parabolic_subpixel_offset_3(row_vals), -0.5, 0.5))

    ky = float(np.interp(peak_row_f, np.arange(ny, dtype=float), fy))
    kx = float(np.interp(peak_col_f, np.arange(nx, dtype=float), fx))
    freq = math.hypot(kx, ky)
    if freq <= 0.0:
        return float("nan"), float("nan"), {"fft_valid": False}

    pitch_um = 1.0 / freq
    angle_deg = ((math.degrees(math.atan2(ky, kx)) + 90.0) % 180.0) - 90.0
    return pitch_um, angle_deg, {
        "fft_valid": True,
        "fft_pitch_um": pitch_um,
        "fft_angle_deg": angle_deg,
    }


def inlier_mask(h: np.ndarray, low: float = 1, high: float = 99) -> np.ndarray:
    """Return a percentile inlier mask over finite pixels."""
    a = np.asarray(h, dtype=float)
    finite = np.isfinite(a)
    if not np.any(finite):
        return np.zeros_like(a, dtype=bool)
    p_low, p_high = np.nanpercentile(a[finite], [low, high])
    return finite & (a >= p_low) & (a <= p_high)


def plane_subtract(
    h: np.ndarray, xy_um: float, mask: np.ndarray
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Fit and subtract a least-squares plane using masked inliers."""
    a = np.asarray(h, dtype=float)
    m = np.asarray(mask, dtype=bool) & np.isfinite(a)
    if np.count_nonzero(m) < 3:
        return a.copy(), (0.0, 0.0, 0.0)

    y_idx, x_idx = np.indices(a.shape)
    x = (x_idx[m] * float(xy_um)).ravel()
    y = (y_idx[m] * float(xy_um)).ravel()
    A = np.column_stack([x, y, np.ones_like(x)])
    coeff, *_ = lstsq(A, a[m].ravel(), rcond=None)
    plane = coeff[0] * x_idx * float(xy_um) + coeff[1] * y_idx * float(xy_um) + coeff[2]
    return a - plane, (float(coeff[0]), float(coeff[1]), float(coeff[2]))


def slope_correct_preserve_offset(
    h: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Remove only planar slope (about image center) while preserving offset."""
    a = np.asarray(h, dtype=float)
    m = np.asarray(mask, dtype=bool) & np.isfinite(a)
    if np.count_nonzero(m) < 3:
        return a.copy(), (0.0, 0.0, 0.0)

    yy, xx = np.indices(a.shape)
    A = np.column_stack([xx[m], yy[m], np.ones(np.count_nonzero(m))])
    coef = lstsq(A, a[m], rcond=None)[0]
    ax, by, c0 = float(coef[0]), float(coef[1]), float(coef[2])

    x0 = (a.shape[1] - 1) / 2.0
    y0 = (a.shape[0] - 1) / 2.0
    plane_slope = ax * (xx - x0) + by * (yy - y0)
    return a - plane_slope, (ax, by, c0)


def rotate_to_align(h: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate height map so grooves are horizontal; empty areas are NaN."""
    return rotate(
        np.asarray(h, dtype=float),
        float(angle_deg),
        reshape=False,
        order=1,
        mode="constant",
        cval=np.nan,
    )


def max_inscribed_rect(shape: tuple[int, int], angle_deg: float) -> tuple[int, int]:
    """Return height/width of max axis-aligned rectangle after rotation."""
    h0, w0 = shape
    t = abs(float(angle_deg)) % 180.0
    if t > 90.0:
        t = 180.0 - t
    th = np.deg2rad(t)
    c, s = float(np.cos(th)), float(np.sin(th))
    return (
        max(1, int(np.floor(h0 * c - w0 * s))),
        max(1, int(np.floor(w0 * c - h0 * s))),
    )


def sample_mean_profile(
    h: np.ndarray, n_stripes: int = 20, stripe_hw: int = 10
) -> np.ndarray:
    """Return an averaged 1D profile across multiple horizontal stripes."""
    a = np.asarray(h, dtype=float)
    ny, nx = a.shape
    if ny < 2 * stripe_hw + 1:
        return nanmean_axis0_no_warn(a)

    rows = np.linspace(stripe_hw, ny - stripe_hw - 1, n_stripes, dtype=int)
    profiles: list[np.ndarray] = []

    for r in rows:
        stripe = a[r - stripe_hw : r + stripe_hw + 1, :]
        if np.isnan(stripe).all():
            continue

        prof = nanmean_axis0_no_warn(stripe)
        prof = interpolate_finite_1d(prof)

        if prof.size >= 5:
            x = np.arange(prof.size, dtype=float)
            try:
                coef = np.polyfit(x, prof, 2)
                prof = prof - np.polyval(coef, x)
            except np.linalg.LinAlgError:
                prof = baseline_to_zero(prof)

        profiles.append(prof)

    return np.mean(profiles, axis=0) if profiles else np.full(nx, np.nan, dtype=float)


def detect_peaks_valleys(
    profile: np.ndarray, pitch_px: float, config: AnalyzerConfig = CONFIG
) -> tuple[np.ndarray, np.ndarray]:
    """Detect peaks and valleys on a 1D profile using distance and prominence."""
    if profile is None:
        return np.array([], dtype=int), np.array([], dtype=int)

    p = np.asarray(profile, dtype=float)
    if p.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    if not np.any(np.isfinite(p)):
        return np.array([], dtype=int), np.array([], dtype=int)

    ptp = float(np.nanmax(p) - np.nanmin(p))
    if ptp <= 0.0:
        return np.array([], dtype=int), np.array([], dtype=int)

    prom = float(config.prominence_frac) * ptp
    dist = max(1, int(float(config.min_peak_distance_frac) * float(pitch_px)))
    peaks, _ = find_peaks(p, distance=dist, prominence=prom)
    valleys, _ = find_peaks(-p, distance=dist, prominence=prom)
    return peaks.astype(int), valleys.astype(int)


def measure_pitch_per_groove(
    h_aligned: np.ndarray,
    xy_um: float,
    pitch_px_estimate: float,
    outdir: Path,
    stem: str,
    config: AnalyzerConfig = CONFIG,
) -> MeasurementResult:
    """Measure pitch from peak-to-peak distances on a filtered average profile."""
    if h_aligned is None or np.size(h_aligned) == 0:
        return MeasurementResult(float("nan"))

    avg_profile = interpolate_finite_1d(sample_mean_profile(h_aligned))
    lambda_c_um = (
        float(pitch_px_estimate) * float(xy_um) * float(config.lambda_c_factor)
    )
    sigma_px = iso16610_sigma_px(lambda_c_um, float(xy_um))
    detect_profile = nan_aware_gaussian_1d(avg_profile, sigma_px, mode=config.pad_mode)

    L = int(np.ceil(lambda_c_um / float(xy_um)))
    offset = 0
    if detect_profile.size > 2 * L:
        detect_profile = detect_profile[L:-L]
        offset = L

    peaks, _ = detect_peaks_valleys(detect_profile, pitch_px_estimate, config)
    peaks = peaks + offset
    if peaks.size < 2:
        return MeasurementResult(float("nan"))

    x_um = np.arange(avg_profile.size, dtype=float) * float(xy_um)
    pitch_arr = np.diff(x_um[peaks])
    if pitch_arr.size == 0:
        return MeasurementResult(float("nan"))

    mean_pitch = float(np.mean(pitch_arr))
    n = int(pitch_arr.size)
    std = float(np.std(pitch_arr, ddof=1)) if n > 1 else float("nan")
    u_a = std / math.sqrt(n) if n > 1 else float("nan")
    u_b = type_b_resolution(float(xy_um))

    row_pitch_values: list[float] = []
    for row in np.asarray(h_aligned, dtype=float):
        if not np.any(np.isfinite(row)):
            continue
        row_profile = nan_aware_gaussian_1d(row, sigma_px, mode=config.pad_mode)
        if row_profile.size > 2 * L:
            row_profile = row_profile[L:-L]
            row_offset = L
        else:
            row_offset = 0

        row_peaks, _ = detect_peaks_valleys(row_profile, pitch_px_estimate, config)
        row_peaks = row_peaks + row_offset
        if row_peaks.size < 2:
            continue

        row_pitch = np.diff(x_um[row_peaks])
        if len(row_pitch) >= 12:
            row_clean = row_pitch[tukey_fence_mask(row_pitch, k=config.tukey_k)]
        else:
            row_clean = row_pitch

        row_pitch_values.extend(row_clean.tolist())

    if row_pitch_values:
        arr = np.asarray(row_pitch_values, dtype=float)
        std_fov = float(np.sqrt(np.nanmean((arr - mean_pitch) ** 2)))
        if config.generate_plots:
            plot_histogram(
                arr,
                30,
                 outdir / f"{stem}_pitch_hist",
                "Pitch Distribution",
                "Pitch (µm)",
                config,
            )
    else:
        std_fov = float("nan")

    return MeasurementResult(
        mean_pitch, u_a=u_a, u_b=u_b, n=n, dof=max(0, n - 1), std=std_fov
    )


def measure_depth_per_groove(
    h_aligned: np.ndarray,
    xy_um: float,
    dz_um: float,
    pitch_px_estimate: float,
    bias_mult_factor: float,
    fft_pitch_um: float,
    outdir: Path,
    stem: str,
    config: AnalyzerConfig = CONFIG,
) -> MeasurementResult:
    """Measure depth as peak minus mean of adjacent valleys on a filtered profile."""
    if h_aligned is None or np.size(h_aligned) == 0:
        return MeasurementResult(float("nan"))

    avg_profile = interpolate_finite_1d(sample_mean_profile(h_aligned))
    raw = baseline_to_zero(avg_profile)

    lambda_c_um = (
        float(pitch_px_estimate) * float(xy_um) * float(config.lambda_c_factor)
    )
    sigma_px = iso16610_sigma_px(lambda_c_um, float(xy_um))
    detect_profile = nan_aware_gaussian_1d(raw, sigma_px, mode=config.pad_mode)

    L = int(np.ceil(lambda_c_um / float(xy_um)))
    if detect_profile.size > 2 * L:
        detect_profile = detect_profile[L:-L]
        raw = raw[L:-L]
        avg_profile = avg_profile[L:-L]

    peaks, valleys = detect_peaks_valleys(detect_profile, pitch_px_estimate, config)
    if peaks.size == 0 or valleys.size < 2:
        return MeasurementResult(float("nan"))

    segments: list[np.ndarray] = []
    for i in range(max(0, len(peaks) - 1)):
        lo, hi = int(peaks[i]) , int(peaks[i + 1]) + 1
        seg = avg_profile[lo:hi]
        if seg.size >= 3:
            n_out = max(3, int(round(pitch_px_estimate)))
            interp_seg = np.interp(
                np.linspace(0.0, 1.0, n_out),
                np.linspace(0.0, 1.0, seg.size),
                seg,
            )
            segments.append(interp_seg)

    if segments:
        seg_mean = np.mean(segments, axis=0)
        ref_groove = baseline_to_zero(seg_mean)
        if config.generate_plots:
            plot_reference_groove(
                ref_groove, fft_pitch_um, outdir / f"{stem}_reference_groove", config
            )
        if pd is not None:
            pd.DataFrame(segments).to_csv(
                outdir / f"{stem}_groove_profiles.csv",
                index=False,
                float_format="%.4f",
            )
            pd.DataFrame({"groove": ref_groove}).to_csv(
                outdir / f"{stem}_reference_groove.csv",
                index=False,
                float_format="%.4f",
            )

    depths: list[float] = []
    for p in peaks:
        v_left = valleys[valleys < p]
        v_right = valleys[valleys > p]
        if v_left.size == 0 or v_right.size == 0:
            continue
        vl, vr = int(v_left[-1]), int(v_right[0])
        peak_h = float(raw[p])
        valley_h = 0.5 * (float(raw[vl]) + float(raw[vr]))
        depths.append(peak_h - valley_h)

    if not depths:
        return MeasurementResult(float("nan"))

    depth_arr = np.asarray(depths, dtype=float) * bias_mult_factor
    if depth_arr.size == 0:
        return MeasurementResult(float("nan"))

    mean_depth = float(np.mean(depth_arr))
    n = int(depth_arr.size)
    std = float(np.std(depth_arr, ddof=1)) if n > 1 else float("nan")
    u_a = std / math.sqrt(n) if n > 1 else float("nan")

    u_b_res = type_b_resolution(float(dz_um) * 0.5)
    u_b_cal = float(config.u_rel_calibration) * mean_depth
    u_b = math.hypot(u_b_res, u_b_cal)

    row_depth_values: list[float] = []
    for row in np.asarray(h_aligned, dtype=float):
        if not np.any(np.isfinite(row)):
            continue
        row_raw = baseline_to_zero(row)
        row_profile = nan_aware_gaussian_1d(row_raw, sigma_px, mode=config.pad_mode)
        if row_profile.size > 2 * L:
            row_profile = row_profile[L:-L]
            row_raw = row_raw[L:-L]

        row_peaks, row_valleys = detect_peaks_valleys(
            row_profile, pitch_px_estimate, config
        )
        if row_peaks.size == 0 or row_valleys.size < 2:
            continue

        row_depths: list[float] = []
        for p in row_peaks:
            v_left = row_valleys[row_valleys < p]
            v_right = row_valleys[row_valleys > p]
            if v_left.size == 0 or v_right.size == 0:
                continue
            vl, vr = int(v_left[-1]), int(v_right[0])
            peak_h = float(row_raw[p])
            valley_h = 0.5 * (float(row_raw[vl]) + float(row_raw[vr]))
            row_depths.append(peak_h - valley_h)

        if not row_depths:
            continue
        row_arr = np.asarray(row_depths, dtype=float)
        if len(row_depths) >= 12:
            row_clean = row_arr[tukey_fence_mask(row_arr, k=config.tukey_k)]
        else:
            row_clean = row_arr

        row_depth_values.extend(row_clean.tolist())

    if row_depth_values:
        arr = np.asarray(row_depth_values, dtype=float) * bias_mult_factor
        mean_fov = np.nanmean(arr)
        std_fov = np.nanstd(arr)
        # std_fov = float(np.sqrt(np.nanmean((arr - mean_depth) ** 2)))
        if config.generate_plots:
            plot_histogram(
                arr,
                30,
                outdir / f"{stem}_depth_hist",
                "Depth Distribution",
                "Depth (µm)",
                config,
            )

    else:
        std_fov = float("nan")


    return MeasurementResult(
        mean_fov, u_a=u_a, u_b=u_b, n=n, dof=max(0, n - 1), std=std_fov
    )


def measure_gel_height(
    h_aligned_raw: np.ndarray,
    xy_um: float,
    dz_um: float,
    pitch_px_estimate: float,
    outdir: Path,
    stem: str,
    config: AnalyzerConfig = CONFIG,
) -> MeasurementResult:
    """Measure gel height as median groove-valley height across rows and grooves."""
    if h_aligned_raw is None or np.size(h_aligned_raw) == 0:
        return MeasurementResult(float("nan"))

    h = np.asarray(h_aligned_raw, dtype=float)
    nrows = h.shape[0]
    avg_profile = columnwise_nanmedian(h)
    avg_profile = interpolate_finite_1d(avg_profile)

    lambda_c_um = (
        float(pitch_px_estimate) * float(xy_um) * float(config.lambda_c_factor)
    )
    sigma_px = iso16610_sigma_px(lambda_c_um, float(xy_um))
    detect_profile = nan_aware_gaussian_1d(avg_profile, sigma_px, mode=config.pad_mode)

    _, valleys = detect_peaks_valleys(detect_profile, pitch_px_estimate, config)
    if valleys.size == 0:
        return MeasurementResult(float("nan"))

    half_w = max(1, int(0.35 * float(pitch_px_estimate)))
    valley_heights: list[float] = []

    for y in range(nrows):
        row = h[y, :]
        if not np.any(np.isfinite(row)):
            continue
        row_s = nan_aware_gaussian_1d(
            row, sigma=max(1.0, 0.15 * float(pitch_px_estimate))
        )

        for v_approx in valleys:
            lo = max(0, int(v_approx) - half_w)
            hi = min(row.shape[0], int(v_approx) + half_w + 1)
            seg = row_s[lo:hi]
            if seg.size < 3 or not np.any(np.isfinite(seg)):
                continue
            seg_safe = np.where(np.isfinite(seg), seg, np.inf)
            x_val = lo + int(np.argmin(seg_safe))
            z_val = row[x_val]
            if np.isfinite(z_val):
                valley_heights.append(float(z_val))

    if not valley_heights:
        return MeasurementResult(float("nan"))

    heights = np.asarray(valley_heights, dtype=float)
    clean = heights[mad_clean_mask(heights, k=3.5)]
    if config.generate_plots:
        if clean.size > 0:
            plot_histogram(
                clean,
                60,
                outdir / f"{stem}_gel_height_hist",
                "Groove Valley Height Distribution",
                "Height (µm)",
                config,
            )
    if clean.size == 0:
        return MeasurementResult(float("nan"))

    median_height = float(np.median(clean))
    n = int(clean.size)
    std = float(np.std(clean, ddof=1)) if n > 1 else float("nan")
    if n > 1:
        mad = float(np.median(np.abs(clean - median_height)))
        u_a = 1.2533 * 1.4826 * mad / math.sqrt(n) #correction for median-based MAD
    else:
        u_a = float("nan")
    u_b = type_b_resolution(float(dz_um))
    return MeasurementResult(
        median_height, u_a=u_a, u_b=u_b, n=n, dof=max(0, n - 1), std=std
    )


def plot_height_map(
    h: np.ndarray,
    xy_um: float,
    path: Path,
    config: AnalyzerConfig = CONFIG,
) -> None:
    """Save height map as PNG/PDF next to the given base path."""
    if plt is None:
        return
    a = np.asarray(h, dtype=float)
    finite = np.isfinite(a)
    if not np.any(finite):
        return

    fig, ax = plt.subplots(figsize=(8, 6), dpi=config.plot_dpi)
    extent = [0, a.shape[1] * float(xy_um), 0, a.shape[0] * float(xy_um)]
    p1, p99 = np.nanpercentile(a[finite], [1, 99])
    im = ax.imshow(
        a,
        origin="lower",
        extent=extent,
        vmin=float(p1),
        vmax=float(p99),
        cmap="viridis",
    )
    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.set_title("Height Map")
    fig.colorbar(im, ax=ax, label="Height (µm)")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=config.plot_dpi)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_profile_annotated(
    profile: np.ndarray,
    xy_um: float,
    peaks: np.ndarray,
    valleys: np.ndarray,
    path: Path,
    config: AnalyzerConfig = CONFIG,
) -> None:
    """Save a line profile with peak/valley annotations as PNG/PDF."""
    if plt is None:
        return
    p = np.asarray(profile, dtype=float)
    if p.size == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 4), dpi=config.plot_dpi)
    x = np.arange(p.size, dtype=float) * float(xy_um)
    ax.plot(x, p, linewidth=0.8, label="Profile")

    if peaks.size:
        ax.plot(x[peaks], p[peaks], "o", markersize=4, label="Peaks")
    if valleys.size:
        ax.plot(x[valleys], p[valleys], "v", markersize=4, label="Valleys")

    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Height (µm)")
    ax.set_title("Annotated Average Profile")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=config.plot_dpi)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_histogram(
    data: np.ndarray,
    bins: int,
    path: Path,
    title: str,
    xlabel: str,
    config: AnalyzerConfig = CONFIG,
) -> None:
    """Save a histogram plot as PNG/PDF."""
    if plt is None:
        return
    a = np.asarray(data, dtype=float)
    valid = a[np.isfinite(a)]
    if valid.size == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 4), dpi=config.plot_dpi)
    ax.hist(valid, bins=int(bins), edgecolor="black", alpha=0.7)

    mean = float(np.mean(valid))
    std = float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0
    ax.axvline(mean, linestyle="--", label=f"Mean: {mean:.2f}")
    ax.axvline(mean + std, linestyle=":")
    ax.axvline(mean - std, linestyle=":", label=f"±1σ: {std:.2f}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=config.plot_dpi)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_reference_groove(
    profile: np.ndarray, pitch_um: float, path: Path, config: AnalyzerConfig = CONFIG
) -> None:
    """Save the mean groove shape over one pitch as PNG/PDF."""
    if plt is None:
        return
    p = np.asarray(profile, dtype=float)
    if p.size == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 4), dpi=config.plot_dpi)
    x = np.linspace(0.0, float(pitch_um), p.size)
    ax.plot(x, p, linewidth=1.5)
    ax.fill_between(x, p, alpha=0.3)
    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Height (µm)")
    ax.set_title(f"Reference Groove (pitch = {pitch_um:.2f} µm)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=config.plot_dpi)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)



def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write list-of-dicts to CSV using the union of keys as fieldnames."""
    if not rows:
        return
    fieldnames: list[str] = list(rows[0].keys())
    for r in rows[1:]:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_file(
    fpath: Path,
    xy_um: float,
    dz_um: float,
    z0_um: float = 0.0,
    pitch_hint_um: Optional[float] = None,
    outdir: Optional[Path] = None,
    analysis_sha256: Optional[str] = '',
    analysis_timestamp_utc: Optional[str] = '',
    config: AnalyzerConfig = CONFIG,
) -> dict[str, Any]:
    """Analyze one TIFF stack and return a results dict (also writes outputs)."""
    outdir = outdir or (fpath.parent / f"{fpath.stem}_proc")
    outdir.mkdir(parents=True, exist_ok=True)
    stem = fpath.stem
    analysis_info = (
        f"{ANALYSIS_NAME}|{ANALYSIS_VERSION}|{analysis_sha256[:12]}|{analysis_timestamp_utc}"
    )

    volume = load_volume(fpath)
    h_um, recon_qc = reconstruct_height_map(volume, dz_um, z0_um, config)
    recon_qc = {
        "recon_nz_slices": round_for_csv(recon_qc.get("recon_nz_slices"), 0),
        "recon_valid_prc": round_for_csv(recon_qc.get("recon_valid_frac")*100, 1),
        "NaN_edge_prc": round_for_csv(recon_qc.get("recon_edge_frac")*100, 1),
    }
    valid_frac = float(np.isfinite(h_um).sum() / h_um.size) if h_um.size else 0.0

    if config.generate_plots:
        plot_height_map(h_um, xy_um, outdir / f"{stem}_raw_height_map", config)

    mask_all = inlier_mask(h_um)
    h_detrended, _ = plane_subtract(h_um, xy_um, mask_all)
    h_filtered = nan_aware_median_filter(h_detrended, size=5)
    mask_in = inlier_mask(h_filtered)

    h_in = h_filtered.copy()
    h_in[~mask_in] = np.nan

    h_slope_corrected, _ = slope_correct_preserve_offset(h_um, mask_in)
    h_slope_in = h_slope_corrected.copy()
    h_slope_in[~mask_in] = np.nan

    fft_pitch_um, fft_angle_deg, fft_qc = estimate_pitch_angle_fft(
        h_in, xy_um, pitch_hint_um, config
    )
    fft_qc = {
        "fft_pitch_um": round_for_csv(fft_qc.get("fft_pitch_um"), 1),
        "fft_angle_deg": round_for_csv(fft_qc.get("fft_angle_deg"), 4),
    }
    if not math.isfinite(fft_pitch_um) and pitch_hint_um:
        fft_pitch_um, fft_angle_deg = float(pitch_hint_um), 0.0

    if not math.isfinite(fft_pitch_um):
        return {
            "file": str(fpath),
            "error": "Failed to estimate pitch",
            "valid_pixel_frac": valid_frac,
            **recon_qc,
        }

    pitch_px = float(fft_pitch_um) / float(xy_um)

    if abs(float(fft_angle_deg)) > 0.5:
        # Fold a rotation beyond 45 deg into a <=45 deg rotation plus a transpose
        # (a quarter turn is lossless), so the inscribed-rectangle crop never
        # collapses to a degenerate 1x1 for grooves far from vertical.
        rot_angle = float(fft_angle_deg)
        swap_axes = False
        if rot_angle > 45.0:
            rot_angle -= 90.0
            swap_axes = True
        elif rot_angle < -45.0:
            rot_angle += 90.0
            swap_axes = True

        h_aligned = rotate_to_align(h_in, rot_angle)
        h_slope_aligned = rotate_to_align(h_slope_in, rot_angle)
        if swap_axes:
            h_aligned = h_aligned.T
            h_slope_aligned = h_slope_aligned.T

        rh, rw = max_inscribed_rect(h_aligned.shape, rot_angle)
        h0, w0 = h_aligned.shape
        r0, c0 = (h0 - rh) // 2, (w0 - rw) // 2

        h_crop = h_aligned[r0 : r0 + rh, c0 : c0 + rw]
        h_slope_crop = h_slope_aligned[r0 : r0 + rh, c0 : c0 + rw]
    else:
        h_crop, h_slope_crop = h_in, h_slope_in

    if config.generate_plots:
        plot_height_map(h_crop, xy_um, outdir / f"{stem}_height_map_aligned", config)

    pitch_result = measure_pitch_per_groove(h_crop, xy_um, pitch_px, outdir, stem, config)
    # Depth bias correction: multiplier determined by measured pitch (fallback to FFT pitch).
    pitch_um_for_depth_corr = (
        pitch_result.value
        if math.isfinite(pitch_result.value)
        else float(fft_pitch_um)
    )
    depth_corr_mult = depth_correction_multiplier(pitch_um_for_depth_corr)
    depth_result = measure_depth_per_groove(h_crop, xy_um, dz_um, pitch_px,depth_corr_mult, fft_pitch_um, outdir, stem, config)
    height_result = measure_gel_height(h_slope_crop, xy_um, dz_um, pitch_px, outdir, stem, config)

    avg_profile = baseline_to_zero(sample_mean_profile(h_crop))

    lambda_c_um = float(fft_pitch_um) * float(config.lambda_c_factor)
    sigma_detect = iso16610_sigma_px(lambda_c_um, float(xy_um))
    detect_profile = nan_aware_gaussian_1d(avg_profile, sigma_detect)
    peaks, valleys = detect_peaks_valleys(detect_profile, pitch_px, config)

    if config.generate_plots:
        plot_profile_annotated(
            avg_profile,
            xy_um,
            peaks,
            valleys,
            outdir / f"{stem}_avg_line_profile_annotated",
            config,
        )

    results: dict[str, Any] = {
        "file": fpath.name,
        "folder": fpath.parent.name,
        "pitch_um": round_for_csv(pitch_result.value,1),
        "pitch_u_a_um": round_for_csv(pitch_result.u_a, 1),
        "pitch_u_b_um": round_for_csv(pitch_result.u_b,1),
        "pitch_u_combined_um": round_for_csv(pitch_result.u_combined,1),
        "pitch_std_um": round_for_csv(pitch_result.std,1),
        "pitch_n": round_for_csv(pitch_result.n,0),
        "depth_um": round_for_csv(depth_result.value,1),
        "depth_u_a_um": round_for_csv(depth_result.u_a,1),
        "depth_u_b_um": round_for_csv(depth_result.u_b,1),
        "depth_u_combined_um": round_for_csv(depth_result.u_combined,1),
        "depth_std_um": round_for_csv(depth_result.std,1),
        "depth_n": round_for_csv(depth_result.n,1),
        "depth_bias_corr_factor": round_for_csv(float(depth_corr_mult),3),
        "gel_height_um": round_for_csv(height_result.value,1),
        "gel_height_u_a_um": round_for_csv(height_result.u_a,1),
        "gel_height_u_b_um": round_for_csv(height_result.u_b,1),
        "gel_height_u_combined_um": round_for_csv(height_result.u_combined,1),
        "gel_height_std_um": round_for_csv(height_result.std,1),
        "gel_height_n": round_for_csv(height_result.n,0),
        "fft_pitch_um": round_for_csv(float(fft_pitch_um),1),
        "fft_angle_deg": round_for_csv(float(fft_angle_deg),4),
        "n_grooves": round_for_csv(pitch_result.n,0),
        "valid_pixel_prc": round_for_csv(valid_frac*100, 1),
        **recon_qc,
        **fft_qc,
        "xy_um_per_px": float(xy_um),
        "dz_um_per_slice": float(dz_um),
        "z0_um_offset": float(z0_um),
        "analysis_summary": analysis_info,
    }

    if not (
        math.isfinite(pitch_result.value) or math.isfinite(depth_result.value)
    ):
        results["error"] = "No measurable grooves after alignment and cropping"

    summary_path = outdir / f"{stem}_analysis_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"Groove Analyzer v{__version__} - Analysis Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"File: {fpath}\nTimestamp: {now_utc()}\n\n")
        f.write("PRIMARY MEASUREMENTS (with GUM uncertainty):\n")
        f.write(
            f"  Pitch: {pitch_result.value:.3f} ± {pitch_result.std:.3f} µm "
            f"(n={pitch_result.n})\n"
        )
        f.write(
            f"  Depth: {depth_result.value:.3f} ± {depth_result.std:.3f} µm "
            f"(bias correction: ×{depth_corr_mult:.4f}, n={depth_result.n})\n"
        )
        f.write(
            f"  Gel Height: {height_result.value:.3f} ± {height_result.std:.3f} "
            f"µm "
            f"(n={height_result.n})\n"
        )

    return results


def _resolve_file_outdir(
    input_path: Path, file_path: Path, output_dir: Path
) -> Path:
    """Return per-file output dir under output_dir while preserving structure."""
    if input_path.is_dir():
        try:
            rel_parent = file_path.parent.relative_to(input_path)
            return output_dir / rel_parent / f"{file_path.stem}_proc"
        except ValueError:
            return output_dir / f"{file_path.stem}_proc"
    return output_dir / f"{file_path.stem}_proc"


def analyze_batch(
    input_path: Path,
    xy_um: float,
    dz_um: float,
    z0_um: float = 0.0,
    pitch_hint_um: Optional[float] = None,
    output_dir: Optional[Path] = None,
    analysis_sha256: Optional[str] = '',
    config: AnalyzerConfig = CONFIG,
) -> list[dict[str, Any]]:
    """Analyze a TIFF or directory tree of TIFFs and return list of result dicts."""
    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(
            p
            for p in input_path.rglob("*")
            if p.suffix.lower() in {".tif", ".tiff"}
        )

    root_out = output_dir or (input_path if input_path.is_dir() else input_path.parent)
    root_out.mkdir(parents=True, exist_ok=True)

    meta = {
        "timestamp_utc": now_utc(),
        "version": __version__,
        "analysis_sha256": analysis_sha256,
        "input": str(input_path),
        "n_files": len(files),
        "calibration": {
            "xy_um_per_px": float(xy_um),
            "dz_um_per_slice": float(dz_um),
            "z0_um_offset": float(z0_um),
        },
    }
    with (root_out / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    all_results: list[dict[str, Any]] = []
    for i, fpath in enumerate(files, 1):
        LOGGER.info("[%4d/%4d] %s", i, len(files), fpath.name)
        try:
            file_outdir = _resolve_file_outdir(input_path, fpath, root_out)
            results = analyze_file(
                fpath,
                xy_um,
                dz_um,
                z0_um=z0_um,
                pitch_hint_um=pitch_hint_um,
                outdir=file_outdir,
                analysis_sha256=analysis_sha256,
                analysis_timestamp_utc=meta["timestamp_utc"],
                config=config,
            )
            all_results.append(results)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed %s", fpath)
            all_results.append({"file": fpath.name, "error": str(exc)})

    write_csv(root_out / "groove_recap.csv", all_results)
    return all_results


def gui_main() -> None:
    """Run an interactive GUI workflow to analyze a file or folder."""
    if tk is None or messagebox is None or filedialog is None or simpledialog is None:
        raise RuntimeError("tkinter is required for GUI mode")

    root = tk.Tk()
    root.withdraw()

    is_folder = messagebox.askyesno(
        "Groove Analyzer", "Analyze folder? (No = single file)"
    )
    if is_folder:
        path = filedialog.askdirectory(title="Select folder")
    else:
        path = filedialog.askopenfilename(
            title="Select TIFF", filetypes=[("TIFF", "*.tif *.tiff")]
        )

    if not path:
        raise SystemExit("No selection")

    input_path = Path(path)

    xy_um = simpledialog.askfloat(
        "Calibration", "XY pixel size (µm/px):", minvalue=0.001, initialvalue=1.34
    )
    dz_um = simpledialog.askfloat(
        "Calibration", "Z step (µm/slice):", minvalue=0.001, initialvalue=0.9
    )
    z0_um = simpledialog.askfloat(
        "Z Reference", "Z0 offset (µm):", minvalue=-1e9, initialvalue=0.0
    )
    if None in (xy_um, dz_um, z0_um):
        raise SystemExit("Cancelled")

    analysis_sha256 = sha256_file(__file__)
    results = analyze_batch(input_path, float(xy_um), float(dz_um), float(z0_um),analysis_sha256=analysis_sha256)
    n_success = sum(1 for r in results if "error" not in r)
    messagebox.showinfo("Complete", f"Analyzed {n_success}/{len(results)} files")


def _parse_log_level(level: str) -> int:
    """Return a logging level from a string, defaulting to INFO on errors."""
    name = (level or "INFO").upper()
    return getattr(logging, name, logging.INFO)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entrypoint for batch analysis."""
    parser = argparse.ArgumentParser(description=f"Groove Analyzer v{__version__}")
    parser.add_argument("-i", "--input", type=Path, help="Input TIFF or directory")
    parser.add_argument("-o", "--output", type=Path, help="Output directory")
    parser.add_argument("--xy", type=float, help="XY pixel size (µm/px)")
    parser.add_argument("--dz", type=float, help="Z step (µm/slice)")
    parser.add_argument("--z0", type=float, default=0.0, help="Z0 offset (µm)")
    parser.add_argument("--pitch-hint", type=float, help="Pitch hint (µm)")
    parser.add_argument("--no-plots", action="store_true", help="Disable plots")
    parser.add_argument("--gui", action="store_true", help="Launch GUI")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=_parse_log_level(args.log_level),
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    if args.gui or args.input is None:
        gui_main()
        return

    if args.xy is None or args.dz is None:
        parser.error("--xy and --dz are required for CLI")

    config = AnalyzerConfig(generate_plots=not args.no_plots)
    results = analyze_batch(
        args.input,
        float(args.xy),
        float(args.dz),
        z0_um=float(args.z0),
        pitch_hint_um=args.pitch_hint,
        output_dir=args.output,
        config=config,
    )
    n_success = sum(1 for r in results if "error" not in r)
    print(f"\nComplete: {n_success}/{len(results)} files analyzed")


if __name__ == "__main__":
    main()
