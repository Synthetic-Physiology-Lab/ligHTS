#!/usr/bin/env python3
"""Groove Analyzer (holographic height maps).
Measures groove pitch/depth from TIFF height maps and writes per-file and batch outputs."""


from __future__ import annotations

import argparse
import csv
import hashlib
import datetime as dt
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import matplotlib as mpl

import numpy as np
from numpy.linalg import lstsq
from scipy.ndimage import gaussian_filter1d, generic_filter, median_filter, rotate
from scipy.signal import find_peaks

try:
    import tifffile
except ImportError:
    tifffile = None

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog
except ImportError:
    tk = None
    filedialog = None
    messagebox = None
    simpledialog = None

try:
    mpl.use("Agg", force=True)
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


__version__ = "7.2.0-holo-aligned"
LOGGER = logging.getLogger("groove_analyzer_holo_v7aligned")


def sha256_file(path: str) -> str:
    """Compute SHA-256 hex digest for a file at `path` (streamed, constant memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


try:
    ANALYZER_SHA256 = sha256_file(__file__)
except Exception:  # pragma: no cover - provenance is best-effort
    ANALYZER_SHA256 = "unknown"


@dataclass(frozen=True)
class AnalyzerConfig:
    """Configuration for groove analysis (expected pitch, filtering, QC, plotting)."""

    pitch_range_um: Tuple[float, float] = (15.0, 100.0)
    lambda_c_factor: float = 0.8
    prominence_frac: float = 0.03
    min_peak_distance_frac: float = 0.45
    dc_exclusion_radius_px: int = 5
    fft_use_hann: bool = True
    rotate_if_abs_angle_deg_gt: float = 0.5
    tukey_k: float = 1.5
    u_rel_calibration: float = 0.012
    qc_min_valid_frac: float = 0.50
    qc_min_grooves: int = 6
    generate_plots: bool = True
    plot_dpi: int = 300


CONFIG = AnalyzerConfig()


def evaluate_qc(
    valid_frac: float,
    n_grooves: float,
    config: AnalyzerConfig = CONFIG,
) -> Tuple[bool, str]:
    """Assess one stack against the configured QC thresholds. Report only.

    Returns ``(qc_pass, qc_reasons)``, where ``qc_reasons`` is a
    semicolon-separated list of the criteria that failed (empty when the
    stack passes). Nothing is excluded on the basis of this result: the
    flags are written to the per-file summary, results CSV and run
    metadata so that exclusion decisions are made, and can be audited,
    downstream.
    """
    reasons: List[str] = []

    if not math.isfinite(valid_frac):
        reasons.append("valid_frac=nan")
    elif valid_frac < config.qc_min_valid_frac:
        reasons.append(
            f"valid_frac={valid_frac:.3f}<{config.qc_min_valid_frac:.2f}"
        )

    if not math.isfinite(n_grooves):
        reasons.append("n_grooves=nan")
    elif n_grooves < config.qc_min_grooves:
        reasons.append(f"n_grooves={n_grooves:.0f}<{config.qc_min_grooves}")

    return (not reasons), ";".join(reasons)

FIGSIZE_IN = (6, 6)


def set_pub_style() -> None:
    """Set matplotlib rcParams to match flat_gel_surface_analyzer figures."""
    if plt is None:
        return

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.weight": "bold",
            "font.size": 12,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )



@dataclass
class MeasurementResult:
    """Measurement value with Type-A/Type-B uncertainties and summary stats."""

    value: float
    u_a: float = float("nan")
    u_b: float = float("nan")
    n: int = 0
    rmse: float = float("nan")

    @property
    def u_combined(self) -> float:
        """Combined standard uncertainty: sqrt(u_a^2 + u_b^2)."""
        if not (np.isfinite(self.u_a) and np.isfinite(self.u_b)):
            return float("nan")
        return float(math.sqrt(self.u_a**2 + self.u_b**2))


def now_utc() -> str:
    """Return current UTC timestamp as ISO 8601 string with 'Z' suffix."""
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def type_b_resolution(delta: float) -> float:
    """Return Type-B standard uncertainty for resolution `delta` (|delta|/sqrt(12))."""
    return float(abs(delta) / math.sqrt(12.0))


def iso16610_sigma_um(lambda_c_um: float) -> float:
    """Convert ISO 16610 cutoff wavelength `lambda_c_um` to Gaussian sigma [µm]."""
    return float(lambda_c_um * math.sqrt(math.log(2.0)) / (math.pi * math.sqrt(2.0)))


def iso16610_sigma_px(lambda_c_um: float, xy_um: float) -> float:
    """Convert ISO 16610 cutoff wavelength to Gaussian sigma [px]."""
    return float(iso16610_sigma_um(lambda_c_um) / xy_um)


def nanmedian_window(values: np.ndarray) -> float:
    """Return the median of finite values in a generic_filter window."""
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


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


def tukey_fence_mask(arr: np.ndarray, k: float = 1.5) -> np.ndarray:
    """Return boolean inlier mask using Tukey fences (Q1/Q3 ± k*IQR)."""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return np.zeros_like(arr, dtype=bool)
    valid = np.isfinite(arr)
    q1, q3 = np.nanpercentile(arr[valid], [25.0, 75.0])
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0:
        return valid
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return valid & (arr >= lo) & (arr <= hi)


def load_height_map(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load TIFF as 2D height array (median-project 3D stacks) and metadata dict."""
    if tifffile is None:
        raise ImportError("tifffile is required: pip install tifffile")

    arr = tifffile.imread(str(path))
    meta: Dict[str, Any] = {
        "tiff_shape": tuple(int(x) for x in arr.shape),
        "tiff_dtype": str(arr.dtype),
    }

    if arr.ndim == 2:
        return np.asarray(arr), {**meta, "combine_mode": "2d"}
    if arr.ndim == 3:
        med = np.nanmedian(arr, axis=0)
        if np.issubdtype(arr.dtype, np.integer):
            med = np.rint(med).astype(arr.dtype)
        return np.asarray(med), {**meta, "combine_mode": "median_z"}
    raise ValueError(f"Expected 2D or 3D TIFF, got shape {arr.shape}")


def to_height_um(
    h_raw: np.ndarray,
    dz_um_per_gray: float,
    z0_um: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Convert TIFF values to micrometers using dz/z0 for integer arrays; returns (h_um, meta)."""
    if np.issubdtype(h_raw.dtype, np.floating):
        return h_raw.astype(np.float32, copy=False), {
            "height_units": "um_assumed",
            "applied_scale": False,
        }

    h = h_raw.astype(np.float32, copy=False)
    if not (np.isfinite(dz_um_per_gray) and dz_um_per_gray > 0):
        raise ValueError(f"dz_um_per_gray must be finite and >0, got {dz_um_per_gray}")
    if not np.isfinite(z0_um):
        raise ValueError(f"z0_um must be finite, got {z0_um}")

    result = h * float(dz_um_per_gray) + float(z0_um)
    return result.astype(np.float32, copy=False), {
        "height_units": "um_from_gray",
        "applied_scale": True,
        "dz_um_per_gray": float(dz_um_per_gray),
        "z0_um": float(z0_um),
    }


def inlier_mask_percentile(
    h: np.ndarray,
    low: float = 1.0,
    high: float = 99.0,
) -> np.ndarray:
    """Mask finite values inside [low, high] percentiles for robust plane fitting."""
    h = np.asarray(h, dtype=float)
    finite = np.isfinite(h)
    if not finite.any():
        return np.zeros_like(h, dtype=bool)
    p_low, p_high = np.nanpercentile(h[finite], [low, high])
    return finite & (h >= p_low) & (h <= p_high)


def subtract_plane(
    h: np.ndarray,
    xy_um: float,
    mask: np.ndarray,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Fit and subtract plane on masked pixels; returns (detrended_map, (a,b,c))."""
    h = np.asarray(h, dtype=float)
    if mask.sum() < 10:
        return h.astype(np.float32, copy=False), (0.0, 0.0, 0.0)

    yy, xx = np.indices(h.shape)
    x = (xx[mask] * xy_um).ravel()
    y = (yy[mask] * xy_um).ravel()
    z = h[mask].ravel()

    x0 = float(np.mean(x))
    y0 = float(np.mean(y))
    A = np.column_stack([x - x0, y - y0, np.ones_like(x)])
    coeff, *_ = lstsq(A, z, rcond=None)
    a, b, c_centered = float(coeff[0]), float(coeff[1]), float(coeff[2])

    plane = a * ((xx * xy_um) - x0) + b * ((yy * xy_um) - y0) + c_centered
    c_uncentered = c_centered - a * x0 - b * y0

    return (h - plane).astype(np.float32, copy=False), (a, b, c_uncentered)


def nan_aware_gaussian_1d(
    arr: np.ndarray,
    sigma: float,
    mode: str = "nearest",
) -> np.ndarray:
    """Gaussian-filter a 1D array with normalized convolution to ignore NaNs."""
    arr = np.asarray(arr, dtype=np.float32)
    if sigma <= 0:
        return arr.copy()
    m = np.isfinite(arr).astype(np.float32)
    x0 = np.where(m > 0, arr, 0.0)
    num = gaussian_filter1d(x0, sigma=sigma, mode=mode)
    den = gaussian_filter1d(m, sigma=sigma, mode=mode)
    out = np.full_like(arr, np.nan, dtype=np.float32)
    np.divide(num, den, out=out, where=den > 1e-12)
    return out


def rotate_and_crop(h: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a height map and crop to the largest valid inscribed rectangle."""
    if not np.isfinite(angle_deg) or abs(angle_deg) < 1e-12:
        return h

    hr = rotate(
        h,
        float(angle_deg),
        reshape=False,
        order=1,
        mode="constant",
        cval=np.nan,
    ).astype(np.float32, copy=False)

    ny, nx = hr.shape
    t = abs(float(angle_deg)) % 180.0
    if t > 90.0:
        t = 180.0 - t
    th = np.deg2rad(t)
    c, s = float(np.cos(th)), float(np.sin(th))

    rh = max(1, int(np.floor(ny * c - nx * s)))
    rw = max(1, int(np.floor(nx * c - ny * s)))
    y0 = max(0, (ny - rh) // 2)
    x0 = max(0, (nx - rw) // 2)

    return hr[y0 : y0 + rh, x0 : x0 + rw]


def estimate_pitch_angle_fft(
    h_um: np.ndarray,
    xy_um: float,
    config: AnalyzerConfig = CONFIG,
) -> Tuple[float, float, Dict[str, Any]]:
    """Estimate groove pitch/angle from FFT peak within bandpass; returns (pitch, angle, qc)."""
    h = np.asarray(h_um, dtype=np.float32)
    finite = np.isfinite(h)
    if not finite.any():
        return float("nan"), float("nan"), {"fft_valid": False}

    fill = float(np.nanmedian(h[finite]))
    h0 = np.where(finite, h, fill).astype(np.float32, copy=False)
    h0 = h0 - float(np.mean(h0))

    ny, nx = h0.shape
    if config.fft_use_hann and ny >= 4 and nx >= 4:
        window = np.hanning(ny).astype(np.float32)[:, None]
        window = window * np.hanning(nx).astype(np.float32)[None, :]
        h0 = h0 * window

    F = np.fft.fftshift(np.fft.fft2(h0))
    P = np.abs(F) ** 2

    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=float(xy_um)))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=float(xy_um)))
    FX, FY = np.meshgrid(fx, fy)
    fmag = np.sqrt(FX**2 + FY**2)

    cy, cx = ny // 2, nx // 2
    yy, xx = np.ogrid[:ny, :nx]
    dc_r = int(config.dc_exclusion_radius_px)
    dist_px = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    lo_um, hi_um = config.pitch_range_um
    band = (fmag >= 1.0 / float(hi_um)) & (fmag <= 1.0 / float(lo_um))
    mask = (dist_px >= max(1, dc_r)) & band

    if not mask.any():
        return float("nan"), float("nan"), {"fft_valid": False}

    Pm = np.where(mask, P, 0.0)
    py, px = np.unravel_index(int(np.argmax(Pm)), Pm.shape)
    kx = float(FX[py, px])
    ky = float(FY[py, px])
    f = float(np.hypot(kx, ky))

    if not (np.isfinite(f) and f > 0):
        return float("nan"), float("nan"), {"fft_valid": False}

    pitch_um = float(1.0 / f)
    angle_deg = float(((math.degrees(math.atan2(ky, kx)) + 90.0) % 180.0) - 90.0)

    return (
        pitch_um,
        angle_deg,
        {
            "fft_valid": True,
            "fft_pitch_um": pitch_um,
            "fft_angle_deg": angle_deg,
        },
    )


def sample_mean_profile(
    h: np.ndarray,
    n_stripes: int = 20,
    stripe_hw: int = 10,
) -> np.ndarray:
    """Compute stripe-averaged 1D profile across columns (NaN-tolerant)."""
    h = np.asarray(h, dtype=np.float32)
    ny, nx = h.shape
    if ny < 2 * stripe_hw + 1:
        return np.nanmean(h, axis=0).astype(np.float32, copy=False)

    rows = np.linspace(stripe_hw, ny - stripe_hw - 1, n_stripes, dtype=int)
    profs: List[np.ndarray] = []

    for r in rows:
        stripe = h[r - stripe_hw : r + stripe_hw + 1, :]
        if np.isnan(stripe).all():
            continue
        prof = np.nanmean(stripe, axis=0).astype(np.float32, copy=False)

        if np.isnan(prof).any() and np.isfinite(prof).any():
            idx = np.arange(nx)
            good = np.isfinite(prof)
            prof = np.interp(idx, idx[good], prof[good]).astype(np.float32)

        profs.append(prof)

    if not profs:
        return np.full(nx, np.nan, dtype=np.float32)
    return np.mean(np.stack(profs, axis=0), axis=0).astype(np.float32, copy=False)


def detect_peaks_valleys(
    profile: np.ndarray,
    pitch_px: float,
    config: AnalyzerConfig = CONFIG,
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect peaks and valleys in a 1D profile; returns (peaks, valleys) indices."""
    profile = np.asarray(profile, dtype=np.float32)
    if profile.size < 5 or not np.isfinite(pitch_px) or pitch_px <= 1:
        return np.array([], dtype=int), np.array([], dtype=int)

    amp = float(np.nanpercentile(profile, 99.0) - np.nanpercentile(profile, 1.0))
    if not np.isfinite(amp) or amp <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    prom = float(config.prominence_frac * amp)
    dist = max(1, int(config.min_peak_distance_frac * float(pitch_px)))
    peaks, _ = find_peaks(profile, distance=dist, prominence=prom)
    valleys, _ = find_peaks(-profile, distance=dist, prominence=prom)
    return peaks.astype(int), valleys.astype(int)


def measure_pitch_per_groove(
    h_aligned: np.ndarray,
    xy_um: float,
    pitch_px_seed: float,
    *,
    outdir: Optional[Path] = None,
    file_stem: Optional[str] = None,
    config: AnalyzerConfig = CONFIG,
) -> MeasurementResult:
    """Measure groove pitch from peak-to-peak distances; returns MeasurementResult."""
    avg_profile = sample_mean_profile(h_aligned)
    if not np.isfinite(avg_profile).any():
        return MeasurementResult(float("nan"))

    lambda_c_um = float(config.lambda_c_factor * pitch_px_seed * xy_um)
    sigma_px = iso16610_sigma_px(lambda_c_um, xy_um)
    detect_profile = nan_aware_gaussian_1d(avg_profile, sigma=sigma_px, mode="nearest")

    L = int(math.ceil(lambda_c_um / xy_um))
    offset = 0
    if detect_profile.size > 2 * L:
        detect_profile = detect_profile[L:-L]
        offset = L

    peaks, _ = detect_peaks_valleys(detect_profile, pitch_px_seed, config)
    peaks = peaks + offset
    if peaks.size < 2:
        return MeasurementResult(float("nan"))

    x_um = np.arange(avg_profile.size, dtype=float) * float(xy_um)
    pitches = np.diff(x_um[peaks]).astype(float)
    clean = pitches

    if clean.size < 2:
        val = float(np.mean(clean)) if clean.size == 1 else float("nan")
        return MeasurementResult(val, n=int(clean.size))

    mean = float(np.mean(clean))
    std = float(np.std(clean, ddof=1))
    u_a = float(std / math.sqrt(clean.size))
    u_b = type_b_resolution(xy_um)

    row_pitch_values: list[float] = []
    for row in np.asarray(h_aligned, dtype=float):
        if not np.any(np.isfinite(row)):
            continue
        row_profile = nan_aware_gaussian_1d(row, sigma_px, mode="nearest")
        if row_profile.size > 2 * L:
            row_profile = row_profile[L:-L]
            row_offset = L
        else:
            row_offset = 0

        row_peaks, _ = detect_peaks_valleys(row_profile, pitch_px_seed, config)
        row_peaks = row_peaks + row_offset
        if row_peaks.size < 2:
            continue

        row_pitch = np.diff(x_um[row_peaks])
        if len(row_pitch) >= 8:
            row_clean = row_pitch[tukey_fence_mask(row_pitch, k=config.tukey_k)]
        else:
            row_clean = row_pitch

        row_pitch_values.extend(row_clean.tolist())

    if row_pitch_values:
        arr = np.asarray(row_pitch_values, dtype=float)
        rms_fov = float(np.sqrt(np.mean((arr - mean) ** 2)))
    else:
        rms_fov = float("nan")

    if outdir is not None and file_stem is not None and row_pitch_values:
        pitch_arr = np.asarray(row_pitch_values, dtype=float)
        if config.generate_plots and pitch_arr.size >= 10:
            plot_histogram(
                pitch_arr,
                outdir / f"{file_stem}_pitch_hist",
                "Pitch (µm)",
                "Pitch distribution",
            )

        with (outdir / f"{file_stem}_pitches_um.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            w = csv.writer(f)
            w.writerow(["pitch_um"])
            for v in pitch_arr:
                if np.isfinite(v):
                    w.writerow([float(v)])

    return MeasurementResult(mean, u_a=u_a, u_b=u_b, n=int(clean.size), rmse=rms_fov)


def measure_depth_per_groove(
    h_aligned: np.ndarray,
    xy_um: float,
    dz_um: float,
    pitch_px_seed: float,
    *,
    outdir: Optional[Path] = None,
    file_stem: Optional[str] = None,
    config: AnalyzerConfig = CONFIG,
) -> MeasurementResult:
    """Measure groove depth from peaks vs adjacent valleys; returns MeasurementResult."""
    avg_profile = sample_mean_profile(h_aligned)
    if not np.isfinite(avg_profile).any():
        return MeasurementResult(float("nan"))

    raw = (avg_profile - float(np.nanmin(avg_profile))).astype(np.float32, copy=False)

    lambda_c_um = float(config.lambda_c_factor * pitch_px_seed * xy_um)
    sigma_px = iso16610_sigma_px(lambda_c_um, xy_um)
    detect_profile = nan_aware_gaussian_1d(raw, sigma=sigma_px, mode="nearest")

    L = int(math.ceil(lambda_c_um / xy_um))
    if detect_profile.size > 2 * L:
        detect_profile = detect_profile[L:-L]
        raw = raw[L:-L]

    peaks, valleys = detect_peaks_valleys(detect_profile, pitch_px_seed, config)
    if peaks.size == 0 or valleys.size < 2:
        return MeasurementResult(float("nan"))

    depths: List[float] = []
    for p in peaks:
        v_left = valleys[valleys < p]
        v_right = valleys[valleys > p]
        if v_left.size == 0 or v_right.size == 0:
            continue
        vl, vr = int(v_left[-1]), int(v_right[0])
        peak_h = float(raw[p])
        valley_h = 0.5 * (float(raw[vl]) + float(raw[vr]))
        depths.append(peak_h - valley_h)

    if len(depths) == 0:
        return MeasurementResult(float("nan"))

    d = np.asarray(depths, dtype=float)
    clean = d

    if clean.size < 2:
        mean = float(np.mean(clean)) if clean.size == 1 else float("nan")
        return MeasurementResult(mean, n=int(clean.size))

    mean = float(np.mean(clean))
    std = float(np.std(clean, ddof=1))
    u_a = float(std / math.sqrt(clean.size))

    u_b_res = type_b_resolution(dz_um)
    u_b_cal = float(config.u_rel_calibration * mean) if np.isfinite(mean) else 0.0
    u_b = float(math.sqrt(u_b_res**2 + u_b_cal**2))

    row_depth_values: list[float] = []
    for row in np.asarray(h_aligned, dtype=float):
        if not np.any(np.isfinite(row)):
            continue
        row_profile = nan_aware_gaussian_1d(row, sigma_px, mode="nearest")
        if row_profile.size > 2 * L:
            row_profile = row_profile[L:-L]
            row = row[L:-L]

        row_peaks, row_valleys = detect_peaks_valleys(
            row_profile, pitch_px_seed, config
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
            peak_h = float(row[p])
            valley_h = 0.5 * (float(row[vl]) + float(row[vr]))
            row_depths.append(peak_h - valley_h)

        if not row_depths:
            continue
        row_arr = np.asarray(row_depths, dtype=float)
        if len(row_depths) >= 8:
            row_clean = row_arr[tukey_fence_mask(row_arr, k=config.tukey_k)]
        else:
            row_clean = row_arr

        row_depth_values.extend(row_clean.tolist())

    # The per-row pool is kept only for diagnostic outputs (histogram + CSV). It
    # is NOT used for the reported value or u_a/n: rows one pixel apart sample the
    # same groove and are not independent, so the effective replicate count is the
    # number of grooves (the averaged profile above).
    if row_depth_values and outdir is not None and file_stem is not None:
        arr = np.asarray(row_depth_values, dtype=float)
        if config.generate_plots and arr.size >= 10:
            plot_histogram(
                arr,
                outdir / f"{file_stem}_depth_hist",
                "Depth (µm)",
                "Depth distribution",
            )

        with (outdir / f"{file_stem}_depths_um.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            w = csv.writer(f)
            w.writerow(["depth_um"])
            for v in arr:
                if np.isfinite(v):
                    w.writerow([float(v)])

    return MeasurementResult(mean, u_a=u_a, u_b=u_b, n=int(clean.size), rmse=std)


def plot_height_map(h: np.ndarray, xy_um: float, outpath: Path, title: str) -> None:
    """Save height map plot as PNG and PDF to `outpath` (stem)."""
    if plt is None:
        return

    vals = h[np.isfinite(h)]
    if vals.size < 10:
        return

    vmin, vmax = np.nanpercentile(vals, [1.0, 99.0])
    set_pub_style()

    extent = [0.0, h.shape[1] * xy_um, 0.0, h.shape[0] * xy_um]
    fig, ax = plt.subplots(figsize=FIGSIZE_IN, dpi=CONFIG.plot_dpi)
    im = ax.imshow(
        h,
        origin="lower",
        extent=extent,
        vmin=float(vmin),
        vmax=float(vmax),
        cmap="viridis",
    )
    ax.set_xlabel("X [µm]")
    ax.set_ylabel("Y [µm]")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Height [µm]")

    fig.tight_layout()
    fig.savefig(outpath.with_suffix(".png"), dpi=CONFIG.plot_dpi)
    fig.savefig(outpath.with_suffix(".pdf"))
    plt.close(fig)


def plot_profile_annotated(
    profile: np.ndarray,
    xy_um: float,
    peaks: np.ndarray,
    valleys: np.ndarray,
    outpath: Path,
) -> None:
    """Save annotated 1D profile plot as PNG and PDF to `outpath` (stem)."""
    if plt is None:
        return

    set_pub_style()
    x = np.arange(profile.size, dtype=float) * float(xy_um)

    fig, ax = plt.subplots(figsize=FIGSIZE_IN, dpi=CONFIG.plot_dpi)
    ax.plot(x, profile, linewidth=1.0)

    if peaks.size:
        ax.plot(x[peaks], profile[peaks], "o", markersize=4, label="Peaks")
    if valleys.size:
        ax.plot(x[valleys], profile[valleys], "v", markersize=4, label="Valleys")

    ax.set_xlabel("X [µm]")
    ax.set_ylabel("Height [µm]")
    ax.set_title("Stripe-averaged profile")
    ax.grid(True, alpha=0.3)

    if peaks.size or valleys.size:
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(outpath.with_suffix(".png"), dpi=CONFIG.plot_dpi)
    fig.savefig(outpath.with_suffix(".pdf"))
    plt.close(fig)


def plot_histogram(
    x: np.ndarray,
    outpath: Path,
    xlabel: str,
    title: str,
    bins: int = 40,
) -> None:
    """Save histogram plot as PNG and PDF to `outpath` (stem)."""
    if plt is None:
        return

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return

    set_pub_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_IN, dpi=CONFIG.plot_dpi)
    ax.hist(x, bins=bins, edgecolor="black", alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath.with_suffix(".png"), dpi=CONFIG.plot_dpi)
    fig.savefig(outpath.with_suffix(".pdf"))
    plt.close(fig)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write a list of dict rows to CSV at `path` (no-op if rows is empty)."""
    if not rows:
        return
    fieldnames: List[str] = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        w.writeheader()
        w.writerows(rows)



def analyze_file(
    fpath: Path,
    xy_um: float,
    dz_um_per_gray: float,
    z0_um: float,
    outdir: Optional[Path] = None,
    config: AnalyzerConfig = CONFIG,
) -> Dict[str, Any]:
    """Analyze one TIFF and write per-file outputs; returns results dict (or error key)."""
    if outdir is None:
        outdir = fpath.parent / f"{fpath.stem}_proc"
    outdir.mkdir(parents=True, exist_ok=True)

    h_raw, load_meta = load_height_map(fpath)
    h_um, scale_meta = to_height_um(h_raw, dz_um_per_gray, z0_um)

    valid_frac = (
        float(np.isfinite(h_um).sum() / h_um.size) if h_um.size else float("nan")
    )

    mask = inlier_mask_percentile(h_um, 1.0, 99.0)
    h_detr, (a, b, c) = subtract_plane(h_um, xy_um, mask)

    fft_pitch_um, fft_angle_deg, fft_qc = estimate_pitch_angle_fft(
        h_detr, xy_um, config=config
    )

    if not np.isfinite(fft_pitch_um):
        _, qc_reasons = evaluate_qc(valid_frac, float("nan"), config)
        return {
            "file": fpath.name,
            "error": "Failed to estimate pitch (FFT)",
            "valid_pixel_frac": valid_frac,
            "qc_pass": False,
            "qc_reasons": ";".join(
                r for r in ("fft_pitch_failed", qc_reasons) if r
            ),
            **load_meta,
            **scale_meta,
            **fft_qc,
        }

    pitch_px_seed = float(fft_pitch_um / xy_um)

    h_aligned = nan_aware_median_filter(h_detr,size=5)
    rotation_applied = 0.0
    if (
        np.isfinite(fft_angle_deg)
        and abs(fft_angle_deg) > config.rotate_if_abs_angle_deg_gt
    ):
        h_aligned = rotate_and_crop(h_aligned, float(fft_angle_deg))
        rotation_applied = float(fft_angle_deg)

    pitch_res = measure_pitch_per_groove(
        h_aligned,
        xy_um,
        pitch_px_seed,
        outdir=outdir,
        file_stem=fpath.stem,
        config=config,
    )

    depth_res = measure_depth_per_groove(
        h_aligned,
        xy_um,
        dz_um_per_gray,
        pitch_px_seed,
        outdir=outdir,
        file_stem=fpath.stem,
        config=config,
    )

    avg_profile = sample_mean_profile(h_aligned)
    if np.isfinite(avg_profile).any():
        avg_profile0 = avg_profile - float(np.nanmin(avg_profile))
    else:
        avg_profile0 = avg_profile

    lambda_c_um = float(config.lambda_c_factor * fft_pitch_um)
    sigma_px = iso16610_sigma_px(lambda_c_um, xy_um)
    detect_profile = nan_aware_gaussian_1d(avg_profile0, sigma=sigma_px, mode="nearest")
    peaks, valleys = detect_peaks_valleys(detect_profile, pitch_px_seed, config=config)

    if config.generate_plots:
        plot_height_map(
            h_um, xy_um, outdir / f"{fpath.stem}_height_map", "Height map"
        )

        if np.isfinite(avg_profile0).any():
            plot_profile_annotated(
                avg_profile0.astype(np.float32),
                xy_um,
                peaks,
                valleys,
                outdir / f"{fpath.stem}_avg_profile_annotated",
            )

    qc_pass, qc_reasons = evaluate_qc(
        valid_frac, float(pitch_res.n), config
    )
    if not (
        np.isfinite(pitch_res.value) or np.isfinite(depth_res.value)
    ):
        qc_pass = False
        qc_reasons = ";".join(
            r for r in ("no_measurable_grooves", qc_reasons) if r
        )

    results: Dict[str, Any] = {
        "file": fpath.name,
        "folder": fpath.parent.name,
        "analyzer_version": __version__,
        "analyzer_sha256": ANALYZER_SHA256,
        "timestamp_utc": now_utc(),
        "xy_um_per_px": float(xy_um),
        "dz_um_per_gray": (
            float(dz_um_per_gray) if np.isfinite(dz_um_per_gray) else float("nan")
        ),
        "z0_um": float(z0_um) if np.isfinite(z0_um) else float("nan"),
        "valid_pixel_frac": valid_frac,
        "fft_pitch_um": float(fft_pitch_um),
        "fft_angle_deg": float(fft_angle_deg),
        "rotation_applied_deg": rotation_applied,
        "plane_a": float(a),
        "plane_b": float(b),
        "plane_c": float(c),
        "pitch_um": float(pitch_res.value),
        "pitch_u_a_um": float(pitch_res.u_a),
        "pitch_u_b_um": float(pitch_res.u_b),
        "pitch_u_combined_um": float(pitch_res.u_combined),
        "pitch_n": int(pitch_res.n),
        "pitch_rmse_um": float(pitch_res.rmse),
        "depth_um": float(depth_res.value),
        "depth_u_a_um": float(depth_res.u_a),
        "depth_u_b_um": float(depth_res.u_b),
        "depth_u_combined_um": float(depth_res.u_combined),
        "depth_n": int(depth_res.n),
        "depth_rmse_um": float(depth_res.rmse),
        "n_grooves": int(pitch_res.n),
        "qc_pass": qc_pass,
        "qc_reasons": qc_reasons,
        **load_meta,
        **scale_meta,
        **fft_qc,
    }

    with (outdir / f"{fpath.stem}_analysis_summary.txt").open(
        "w", encoding="utf-8"
    ) as f:
        f.write(f"Groove Analyzer (Holo) v{__version__}\n")
        f.write(f"SHA-256: {ANALYZER_SHA256}\n{'=' * 60}\n")
        f.write(f"File: {fpath}\nTimestamp: {results['timestamp_utc']}\n\n")
        f.write("PRIMARY MEASUREMENTS (with GUM-style uncertainty):\n")
        if np.isfinite(pitch_res.value):
            f.write(
                f"  Pitch: {pitch_res.value:.3f} ± {pitch_res.u_combined:.3f} µm "
                f"(n={pitch_res.n})\n"
            )
        else:
            f.write("  Pitch: NaN\n")
        if np.isfinite(depth_res.value):
            f.write(
                f"  Depth: {depth_res.value:.3f} ± {depth_res.u_combined:.3f} µm "
                f"(n={depth_res.n})\n"
            )
        else:
            f.write("  Depth: NaN\n")
        f.write("\nQUALITY CONTROL (reported, not applied):\n")
        f.write(
            f"  Valid pixels: {valid_frac * 100:.1f}% "
            f"(threshold {config.qc_min_valid_frac * 100:.0f}%)\n"
        )
        f.write(
            f"  Grooves measured: {pitch_res.n} "
            f"(threshold {config.qc_min_grooves})\n"
        )
        f.write(f"  qc_pass: {qc_pass}\n")
        f.write(f"  qc_reasons: {qc_reasons or '-'}\n")
        f.write(
            "\nNOTE: Gel thickness/height is not estimated from holographic "
            "phase imaging in this pipeline.\n"
        )

    meta = {
        "timestamp_utc": results["timestamp_utc"],
        "analyzer_version": __version__,
        "analyzer_sha256": ANALYZER_SHA256,
        "input_file": str(fpath),
        "load_meta": load_meta,
        "scale_meta": scale_meta,
        "fft_qc": fft_qc,
        "qc": {
            "qc_pass": bool(qc_pass),
            "qc_reasons": qc_reasons,
            "valid_pixel_frac": float(valid_frac),
            "n_grooves": int(pitch_res.n),
        },
        "config": {k: getattr(config, k) for k in config.__dataclass_fields__.keys()},
    }
    with (outdir / f"{fpath.stem}_run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    write_csv(outdir / f"{fpath.stem}_results.csv", [results])

    return results


def analyze_batch(
    input_path: Path,
    xy_um: float,
    dz_um_per_gray: float,
    z0_um: float,
    output_dir: Optional[Path] = None,
    config: AnalyzerConfig = CONFIG,
) -> List[Dict[str, Any]]:
    """Analyze a TIFF or directory of TIFFs and write batch summaries; returns results list."""
    if input_path.is_file():
        files = [input_path]
        root = output_dir if output_dir is not None else input_path.parent
    else:
        files = sorted(
            [p for p in input_path.rglob("*") if p.suffix.lower() in {".tif", ".tiff"}]
        )
        root = output_dir if output_dir is not None else input_path
    root.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "timestamp_utc": now_utc(),
        "analyzer_version": __version__,
        "analyzer_sha256": ANALYZER_SHA256,
        "input": str(input_path),
        "n_files": len(files),
        "calibration": {
            "xy_um_per_px": float(xy_um),
            "dz_um_per_gray": float(dz_um_per_gray),
            "z0_um": float(z0_um),
        },
    }
    with (root / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    all_results: List[Dict[str, Any]] = []
    for i, p in enumerate(files, 1):
        LOGGER.info("[%d/%d] %s", i, len(files), p.name)
        try:
            outdir = (
                (p.parent / f"{p.stem}_proc")
                if input_path.is_dir()
                else (root / f"{p.stem}_proc")
            )
            all_results.append(
                analyze_file(
                    p,
                    xy_um=xy_um,
                    dz_um_per_gray=dz_um_per_gray,
                    z0_um=z0_um,
                    outdir=outdir,
                    config=config,
                )
            )
        except Exception as exc:
            LOGGER.exception("Failed %s", p)
            all_results.append(
                {
                    "file": p.name,
                    "error": str(exc),
                    "qc_pass": False,
                    "qc_reasons": "analysis_exception",
                }
            )

    write_csv(root / "groove_recap.csv", all_results)

    n_flagged = sum(1 for r in all_results if r.get("qc_pass") is False)
    if n_flagged:
        LOGGER.warning(
            "QC flagged %d/%d files (reported only, none excluded)",
            n_flagged,
            len(all_results),
        )

    return all_results


def gui_main() -> None:
    """Run GUI workflow (file/folder selection, calibration prompts) and execute analysis."""
    if tk is None or filedialog is None or messagebox is None or simpledialog is None:
        raise RuntimeError("tkinter is required for GUI mode")

    root = tk.Tk()
    root.withdraw()

    is_folder = messagebox.askyesno(
        "Groove Analyzer (Holo)",
        "Analyze folder? (No = single file)",
    )
    if is_folder:
        path = filedialog.askdirectory(title="Select folder")
    else:
        path = filedialog.askopenfilename(
            title="Select TIFF",
            filetypes=[("TIFF", "*.tif *.tiff")],
        )
    if not path:
        raise SystemExit("No selection")

    xy_um = simpledialog.askfloat(
        "Calibration",
        "XY pixel size (µm/px):",
        minvalue=0.001,
        initialvalue=0.54,
    )
    dz_um = simpledialog.askfloat(
        "Calibration",
        "Height scale (µm per gray count):",
        minvalue=1e-6,
        initialvalue=0.294,
    )
    z0_um = simpledialog.askfloat(
        "Calibration",
        "Z0 offset (µm) at gray=0:",
        minvalue=-1e9,
        initialvalue=-25.0,
    )
    if None in (xy_um, dz_um, z0_um):
        raise SystemExit("Cancelled")

    results = analyze_batch(
        Path(path),
        xy_um=float(xy_um),
        dz_um_per_gray=float(dz_um),
        z0_um=float(z0_um),
    )
    n_ok = sum(1 for r in results if "error" not in r)
    messagebox.showinfo("Complete", f"Analyzed {n_ok}/{len(results)} files")


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse CLI args and run GUI or batch analysis."""
    p = argparse.ArgumentParser(
        description=f"Groove Analyzer (Holo) v{__version__}",
    )
    p.add_argument(
        "-i",
        "--input",
        type=Path,
        help="Input TIFF or directory (GUI if omitted)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output directory (default: input folder)",
    )
    p.add_argument(
        "--xy",
        type=float,
        help="XY pixel size (µm/px)",
    )
    p.add_argument(
        "--dz",
        type=float,
        help="Height scale (µm per gray count) for uint8/uint16 TIFFs",
    )
    p.add_argument(
        "--z0",
        type=float,
        default=0.0,
        help="Z0 offset (µm) at gray=0 for integer TIFFs",
    )

    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plots",
    )
    p.add_argument(
        "--gui",
        action="store_true",
        help="Launch GUI",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
    )
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.gui or args.input is None:
        gui_main()
        return

    if args.xy is None:
        p.error("--xy is required for CLI")
    if args.dz is None:
        p.error("--dz is required for CLI when input TIFF is integer-coded height")

    cfg = AnalyzerConfig(generate_plots=not bool(args.no_plots))
    analyze_batch(
        args.input,
        xy_um=float(args.xy),
        dz_um_per_gray=float(args.dz),
        z0_um=float(args.z0),
        output_dir=args.output,
        config=cfg,
    )


if __name__ == "__main__":
    main()
