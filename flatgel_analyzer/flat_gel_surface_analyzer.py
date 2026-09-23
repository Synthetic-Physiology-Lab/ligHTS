#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog
except Exception:
    tk = None
    filedialog = None
    messagebox = None
    simpledialog = None

import matplotlib as _mpl

_mpl.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter, MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import distance_transform_edt, gaussian_filter

try:
    import cv2
except Exception as exc:
    raise SystemExit("OpenCV (cv2) is required. Install: pip install opencv-python") from exc

try:
    import nd2

    ND2_AVAILABLE = True
except Exception:
    ND2_AVAILABLE = False

try:
    from aicsimageio import AICSImage

    AICS_AVAILABLE = True
except Exception:
    AICS_AVAILABLE = False

ANALYSIS_NAME = "gel_top_surface_uniformity"
ANALYSIS_VERSION = "2026.07.22"

SAVE_PNGS: bool = bool(int(os.environ.get("GEL_SURF_SAVE_PNGS", "1")))
SAVE_HEIGHT_PNG: bool = True
SAVE_BANDPASS_RESIDUAL_PNG: bool = True
PNG_DPI: int = 300

WELL_EDGE_MM: float = 3.3
WELL_SIZE_TOL_FRAC: float = 0.15
WELL_CORNER_RADIUS_FRAC: float = 0.10

# ROI scaling to ensure analysis in performed on gel-containing ROI (0.90 = analyze central 90%)
MASK_SCALE_FRAC: float = 0.90

# When stacks contain non-spatial axes, these indices are selected (0-based).
ND2_SELECT_INDEX: dict[str, int] = {"T": 0, "C": 0, "P": 0, "S": 0}
TIFF_NONSPATIAL_INDEX: int = 0

# Truncation detection: require boundary slice signal to be a fraction of MIP signal.
TRUNC_EDGE_REL_TO_MIP: float = 0.5

CORE_MARGIN_PX: int = 10
LFILTER_LAMBDA_C_UM: float = 120.0
LAMBDA_S_OVERRIDE_UM: float = float("nan")
INLIER_LO_PCT: float = 5.0
INLIER_HI_PCT: float = 95.0
OUTLIER_MAD_K: float = 3.5
BEAD_DIAMETER_NOMINAL_UM: float = 0.20
S_FILTER_LAMBDA_S_UM_DEFAULT: float = 25.0
S_FILTER_LAMBDA_MULT_FROM_DIAM: float = 2.5
S_FILTER_MIN_SIGMA_PX: float = 0.6
MIN_ENVELOPE_RADIUS_PX: int = 1
HEIGHT_COLORMAP_PERCENTILES: tuple[float, float] | None = (5.0, 95.0)

CSV_COLUMNS_PRIMARY: list[str] = [
    "File",
    "TopZ_Median_um",          # robust GT-comparable apparent optical top-surface height
    "Sq_SL_um",                # ISO 25178-2-aligned areal RMS on the S-L surface (no outlier rejection)
    "TopZ_LFilteredMean_um",
    "BandpassStd_um",
    "TopZ_LFilteredStd_um",
    "TiltApplied",
    "TiltAngle_deg",
    "Plane_a_um_per_mm",
    "Plane_b_um_per_mm",
    "Plane_c_um",
    "Valid_frac",
    "Core_valid_frac",
]

CSV_COLUMNS_SECONDARY: list[str] = [
    "Well_px_n",
    "Valid_px_n",
    "Core_px_n",
    "Core_valid_px_n",
    "Plane_R2",
    "Z_slices",
    "TruncMode",
]

CSV_COLUMNS_METADATA: list[str] = [
    "XY_um_per_px",
    "Z_step_um",
    "Z0_um",
    "LambdaC_um",
    "LambdaS_um",
    "BeadDiamEff_um",
    "MaskScaleFrac",
    "AnalysisInfo",
]

CSV_COLUMNS = CSV_COLUMNS_PRIMARY + CSV_COLUMNS_SECONDARY + CSV_COLUMNS_METADATA


def utc_now_iso() -> str:
    """Return current UTC timestamp as ISO 8601 string (seconds resolution)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: str) -> str:
    """Compute SHA-256 hex digest for a file at `path` (streamed, constant memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_run_metadata(
    *,
    analysis_name: str,
    analysis_version: str,
    analysis_sha256: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a run metadata dict (versions, platform, parameters) for provenance.
    Returned object is JSON-serializable and intended to be embedded in outputs.
    """
    return {
        "analysis_name": analysis_name,
        "analysis_version": analysis_version,
        "analysis_sha256": analysis_sha256,
        "timestamp_utc": utc_now_iso(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "tifffile": tifffile.__version__,
            "matplotlib": _mpl.__version__,
            "scipy": getattr(sys.modules.get("scipy"), "__version__", "unknown"),
            "opencv": getattr(cv2, "__version__", "unknown"),
        },
        "params": params or {},
        "notes": [
            "z=0 is defined by acquisition using beads-on-glass fiducials in nearby wells.",
            "Metrics are computed on the top-surface bead layer; tilt removal uses the fitted plane a,b terms.",
            "BandpassStd_um is a band-pass residual dispersion metric (areal RMS-like).",
            f"When a well is segmented, ROI may be scaled by {MASK_SCALE_FRAC:.0%} to exclude retracted gel edges; full-field stacks are not scaled.",
        ],
    }


def _json_sanitize(obj: Any) -> Any:
    """Recursively replace non-finite floats (NaN/Inf) with None for strict JSON."""
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def round_for_csv(value: Any, decimals: int = 2) -> Any:
    """Round finite floats for CSV output; return non-floats unchanged."""
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return value
        return round(float(value), decimals)
    return value


def get_float_attr(elem: Any, attr: str) -> float | None:
    """Parse XML attribute `attr` from `elem` as float; return None on missing/invalid."""
    v = elem.get(attr) if elem is not None else None
    try:
        return float(v) if v is not None else None
    except Exception:
        return None


def percentile_in_mask(a: np.ndarray, mask: np.ndarray, q: float) -> float:
    """Compute percentile `q` over finite values in `a` restricted by `mask` (fallback: all)."""
    arr = a[mask] if np.any(mask) else a.ravel()
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, q)) if arr.size else float("nan")


# I/O FUNCTIONS


def read_tiff(path: str) -> np.ndarray:
    """Read a TIFF stack and return a ZYX float32 array (best-effort axis normalization)."""
    arr = None
    axes = None
    try:
        with tifffile.TiffFile(path) as tfh:
            series = tfh.series[0]
            axes = getattr(series, "axes", None)
            arr = np.asarray(series.asarray())
    except Exception:
        arr = np.asarray(tifffile.imread(path))
    if arr is None:
        raise RuntimeError(f"Failed to read TIFF: {path}")
    if axes and isinstance(axes, str) and len(axes) == arr.ndim:
        axes_u = axes.upper()
        # A plain multi-page TIFF carries no Z axis: tifffile labels the page
        # axis 'I', or 'Q' when it cannot name it. Treat a single such axis as Z,
        # otherwise the page index would be selected like any other non-spatial
        # axis and the stack would collapse to one slice.
        if "Z" not in axes_u:
            generic = [i for i, ax in enumerate(axes_u) if ax in ("I", "Q")]
            if len(generic) == 1:
                i_gen = generic[0]
                logging.info(
                    "TIFF '%s' has axes '%s' and no Z axis; treating page axis "
                    "'%s' (n=%d) as Z.",
                    os.path.basename(path),
                    axes,
                    axes_u[i_gen],
                    int(arr.shape[i_gen]),
                )
                axes_u = axes_u[:i_gen] + "Z" + axes_u[i_gen + 1 :]
            elif len(generic) > 1:
                raise RuntimeError(
                    f"TIFF '{os.path.basename(path)}' has axes '{axes}' with more "
                    "than one generic page axis and no Z axis; cannot tell which "
                    "one is Z. Export OME-TIFF with explicit axes, or convert to a "
                    "ZYX TIFF/ND2."
                )
        slicer = []
        kept_axes: list[str] = []
        for i_ax, ax in enumerate(axes_u):
            if ax in ("Z", "Y", "X"):
                slicer.append(slice(None))
                kept_axes.append(ax)
            else:
                if arr.shape[i_ax] > 1:
                    logging.warning(
                        "TIFF has non-spatial axis %s=%d; selecting index %d",
                        ax,
                        int(arr.shape[i_ax]),
                        int(TIFF_NONSPATIAL_INDEX),
                    )
                slicer.append(int(TIFF_NONSPATIAL_INDEX))
        arr = arr[tuple(slicer)]

        if arr.ndim == 2 and kept_axes == ["Y", "X"]:
            arr = arr[np.newaxis, ...]
            kept_axes = ["Z", "Y", "X"]
        if set(kept_axes) >= {"Z", "Y", "X"} and arr.ndim == 3:
            order = [kept_axes.index("Z"), kept_axes.index("Y"), kept_axes.index("X")]
            arr = np.transpose(arr, order)
        return arr.astype(np.float32, copy=False)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    elif arr.ndim > 3:
        raise RuntimeError(
            f"TIFF '{os.path.basename(path)}' has shape {arr.shape} but no axis metadata; "
            "cannot safely infer ZYX. Export OME-TIFF (with axes) or convert to a ZYX TIFF/ND2."
       )
    if arr.ndim != 3:
        raise RuntimeError(f"TIFF could not be normalized to ZYX. shape={arr.shape} axes={axes}")
    return arr.astype(np.float32)



def read_nd2(path: str) -> np.ndarray:
    """Read an ND2 stack and return a ZYX float32 array (single position/channel use-case)."""
    if ND2_AVAILABLE:
        with nd2.ND2File(path) as f:
            a = f.asarray()
            dims = list(f.sizes.keys())

            for d, sel in ND2_SELECT_INDEX.items():
                if d in f.sizes and int(f.sizes[d]) > 1:
                    logging.warning(
                        "ND2 has axis %s=%d; selecting index %d",
                        d,
                        int(f.sizes[d]),
                        int(sel),
                    )

            # Drop non-spatial axes by selecting configured indices
            slicer = [int(ND2_SELECT_INDEX[d]) if d in ND2_SELECT_INDEX else slice(None) for d in dims]
            a = a[tuple(slicer)]
            dims = [d for d in dims if d not in ND2_SELECT_INDEX]


            # Ensure ZYX
            if a.ndim == 2:
                a = np.expand_dims(a, 0)
                dims = ["Z", "Y", "X"]
            elif a.ndim == 3 and "Z" not in dims and set(dims) >= {"Y", "X"}:
                a = np.expand_dims(a, 0)
                dims = ["Z"] + [d for d in dims if d in ("Y", "X")]

            order = [dims.index(d) for d in ("Z", "Y", "X")]
            return np.transpose(a, order).astype(np.float32, copy=False)

    if AICS_AVAILABLE:
        img = AICSImage(path)
        try:
            return img.get_image_data("ZYX", T=0, S=0, C=0).astype(np.float32)
        finally:
            try:
                img.close()
            except Exception:
                pass

    raise RuntimeError("ND2 reading requires 'nd2' or 'aicsimageio[nd2]'.")


def read_stack(path: str) -> np.ndarray:
    """Dispatch stack loading by extension and return a ZYX float32 volume."""
    ext = os.path.splitext(path.lower())[1]
    if ext in (".tif", ".tiff"):
        return read_tiff(path)
    if ext == ".nd2":
        return read_nd2(path)
    raise ValueError(f"Unsupported extension for {path}")


def infer_voxel_sizes_from_file(path: str) -> tuple[float | None, float | None]:
    """Infer (xy_um_per_px, z_step_um) from image metadata when available.

    Tries OME-TIFF metadata, then ``nd2`` for .nd2 files, then aicsimageio.
    Returns (None, None) if metadata is missing or unreadable.
    """

    try:
        with tifffile.TiffFile(path) as tfh:
            mm = getattr(tfh, "ome_metadata", None)
        if mm:
            root = ET.fromstring(mm)
            px = root.find(".//{http://www.openmicroscopy.org/Schemas/OME/2016-06}Pixels")
            if px is None:
                px = root.find(".//Pixels")
            if px is not None:
                return get_float_attr(px, "PhysicalSizeX"), get_float_attr(px, "PhysicalSizeZ")
    except Exception:
        pass
    if ND2_AVAILABLE and os.path.splitext(path)[1].lower() == ".nd2":
        try:
            with nd2.ND2File(path) as f:
                vs = f.voxel_size()
                sx = float(vs.x) if vs.x else None
                sz = float(vs.z) if vs.z else None
            if sx or sz:
                return sx, sz
        except Exception:
            pass
    try:
        if AICS_AVAILABLE:
            img = AICSImage(path)
            try:
                sx = getattr(img.physical_pixel_sizes, "X", None)
                sz = getattr(img.physical_pixel_sizes, "Z", None)
                return (float(sx) if sx else None), (float(sz) if sz else None)
            finally:
                try:
                    img.close()
                except Exception:
                    pass
    except Exception:
        pass
    return None, None


def rounded_rect_mask(
    shape: tuple[int, int], center: tuple[float, float], side: float, corner: float
) -> np.ndarray:
    """Create a uint8 rounded-rectangle mask (255 inside) for a given image `shape`."""
    h, w = shape
    cx, cy = map(float, center)
    side = float(max(4.0, side))
    r = float(max(1.0, min(corner, side / 2.0)))
    x0, y0 = round(cx - side / 2.0), round(cy - side / 2.0)
    x1, y1 = round(cx + side / 2.0), round(cy + side / 2.0)
    r = round(r)
    msk = np.zeros((h, w), np.uint8)
    cv2.rectangle(msk, (x0 + r, y0), (x1 - r, y1), 255, -1)
    cv2.rectangle(msk, (x0, y0 + r), (x1, y1 - r), 255, -1)
    for xc, yc in [(x0 + r, y0 + r), (x1 - r, y0 + r), (x1 - r, y1 - r), (x0 + r, y1 - r)]:
        cv2.circle(msk, (xc, yc), r, 255, -1)
    return msk


def _cv_find_contours(mask_u8: np.ndarray, mode: int, method: int):
    """Call OpenCV findContours with compatibility across 2/3-return-value variants."""
    res = cv2.findContours(mask_u8, mode, method)
    if len(res) == 2:
        contours, hierarchy = res
    elif len(res) == 3:
        _img, contours, hierarchy = res
    else:
        raise RuntimeError(f"Unexpected cv2.findContours return signature: {len(res)} values")
    return contours, hierarchy


def closed_contour(mask_u8: np.ndarray) -> np.ndarray:
    """Return the largest external contour as Nx2 float32, explicitly closed (last=first)."""
    cnts, _ = _cv_find_contours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return np.zeros((0, 2), np.float32)
    cont = max(cnts, key=cv2.contourArea).squeeze()
    if cont.ndim != 2 or cont.shape[0] < 4:
        return np.zeros((0, 2), np.float32)
    if (cont[0, 0] != cont[-1, 0]) or (cont[0, 1] != cont[-1, 1]):
        cont = np.vstack([cont, cont[0:1, :]])
    return cont.astype(np.float32)


def mask_centroid_xy(mask_u8: np.ndarray) -> tuple[float, float]:
    """Compute (x,y) centroid of a binary mask via image moments (fallback: image center)."""
    m = cv2.moments((mask_u8 > 0).astype(np.uint8))
    if m["m00"] > 0:
        return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])
    h, w = mask_u8.shape
    return (w - 1) / 2.0, (h - 1) / 2.0


def score_well_mask(
    mask_u8: np.ndarray,
    cont_xy: np.ndarray,
    bright: np.ndarray,
    dt: np.ndarray,
    grad: np.ndarray,
    w: int,
    h: int,
) -> float:
    """Score an ROI candidate using bead coverage + edge alignment heuristics."""
    if cont_xy.size == 0:
        return -1e9
    denom = int(bright.sum())
    cover = 0.0 if denom == 0 else float((bright & (mask_u8 > 0)).sum()) / denom
    px = cont_xy[:, 0].clip(0, w - 1).astype(int)
    py = cont_xy[:, 1].clip(0, h - 1).astype(int)
    chamf = float(np.mean(dt[py, px]))
    bgrad = float(np.median(grad[py, px]))
    xs, ys = cont_xy[:, 0], cont_xy[:, 1]
    dist_b = np.minimum.reduce([xs, ys, (w - 1) - xs, (h - 1) - ys])
    bfrac = float(np.mean(dist_b <= 4))
    return (2.0 * cover) + (0.5 * bgrad) - (0.10 * chamf) - (0.50 * bfrac)


def make_rounded_mask(
    shape: tuple[int, int], center: tuple[float, float], side: float, rfrac: float
) -> np.ndarray:
    """Make a rounded-rectangle uint8 ROI mask."""
    s = float(np.clip(side, 4.0, 0.98 * min(shape)))
    r = float(np.clip(rfrac * s, 1.0, s / 2.0))
    return rounded_rect_mask(shape, center, s, r)


def make_base_row(
    *,
    file_name: str,
    z_slices: int,
    xy_um_per_px: float,
    z_step_um: float,
    z0_um: float,
    analysis_info: str,
) -> dict[str, Any]:
    """Initialize a summary row dict with fixed per-file metadata and placeholders."""
    return {
        "File": file_name,
        "Z_slices": int(z_slices),
        "XY_um_per_px": round_for_csv(float(xy_um_per_px)),
        "Z_step_um": round_for_csv(float(z_step_um)),
        "Z0_um": round_for_csv(float(z0_um)),
        "LambdaC_um": round_for_csv(float(LFILTER_LAMBDA_C_UM)),
        "MaskScaleFrac": round_for_csv(float(MASK_SCALE_FRAC)),
        "AnalysisInfo": analysis_info,
    }


def make_row(base_row: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Return a new row dict = base_row + overrides."""
    row = dict(base_row)
    row.update(overrides)
    return row


def scale_mask_and_contour(
    mask_u8: np.ndarray, contour_xy: np.ndarray | None, scale: float
) -> tuple[np.ndarray, np.ndarray | None]:
    """Scale an ROI mask/contour about its centroid to exclude edges while preserving shape."""
    if not np.isfinite(scale) or scale <= 0:
        return mask_u8, contour_xy
    h, w = mask_u8.shape
    cx, cy = mask_centroid_xy(mask_u8)
    mat = cv2.getRotationMatrix2D((cx, cy), 0.0, float(scale))
    scaled = cv2.warpAffine(
        mask_u8, mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    _, scaled = cv2.threshold(scaled.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    if contour_xy is not None and contour_xy.size >= 2:
        pts = np.hstack(
            [contour_xy.astype(np.float32), np.ones((contour_xy.shape[0], 1), np.float32)]
        )
        contour_xy = (pts @ mat.T).astype(np.float32)
    return scaled, contour_xy


def segment_well_from_stack(
    stack: np.ndarray, xy_um_per_px: float
) -> tuple[np.ndarray, np.ndarray]:
    """Segment the well/ROI from a 3D stack using intensity and edge heuristics.
    Returns (well_mask_u8, contour_xy) where mask is uint8 (0/255) and contour is Nx2 pixel coordinates (float32), explicitly closed.
    """
    mip = np.max(stack, axis=0).astype(np.float32)
    h, w = mip.shape
    side_est_px = (WELL_EDGE_MM * 1000.0) / max(xy_um_per_px, 1e-6)
    if side_est_px >= 0.98 * min(h, w):
        return np.full((h, w), 255, np.uint8), np.zeros((0, 2), np.float32)
    mip_u8 = cv2.normalize(mip, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(mip_u8)
    k_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    top = cv2.morphologyEx(eq, cv2.MORPH_TOPHAT, k_big)
    _, bw = cv2.threshold(top, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(
        bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    )
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (51, 51)))
    thr_otsu, _ = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t1, t2 = max(0, int(0.66 * thr_otsu)), min(255, int(1.33 * thr_otsu))
    edges = cv2.Canny(eq, t1, t2)
    dt = distance_transform_edt((edges == 0).astype(np.uint8))
    gx = cv2.Sobel(eq, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(eq, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad /= float(grad.max()) or 1.0
    side_min = (1.0 - WELL_SIZE_TOL_FRAC) * side_est_px
    side_max = (1.0 + WELL_SIZE_TOL_FRAC) * side_est_px
    thr90 = int(np.percentile(mip_u8, 90))
    bright = mip_u8 >= thr90

    best = (-1e9, None, None)
    contours, _ = _cv_find_contours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours or []:
        if cv2.contourArea(cnt) < 2000:
            continue
        (cx, cy), (ww, hh), _ = cv2.minAreaRect(cnt)
        side0 = float(np.clip(max(ww, hh), side_min, side_max))
        m0 = make_rounded_mask((h, w), (cx, cy), side0, WELL_CORNER_RADIUS_FRAC)
        c0 = closed_contour(m0)
        s0 = score_well_mask(m0, c0, bright, dt, grad, w, h)
        if s0 > best[0]:
            best = (s0, m0, c0)
    if best[1] is None:
        ys, xs = np.nonzero(bright)
        cy = float(np.mean(ys)) if ys.size else h / 2.0
        cx = float(np.mean(xs)) if xs.size else w / 2.0
        side0 = float(np.clip(side_est_px, 0.40 * min(h, w), 0.95 * min(h, w)))
        m0 = make_rounded_mask((h, w), (cx, cy), side0, WELL_CORNER_RADIUS_FRAC)
        c0 = closed_contour(m0)
        best = (score_well_mask(m0, c0, bright, dt, grad, w, h), m0, c0)
    cont = best[2]
    cx, cy = float(np.mean(cont[:, 0])), float(np.mean(cont[:, 1]))
    side0 = max(np.ptp(cont[:, 0]), np.ptp(cont[:, 1]))
    rfrac = WELL_CORNER_RADIUS_FRAC
    for dx in (-3, -1, 0, 1, 3):
        for dy in (-3, -1, 0, 1, 3):
            for scale in (0.96, 0.98, 1.00, 1.02, 1.04):
                side = float(np.clip(side0 * scale, side_min, side_max))
                cx2 = float(np.clip(cx + dx, side / 2.0, w - side / 2.0))
                cy2 = float(np.clip(cy + dy, side / 2.0, h - side / 2.0))
                m = make_rounded_mask((h, w), (cx2, cy2), side, rfrac)
                c2 = closed_contour(m)
                s = score_well_mask(m, c2, bright, dt, grad, w, h)
                if s > best[0]:
                    best = (s, m, c2)
                    cx, cy = cx2, cy2
    return best[1], best[2]


def subpixel_argmax_z(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate subpixel Z argmax via parabolic fit around the peak.
Returns (z_hat, i0, idx, prom) arrays for each (H, W) pixel."""

    z, h, w = stack.shape
    idx_arg = np.argmax(stack, axis=0).astype(np.int32)

    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]

    # For Z<3 we cannot do a parabola fit; return integer argmax with NaN quality metrics.
    if z < 3:
        i0 = stack[idx_arg, rows, cols].astype(np.float32, copy=False)
        z_hat = idx_arg.astype(np.float32)
        prom = np.full((h, w), np.nan, np.float32)
        return z_hat, i0, idx_arg, prom

    idx = np.clip(idx_arg, 1, z - 2)
    i_m1 = stack[idx - 1, rows, cols].astype(np.float32, copy=False)
    i0 = stack[idx, rows, cols].astype(np.float32, copy=False)
    i_p1 = stack[idx + 1, rows, cols].astype(np.float32, copy=False)

    denom = i_m1 - 2.0 * i0 + i_p1
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = 0.5 * (i_m1 - i_p1) / denom
    delta = np.clip(np.nan_to_num(delta, nan=0.0), -0.5, 0.5).astype(np.float32)

    z_hat = idx.astype(np.float32) + delta
    prom = (i0 - 0.5 * (i_m1 + i_p1)).astype(np.float32, copy=False)
    return z_hat, i0, idx_arg, prom


def inpaint_nearest_within_mask(image: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """Fill NaNs/invalid pixels inside `roi` using nearest-neighbor propagation (2D). Pixels outside `roi` are left as NaN."""
    work = image.copy()
    roi_bool = np.asarray(roi, bool)
    invalid = ~np.isfinite(work)
    to_fill = roi_bool & invalid
    if not np.any(to_fill):
        return work
    valid_src = roi_bool & np.isfinite(work)
    if not np.any(valid_src):
        return work
    bad = ~valid_src
    _, idxs = distance_transform_edt(bad, return_indices=True)
    filled = work[tuple(idxs)]
    filled[~roi_bool] = np.nan
    return filled


def detect_stack_truncation(
    stack: np.ndarray,
    well_mask: np.ndarray,
    boundary_frac_thresh: float = 0.30,
    edge_hi_pct: float = 92.5,
) -> str | None:
    """Detect likely z-truncation (surface near stack boundary) and return mode/flags."""
    z, *_ = stack.shape
    if z < 3:
        return None
    in_well = well_mask.astype(bool)
    well_n = int(np.sum(in_well))
    if well_n == 0:
        return None

    idx = np.argmax(stack, axis=0)
    frac_top = float(np.sum((idx == z - 1) & in_well)) / max(1, well_n)
    frac_bot = float(np.sum((idx == 0) & in_well)) / max(1, well_n)

    mip = np.max(stack, axis=0)
    thr_ref = percentile_in_mask(mip, in_well, edge_hi_pct)

    thr_top = percentile_in_mask(stack[-1], in_well, edge_hi_pct)
    thr_bot = percentile_in_mask(stack[0], in_well, edge_hi_pct)

    rel = float(max(0.0, TRUNC_EDGE_REL_TO_MIP))
    top_has_signal = (
        np.isfinite(thr_top) and np.isfinite(thr_ref) and (thr_ref > 0.0) and (thr_top >= rel * thr_ref)
    )
    bot_has_signal = (
        np.isfinite(thr_bot) and np.isfinite(thr_ref) and (thr_ref > 0.0) and (thr_bot >= rel * thr_ref)
    )

    top_ok = (frac_top >= boundary_frac_thresh) and top_has_signal
    bot_ok = (frac_bot >= boundary_frac_thresh) and bot_has_signal

    if top_ok and (frac_top >= frac_bot or not bot_ok):
        return "top"
    if bot_ok:
        return "bottom"
    return None


def sigma_px_from_lambda_c(lambda_c_um: float, xy_um_per_px: float) -> float:
    """Convert spatial cutoff lambda_c [µm] into Gaussian sigma [px] given xy scale [µm/px]."""
    lam = float(lambda_c_um)
    xy = max(xy_um_per_px, 1e-6)
    sigma_um = (lam / np.pi) * np.sqrt(np.log(2.0) / 2.0)
    return max(0.0, sigma_um / xy)


def adaptive_bead_and_sfilter(
    xy_um_per_px: float,
    bead_diam_nom_um: float = BEAD_DIAMETER_NOMINAL_UM,
    k_lambda: float = S_FILTER_LAMBDA_MULT_FROM_DIAM,
    min_sigma_px: float = S_FILTER_MIN_SIGMA_PX,
) -> tuple[float, float, int, float]:
    """Derive effective bead diameter and S-filter cutoff from imaging scale and heuristics."""
    # Effective lateral resolution (reported diagnostic only; must NOT drive the
    # roughness bandwidth, otherwise binning/objective changes shift lambda_s -> B2).
    d_eff_um = max(float(bead_diam_nom_um), 2.0 * float(xy_um_per_px))
    r_eff_um = 0.5 * d_eff_um
    r_px = max(int(MIN_ENVELOPE_RADIUS_PX), math.ceil(r_eff_um / max(xy_um_per_px, 1e-9)))
    # S-filter cutoff: fixed physical length by default (pixel-size independent).
    # Only floored up for realizability if the pixel is too coarse to resolve it,
    # in which case a warning is logged because paired comparability is then broken.
    min_realizable = 2.355 * float(min_sigma_px) * float(xy_um_per_px)
    if isinstance(LAMBDA_S_OVERRIDE_UM, float) and not math.isnan(LAMBDA_S_OVERRIDE_UM):
        lambda_s_um = float(LAMBDA_S_OVERRIDE_UM)
    else:
        lambda_s_um = float(S_FILTER_LAMBDA_S_UM_DEFAULT)
    if lambda_s_um < min_realizable:
        logging.warning(
            "lambda_s=%.2f um < realizable %.2f um at xy=%.3f um/px; clamping (comparability may break).",
            lambda_s_um, min_realizable, float(xy_um_per_px),
        )
        lambda_s_um = min_realizable
    return r_eff_um, d_eff_um, r_px, lambda_s_um


def nan_gaussian_smooth2d(image: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian-smooth a 2D array while ignoring NaNs (normalize by smoothed validity mask)."""
    if sigma_px <= 0:
        return image.copy()
    nan = ~np.isfinite(image)
    work = image.copy()
    work[nan] = 0.0
    sm = gaussian_filter(work, sigma=sigma_px, mode="reflect")
    norm = gaussian_filter((~nan).astype(np.float32), sigma=sigma_px, mode="reflect")
    sm = sm / np.maximum(norm, 1e-6)
    sm[nan] = np.nan
    return sm


def sl_bandpass(
    height_um: np.ndarray,
    well_mask_bool: np.ndarray,
    lambda_s_um: float,
    lambda_c_um: float,
    xy_um_per_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute S–L band-pass residual (short-scale roughness) from a height map in microns."""
    sigma_s = sigma_px_from_lambda_c(lambda_s_um, xy_um_per_px)
    sigma_c = sigma_px_from_lambda_c(lambda_c_um, xy_um_per_px)
    in_well = well_mask_bool & np.isfinite(height_um)
    h = np.full_like(height_um, np.nan, np.float32)
    h[in_well] = height_um[in_well]
    h_s = nan_gaussian_smooth2d(h, sigma_s)
    h_L = nan_gaussian_smooth2d(h_s, sigma_c)
    return h_s, h_L, h_s - h_L


def inliers_mad(field: np.ndarray, roi_bool: np.ndarray, k: float = OUTLIER_MAD_K) -> np.ndarray:
    """Return boolean inlier mask using MAD thresholding within `mask` (robust to outliers)."""
    m = roi_bool & np.isfinite(field)
    if not np.any(m):
        return np.zeros_like(roi_bool, bool)
    vals = field[m]
    med = np.nanmedian(vals)
    mad = np.nanmedian(np.abs(vals - med))
    scale = 1.4826 * (mad if mad > 0 else (np.nanstd(vals) + 1e-9))
    return m & (np.abs(field - med) <= k * scale)


def fit_plane_and_r2_on_mask(
    Z: np.ndarray, mask: np.ndarray, xy_um_per_px: float
) -> tuple[tuple[float, float, float], float]:
    """Fit z = a*x + b*y + c on `mask` (finite pixels only) and return (a,b,c) and R^2.
    x and y are expressed in micrometers using `xy_um_per_px`, centered at the image center."""
    xy = float(max(xy_um_per_px, 1e-12))
    Zd = Z.astype(np.float64, copy=False)
    m = np.asarray(mask, bool) & np.isfinite(Zd)
    if not np.any(m):
        return (float("nan"), float("nan"), float("nan")), float("nan")

    h, w = Z.shape
    x0 = (w - 1) / 2.0
    y0 = (h - 1) / 2.0

    ys, xs = np.nonzero(m)
    X = (xs.astype(np.float64) - x0) * xy
    Y = (ys.astype(np.float64) - y0) * xy
    zvals = Zd[m]

    A = np.column_stack([X, Y, np.ones_like(X)])
    coeffs, *_ = np.linalg.lstsq(A, zvals, rcond=None)

    zfit = A @ coeffs
    ss_res = float(np.sum((zvals - zfit) ** 2))
    ss_tot = float(np.sum((zvals - float(np.mean(zvals))) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return (float(coeffs[0]), float(coeffs[1]), float(coeffs[2])), float(r2)


def tilt_correct_only_ab(Z: np.ndarray, a: float, b: float, xy_um_per_px: float) -> np.ndarray:
    """Remove a*x + b*y tilt component about the image center (keeps the mean height near the center)."""
    if not np.isfinite(a) or not np.isfinite(b):
        return Z.copy()
    xy = float(max(xy_um_per_px, 1e-12))
    h, w = Z.shape
    x = (np.arange(w, dtype=np.float64) - (w - 1) / 2.0) * xy
    y = (np.arange(h, dtype=np.float64) - (h - 1) / 2.0) * xy
    plane = (a * x[None, :]) + (b * y[:, None])
    return (Z.astype(np.float64, copy=False) - plane).astype(np.float32)


def comp_tilt_angle_deg(a: float, b: float) -> float:
    """Compute dominant tilt angle [deg] from plane parameters and return a scalar summary."""
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    return math.degrees(math.atan(math.sqrt(a * a + b * b)))


def set_pub_style() -> None:
    """Apply consistent matplotlib rcParams for figure generation."""
    _mpl.rcParams.update(
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


def verticalize_labels(labels) -> None:
    """Rotate/format axis tick labels to reduce overlap in dense plots."""
    for lbl in labels:
        lbl.set_rotation(90)
        lbl.set_verticalalignment("center")
        lbl.set_horizontalalignment("center")
        lbl.set_rotation_mode("anchor")


def save_mip_well_overlay(
    mip: np.ndarray, contour_px: np.ndarray, xy_um_per_px: float, out_png: str
) -> None:
    """Save a max-intensity projection with ROI mask/contour overlay for QC."""
    h, w = mip.shape
    extent = (0, w * round(xy_um_per_px, 2), 0, h * round(xy_um_per_px, 2))
    set_pub_style()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(mip, origin="lower", extent=extent)
    if contour_px is not None and contour_px.size >= 2:
        xs = contour_px[:, 0] * xy_um_per_px
        ys = contour_px[:, 1] * xy_um_per_px
        ax.plot(
            xs, ys, linewidth=2, color="cyan", linestyle="--", label=f"ROI ({MASK_SCALE_FRAC:.0%})"
        )
        ax.legend(loc="upper right")
    ax.set_xlabel("X [µm]")
    ax.set_ylabel("Y [µm]")
    ax.set_title(f"Max-Z with ROI (scaled {MASK_SCALE_FRAC:.0%})")
    verticalize_labels(ax.get_yticklabels())
    fig.tight_layout()
    fig.savefig(out_png, dpi=PNG_DPI)
    plt.close(fig)


def percentile_limits(vals: np.ndarray, lo_hi: tuple[float, float]) -> tuple[float, float]:
    """Compute robust display limits (vmin,vmax) from percentiles of finite values in a map."""
    lo_pct, hi_pct = lo_hi
    lo, hi = np.percentile(vals, [lo_pct, hi_pct])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    return float(lo), float(hi)


def save_height_map(
    out_png: str,
    height_um: np.ndarray,
    roi: np.ndarray,
    xy_um_per_px: float,
    vmin_um: float | None = None,
    vmax_um: float | None = None,
    title: str = "Height [µm]",
) -> None:
    """Save a height map visualization (µm) with mask-aware scaling and optional clipping."""
    h, w = height_um.shape
    extent = (0, w * round(xy_um_per_px, 2), 0, h * round(xy_um_per_px, 2))
    if (vmin_um is None or vmax_um is None) and (HEIGHT_COLORMAP_PERCENTILES is not None):
        vals = height_um[np.asarray(roi, bool) & np.isfinite(height_um)]
        if vals.size >= 10:
            vmin_um, vmax_um = percentile_limits(vals, HEIGHT_COLORMAP_PERCENTILES)
    set_pub_style()
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        np.ma.masked_where(~roi, height_um),
        origin="lower",
        extent=extent,
        vmin=vmin_um,
        vmax=vmax_um,
    )
    ax.set_xlabel("X [µm]")
    ax.set_ylabel("Y [µm]")
    ax.set_title(title)
    cbar = fig.colorbar(im)
    cbar.set_label("µm")
    cbar.locator = MaxNLocator(integer=True)
    cbar.update_ticks()
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(round(x))}"))
    fig.tight_layout()
    fig.savefig(out_png, dpi=PNG_DPI)
    plt.close(fig)


def round_symmetric_limit(vlim: float) -> float:
    """Round a symmetric visualization limit to a readable value while preserving scale.
    Intended for colorbar limits only; does not affect numeric metrics."""
    v = float(abs(vlim))
    if not np.isfinite(v) or v <= 0:
        return v
    exp10 = 10.0 ** math.floor(math.log10(v))
    mant = v / exp10
    if mant <= 1.0:
        mant_r = 1.0
    elif mant <= 2.0:
        mant_r = 2.0
    elif mant <= 5.0:
        mant_r = 5.0
    else:
        mant_r = 10.0
    return float(mant_r * exp10)


def save_residual_map(
    out_png: str,
    residual_um: np.ndarray,
    roi: np.ndarray,
    xy_um_per_px: float,
    title: str = "Residual [µm]",
    vlim_um: float | None = None,
) -> None:
    """Save a residual/roughness map visualization (µm) with consistent masking and labeling."""
    set_pub_style()
    h, w = residual_um.shape
    extent = (0, w * round(xy_um_per_px, 2), 0, h * round(xy_um_per_px, 2))
    roi_bool = np.asarray(roi, bool)
    vals = residual_um[roi_bool & np.isfinite(residual_um)]
    if vals.size == 0:
        return
    if vlim_um is None:
        lo, hi = np.percentile(vals, [1.0, 99.0])
        vlim = float(np.nanmax(np.abs([lo, hi])))
        if not np.isfinite(vlim) or vlim <= 0:
            vlim = float(np.nanmax(np.abs(vals)))
        vlim = max(vlim, 1e-3)
    else:
        vlim = float(abs(vlim_um))
    vlim_rounded = round_symmetric_limit(vlim)
    norm = TwoSlopeNorm(vmin=-vlim_rounded, vcenter=0.0, vmax=vlim_rounded)
    fig = plt.figure(figsize=(80 / 25.4, 80 / 25.4))
    ax = fig.add_subplot(111)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.02)
    im = ax.imshow(
        np.ma.masked_where(~roi_bool, residual_um),
        origin="lower",
        extent=extent,
        cmap="RdBu_r",  # diverging blue(-)-white(0)-red(+), matches residuals.svg
        norm=norm,
        aspect="equal",
    )
    ys, xs = np.nonzero(roi_bool)
    if xs.size and ys.size:
        pad = int(0.02 * max(xs.max() - xs.min(), ys.max() - ys.min()))
        ax.set_xlim(
            (max(xs.min() - pad, 0)) * xy_um_per_px, (min(xs.max() + pad, w - 1) + 1) * xy_um_per_px
        )
        ax.set_ylim(
            (max(ys.min() - pad, 0)) * xy_um_per_px, (min(ys.max() + pad, h - 1) + 1) * xy_um_per_px
        )
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    # Dynamic ticks from the physical extent of the displayed ROI.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=False))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=False))
    ax.minorticks_off()
    verticalize_labels(ax.get_yticklabels())
    ax.tick_params(axis="y", which="major", pad=6)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Residual [µm]")
    ticks = np.linspace(-vlim_rounded, vlim_rounded, 5)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:+.0f}" for t in ticks])
    verticalize_labels(cbar.ax.get_yticklabels())
    cbar.ax.tick_params(which="major", pad=8)
    fig.tight_layout(pad=0.02)
    fig.savefig(out_png, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


def save_float_tiff(path: str, arr: np.ndarray) -> None:
    """Save a 2D array as lossless float32 TIFF for quantitative downstream use."""
    tifffile.imwrite(path, np.asarray(arr, dtype=np.float32))


def gui_available() -> bool:
    """Return True if Tkinter dialogs can be used in the current runtime environment."""
    if tk is None or filedialog is None or messagebox is None or simpledialog is None:
        return False
    if sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            return False
    try:
        r = tk.Tk()
        r.withdraw()
        r.update_idletasks()
        r.destroy()
    except Exception:
        return False
    return True


def ask_root_folder() -> str | None:
    """Prompt for the analysis root folder via GUI (if available) or return None."""
    if not gui_available():
        return None

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(
            "Gel Surface Reconstruction",
            "Select a folder. All .tif/.tiff/.nd2 files will be processed.",
        )
        folder = filedialog.askdirectory(title="Select root folder")
        return folder or None
    except Exception:
        return None
    finally:
        try:
            if root is not None:
                root.destroy()
        except Exception:
            pass


def ask_folder_params(folder: str) -> dict[str, float] | None:
    """Prompt for per-folder voxel parameters (xy, dz, z0) and return numeric values."""
    if not gui_available():
        return None

    root = None
    try:
        root = tk.Tk()
        root.withdraw()

        exts = {".tif", ".tiff", ".nd2"}
        cand = next(
            (
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if os.path.splitext(f.lower())[1] in exts
            ),
            None,
        )

        xy0, dz0 = infer_voxel_sizes_from_file(cand) if cand else (None, None)
        message = f"Provide parameters for:\n{folder}\n\nUnits are micrometers (um)."
        if xy0 or dz0:
            message += (
                f"\n\nDetected: XY: {xy0 if xy0 else 'n/a'} um/px, dz: {dz0 if dz0 else 'n/a'} um"
            )
        messagebox.showinfo("Folder parameters", message)

        xy = simpledialog.askfloat(
            "Pixel size (XY)",
            "um per pixel (XY):",
            initialvalue=xy0,
            minvalue=1e-6,
            maxvalue=1e6,
        )
        if xy is None:
            return None

        dz = simpledialog.askfloat(
            "Z step",
            "Step between Z-slices [µm]:",
            initialvalue=dz0,
            minvalue=1e-6,
            maxvalue=1e6,
        )
        if dz is None:
            return None

        z0 = simpledialog.askfloat(
            "Z0 offset",
            "Reference offset Z0 [µm]:",
            initialvalue=0.0,
            minvalue=-1e6,
            maxvalue=1e6,
        )
        if z0 is None:
            return None

        return {"xy_um_per_px": float(xy), "z_step_um": float(dz), "z0_um": float(z0)}
    except Exception:
        return None
    finally:
        try:
            if root is not None:
                root.destroy()
        except Exception:
            pass


def list_image_files(root_dir: str) -> dict[str, list[str]]:
    """Enumerate stack files in a folder with supported extensions; return sorted paths."""
    exts = {".tif", ".tiff", ".nd2"}
    folder_map: dict[str, list[str]] = {}
    for current_root, _, files in os.walk(root_dir):
        sel = [
            os.path.join(current_root, f) for f in files if os.path.splitext(f.lower())[1] in exts
        ]
        if sel:
            folder_map[current_root] = sorted(sel)
    return folder_map


def compute_height_strict_truncated(
    stack: np.ndarray,
    well_mask: np.ndarray,
    z0_um: float,
    z_step_um: float,
    bright_hi_pct: float = 90.0,
    use_subpixel: bool = False,
) -> np.ndarray:
    """Estimate top-surface height map from bead signal with strict handling of truncation/QC."""
    z, h, w = stack.shape
    zmax_um = float(z0_um + (z - 1) * z_step_um)
    beyond = float(zmax_um + 0.5 * z_step_um)

    z_hat_subpx, _i0, idx_arg, _prom = subpixel_argmax_z(stack)
    z_hat = z_hat_subpx if (use_subpixel and z >= 3) else idx_arg.astype(np.float32)

    in_well = well_mask.astype(bool)
    interior = (idx_arg > 0) & (idx_arg < z - 1)

    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    i_arg = stack[idx_arg, rows, cols].astype(np.float32, copy=False)

    vals = i_arg[in_well & interior]
    vals = vals[np.isfinite(vals)]

    height = np.full((h, w), np.nan, np.float32)
    if vals.size == 0:
        height[in_well] = beyond
        return height

    thr = float(np.percentile(vals, bright_hi_pct))
    valid = in_well & interior & (i_arg >= thr)

    height[valid] = z0_um + z_hat[valid] * z_step_um
    height[in_well & (~valid)] = beyond
    return height


def core_roi_from_well_mask(
    well_mask_u8: np.ndarray, well_mask_bool: np.ndarray, margin_px: int
) -> np.ndarray:
    """Compute a conservative core ROI mask by eroding/scaling the well mask to avoid edges."""
    if np.all(well_mask_u8 > 0):
        return well_mask_bool.copy()
    se_core = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin_px + 1,) * 2)
    return cv2.erode(well_mask_u8, se_core).astype(bool)


def process_stack(
    path: str,
    xy_um_per_px: float,
    z_step_um: float,
    z0_um: float,
    out_dir: str,
    *,
    analysis_sha256: str,
    analysis_timestamp_utc: str,
) -> dict[str, Any]:
    """Analyze one stack and write per-stack figures; return a summary-row dict for CSV.
    Computes a bead-layer top-surface height map, removes planar tilt, and reports dispersion metrics from the L-filtered surface and the S–L band-pass residual
    """
    base = os.path.splitext(os.path.basename(path))[0]
    out_prefix = os.path.join(out_dir, base)
    analysis_info = (
        f"{ANALYSIS_NAME}|{ANALYSIS_VERSION}|{analysis_sha256[:12]}|{analysis_timestamp_utc}"
    )

    stack = read_stack(path)
    z, h, w = stack.shape
    if z < 3:
        raise ValueError(
            f"Stack '{os.path.basename(path)}' has z={z} < 3 slices; need >=3. "
            "If this is a multi-page TIFF, check that its axis metadata names the "
            "page axis (Z, I or Q) rather than a channel or time axis."
        )

    well_mask_u8, well_contour_px = segment_well_from_stack(stack, xy_um_per_px)
    file_name = os.path.basename(path)
    base_row = make_base_row(
        file_name=file_name,
        z_slices=z,
        xy_um_per_px=xy_um_per_px,
        z_step_um=z_step_um,
        z0_um=z0_um,
        analysis_info=analysis_info,
    )

    is_full_field = bool((well_contour_px.size == 0) and np.all(well_mask_u8 > 0))
    if is_full_field:
        logging.info("ROI mode: full-field (FOV within well); skipping MASK_SCALE_FRAC scaling.")
    elif np.isfinite(MASK_SCALE_FRAC) and (MASK_SCALE_FRAC != 1.0):
        well_mask_u8, well_contour_px = scale_mask_and_contour(
            well_mask_u8, well_contour_px, float(MASK_SCALE_FRAC)
        )

    effective_scale = 1.0
    if (not is_full_field) and np.isfinite(MASK_SCALE_FRAC):
        effective_scale = float(MASK_SCALE_FRAC)
    base_row["MaskScaleFrac"] = round_for_csv(effective_scale)

    well_mask = well_mask_u8.astype(bool)

    mip = np.max(stack, axis=0).astype(np.float32)
    save_mip_well_overlay(mip, well_contour_px, xy_um_per_px, out_prefix + "_mip_well_outline.png")

    z_hat, i_0, idx_arg, prom = subpixel_argmax_z(stack)
    boundary = (idx_arg == 0) | (idx_arg == z - 1)

    with np.errstate(all="ignore"):
        prom_well = prom[well_mask]
        thr_prom = (
            np.percentile(prom_well[np.isfinite(prom_well)], 5)
            if np.isfinite(prom_well).any()
            else 0.0
        )

    low_prom = prom < thr_prom
    bad = boundary | low_prom

    trunc_mode = detect_stack_truncation(
        stack, well_mask, boundary_frac_thresh=0.30, edge_hi_pct=92.5
    )

    if trunc_mode is not None:
        height_raw = compute_height_strict_truncated(
            stack, well_mask, z0_um, z_step_um, bright_hi_pct=90.0, use_subpixel=False
        )
        zmax_um = float(z0_um + (z - 1) * z_step_um)
        valid_mask = well_mask & np.isfinite(height_raw) & (height_raw <= zmax_um)
        well_px_n = int(np.sum(well_mask))
        valid_px_n = int(np.sum(valid_mask))
        valid_frac = float(valid_px_n) / well_px_n if well_px_n else float("nan")
        roi_core = core_roi_from_well_mask(well_mask_u8, well_mask, CORE_MARGIN_PX)
        core_px_n = int(np.sum(roi_core))
        core_valid_px_n = int(np.sum(valid_mask & roi_core))
        core_valid_frac = float(core_valid_px_n) / core_px_n if core_px_n else float("nan")
        height_vis = np.where(valid_mask, height_raw, np.nan).astype(np.float32)
        vis_height_raw = inpaint_nearest_within_mask(height_vis, well_mask)
        if SAVE_PNGS:
            save_height_map(
                out_prefix + "_height_raw.png",
                vis_height_raw,
                well_mask,
                xy_um_per_px,
                title="Top-surface Z [µm] (raw; truncation)",
            )
        return make_row(
            base_row,
            TopZ_Median_um=round_for_csv(float("nan")),
            Sq_SL_um=round_for_csv(float("nan")),
            TopZ_LFilteredMean_um=round_for_csv(float("nan")),
            TopZ_LFilteredStd_um=round_for_csv(float("nan")),
            BandpassStd_um=round_for_csv(float("nan")),
            TiltApplied=False,
            TiltAngle_deg=round_for_csv(float("nan")),
            Plane_a_um_per_mm=round_for_csv(float("nan")),
            Plane_b_um_per_mm=round_for_csv(float("nan")),
            Plane_c_um=round_for_csv(float("nan")),
            Plane_R2=round_for_csv(float("nan")),
            Valid_frac=round_for_csv(valid_frac),
            Core_valid_frac=round_for_csv(core_valid_frac),
            Well_px_n=well_px_n,
            Valid_px_n=valid_px_n,
            Core_px_n=core_px_n,
            Core_valid_px_n=core_valid_px_n,
            TruncMode=str(trunc_mode),
            LambdaS_um=round_for_csv(float("nan")),
            BeadDiamEff_um=round_for_csv(float("nan")),
        )

    height_raw = np.full_like(z_hat, np.nan, np.float32)
    ok = well_mask & (~bad)
    height_raw[ok] = z0_um + z_hat[ok] * z_step_um
    well_px_n = int(np.sum(well_mask))
    valid_px_n = int(np.sum(ok))
    valid_frac = float(valid_px_n) / well_px_n if well_px_n else float("nan")

    if SAVE_PNGS and SAVE_HEIGHT_PNG:
        vis_height_abs = inpaint_nearest_within_mask(height_raw, well_mask)
        vmin_abs = float(z0_um)
        vmax_abs = float(z0_um + (z - 1) * z_step_um)
        save_height_map(
            out_prefix + "_height.png",
            vis_height_abs,
            well_mask,
            xy_um_per_px,
            vmin_um=vmin_abs,
            vmax_um=vmax_abs,
            title="Top-surface Z [µm]",
        )

    roi_core = core_roi_from_well_mask(well_mask_u8, well_mask, CORE_MARGIN_PX)
    core_px_n = int(np.sum(roi_core))
    core_valid_px_n = int(np.sum(roi_core & np.isfinite(height_raw)))
    core_valid_frac = float(core_valid_px_n) / core_px_n if core_px_n else float("nan")
    existing_core = roi_core & np.isfinite(height_raw)
    if not existing_core.any():
        existing_core = well_mask & np.isfinite(height_raw)

    _, d_eff_um, _, lambda_s_um = adaptive_bead_and_sfilter(
        xy_um_per_px,
        bead_diam_nom_um=BEAD_DIAMETER_NOMINAL_UM,
    )
    (a_fit, b_fit, c_fit), r2_plane = fit_plane_and_r2_on_mask(
        height_raw, existing_core, xy_um_per_px
    )
    tilt_angle_deg = comp_tilt_angle_deg(a_fit, b_fit)
    height_tiltcorr = tilt_correct_only_ab(height_raw, a_fit, b_fit, xy_um_per_px)
    tilt_applied = bool(np.isfinite(a_fit) and np.isfinite(b_fit))

    sigma_px = sigma_px_from_lambda_c(LFILTER_LAMBDA_C_UM, xy_um_per_px)
    existing_well = well_mask & np.isfinite(height_tiltcorr)
    height_for_stats = np.full_like(height_tiltcorr, np.nan, np.float32)
    height_for_stats[existing_well] = height_tiltcorr[existing_well]
    height_L = nan_gaussian_smooth2d(height_for_stats, sigma_px=sigma_px)
    finite_well_L = well_mask & np.isfinite(height_L)
    finite_core_L = existing_core & np.isfinite(height_L)

    if finite_core_L.any():
        lo, hi = np.percentile(height_L[finite_core_L], [INLIER_LO_PCT, INLIER_HI_PCT])
        inliers = (height_L >= lo) & (height_L <= hi) & finite_well_L & existing_core
    else:
        inliers = np.zeros_like(well_mask, bool)

    height_mean_um = float(np.nanmean(height_L[inliers])) if inliers.any() else float("nan")
    height_std_um = float(np.nanstd(height_L[inliers])) if inliers.any() else float("nan")
    mu_surface = float(height_mean_um) if np.isfinite(height_mean_um) else 0.0
    residual_surface = height_L - mu_surface

    h_s, _h_L_s, r_bp = sl_bandpass(
        height_tiltcorr, well_mask, lambda_s_um, LFILTER_LAMBDA_C_UM, xy_um_per_px
    )
    inliers_bp = inliers_mad(r_bp, (roi_core & well_mask), k=OUTLIER_MAD_K)

    if inliers_bp.any():
        bp_vals = r_bp[inliers_bp]
    else:
        bp_vals = r_bp[(roi_core & well_mask) & np.isfinite(r_bp)]

    std_unf_um = float(np.nanstd(bp_vals)) if bp_vals.size else float("nan")

    # --- Robust GT-comparable apparent height: median over core valid area.
    # Uses the tilt-corrected surface before L-filtering to avoid edge smoothing bias,
    # restricted to the core ROI; median is symmetric and outlier-robust.
    core_valid = existing_core & np.isfinite(height_tiltcorr)
    if core_valid.any():
        topz_median_um = float(np.nanmedian(height_tiltcorr[core_valid]))
    else:
        topz_median_um = float("nan")

    # --- ISO 25178-2-aligned areal Sq on the S-L (band-pass) surface: RMS over the
    # valid evaluation area with no outlier rejection (difference from BandpassStd_um).
    sq_area = (roi_core & well_mask) & np.isfinite(r_bp)
    if sq_area.any():
        rv = r_bp[sq_area].astype(np.float64)
        sq_sl_um = float(np.sqrt(np.mean((rv - rv.mean()) ** 2)))
    else:
        sq_sl_um = float("nan")

    # --- Per-pixel confidence map (normalized peak prominence in [0,1]) + validity mask.
    conf = np.full_like(prom, np.nan, np.float32)
    pw = prom[well_mask & np.isfinite(prom)]
    if pw.size:
        p_hi = float(np.percentile(pw, 99.0)) or 1.0
        conf_vals = np.clip(prom / (p_hi if p_hi > 0 else 1.0), 0.0, 1.0)
        conf[well_mask] = conf_vals[well_mask]
    validity = (well_mask & np.isfinite(height_raw)).astype(np.float32)

    # --- Lossless numeric maps for downstream metrology. NaN outside ROI.
    if SAVE_PNGS:
        save_float_tiff(out_prefix + "_height_abs_um.tif", np.where(well_mask, height_raw, np.nan))
        save_float_tiff(out_prefix + "_height_tiltcorr_um.tif", np.where(well_mask, height_tiltcorr, np.nan))
        save_float_tiff(out_prefix + "_residual_surface_um.tif", np.where(well_mask, residual_surface, np.nan))
        save_float_tiff(out_prefix + "_residual_bandpass_um.tif", np.where(well_mask, r_bp, np.nan))
        save_float_tiff(out_prefix + "_confidence.tif", conf)
        save_float_tiff(out_prefix + "_validity.tif", validity)

    if SAVE_PNGS:
        vis_resid_surface = inpaint_nearest_within_mask(residual_surface, well_mask)
        save_residual_map(
            out_prefix + "_residual_surface.png",
            vis_resid_surface,
            well_mask,
            xy_um_per_px,
            title="Residual (tilt-removed, L-filtered)",
        )
        save_residual_map(
            out_prefix + "_residual_surface_pm100um.png",
            vis_resid_surface,
            well_mask,
            xy_um_per_px,
            title="Residual (tilt-removed, L-filtered) (±100 µm)",
            vlim_um=100.0,
        )
        if SAVE_BANDPASS_RESIDUAL_PNG:
            vis_r_bp = inpaint_nearest_within_mask(r_bp, well_mask)
            save_residual_map(
                out_prefix + "_residual_bandpass.png",
                vis_r_bp,
                well_mask,
                xy_um_per_px,
                title="Residual (S-L band-pass)",
            )

    return make_row(
        base_row,
        TopZ_Median_um=round_for_csv(topz_median_um),
        Sq_SL_um=round_for_csv(sq_sl_um),
        TopZ_LFilteredMean_um=round_for_csv(height_mean_um),
        TopZ_LFilteredStd_um=round_for_csv(height_std_um),
        BandpassStd_um=round_for_csv(std_unf_um),
        TiltApplied=tilt_applied,
        TiltAngle_deg=round_for_csv(tilt_angle_deg),
        Plane_a_um_per_mm=round_for_csv(a_fit * 1000),
        Plane_b_um_per_mm=round_for_csv(b_fit * 1000),
        Plane_c_um=round_for_csv(c_fit),
        Plane_R2=round_for_csv(r2_plane),
        Valid_frac=round_for_csv(valid_frac),
        Core_valid_frac=round_for_csv(core_valid_frac),
        Well_px_n=well_px_n,
        Valid_px_n=valid_px_n,
        Core_px_n=core_px_n,
        Core_valid_px_n=core_valid_px_n,
        TruncMode="none",
        LambdaS_um=round_for_csv(lambda_s_um),
        BeadDiamEff_um=round_for_csv(d_eff_um),
    )


def init_logging() -> None:
    """Configure root logger formatting and default verbosity for CLI and batch runs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def write_folder_outputs(
    out_dir: str,
    rows: list[dict[str, Any]],
    *,
    xy_um_per_px: float,
    z_step_um: float,
    z0_um: float,
    analysis_sha256: str,
) -> None:
    """Write per-folder outputs (CSV + optional figures) from collected per-stack rows."""
    df = pd.DataFrame(rows).reindex(columns=CSV_COLUMNS)
    csv_path = os.path.join(out_dir, "summary_metrics.csv")
    df.to_csv(csv_path, index=False, float_format="%.2f")

    meta = build_run_metadata(
        analysis_name=ANALYSIS_NAME,
        analysis_version=ANALYSIS_VERSION,
        analysis_sha256=analysis_sha256,
        params={
            "xy_um_per_px": float(xy_um_per_px),
            "z_step_um": float(z_step_um),
            "z0_um": float(z0_um),

            # ROI / segmentation
            "well_edge_mm": float(WELL_EDGE_MM),
            "well_size_tol_frac": float(WELL_SIZE_TOL_FRAC),
            "well_corner_radius_frac": float(WELL_CORNER_RADIUS_FRAC),
            "mask_scale_frac": float(MASK_SCALE_FRAC),
            "core_margin_px": int(CORE_MARGIN_PX),

            # Height map + filtering
            "lambda_c_um": float(LFILTER_LAMBDA_C_UM),
            "inlier_lo_pct": float(INLIER_LO_PCT),
            "inlier_hi_pct": float(INLIER_HI_PCT),

            # Band-pass / robustness
            "outlier_mad_k": float(OUTLIER_MAD_K),
            "bead_diameter_nominal_um": float(BEAD_DIAMETER_NOMINAL_UM),
            "s_filter_lambda_mult_from_diam": float(S_FILTER_LAMBDA_MULT_FROM_DIAM),
            "s_filter_min_sigma_px": float(S_FILTER_MIN_SIGMA_PX),
            "min_envelope_radius_px": int(MIN_ENVELOPE_RADIUS_PX),
            "lambda_s_override_um": float(LAMBDA_S_OVERRIDE_UM),

            # Truncation + multi-dim handling
            "trunc_edge_rel_to_mip": float(TRUNC_EDGE_REL_TO_MIP),
            "nd2_select_index": dict(ND2_SELECT_INDEX),
            "tiff_nonspatial_index": int(TIFF_NONSPATIAL_INDEX),
        },
    )

    meta_path = os.path.join(out_dir, "run_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(_json_sanitize(meta), f, indent=2, allow_nan=False)

    logging.info("Wrote: %s", meta_path)
    logging.info("Wrote: %s", csv_path)


def analyze_folder(
    folder: str,
    files: list[str],
    params: dict[str, float],
    *,
    analysis_sha256: str,
    analysis_timestamp_utc: str,
) -> None:
    """Analyze all stacks in a folder using shared voxel parameters; return summary rows."""
    out_dir = os.path.join(folder, "gel_surface_out")
    os.makedirs(out_dir, exist_ok=True)

    rows: list[dict[str, Any]] = []
    logging.info("=== Processing folder: %s ===", folder)
    logging.info("Found %d file(s). Output -> %s", len(files), out_dir)
    logging.info("ROI scaling: %.0f%%", MASK_SCALE_FRAC * 100)

    for i, path in enumerate(files, 1):
        logging.info("[%d/%d] %s", i, len(files), os.path.basename(path))
        try:
            rows.append(
                process_stack(
                    path,
                    params["xy_um_per_px"],
                    params["z_step_um"],
                    params["z0_um"],
                    out_dir,
                    analysis_sha256=analysis_sha256,
                    analysis_timestamp_utc=analysis_timestamp_utc,
                )
            )
        except Exception as exc:
            logging.error("Error processing %s: %s", path, exc)
            continue

    if not rows:
        logging.warning("No successful analyses in folder: %s", folder)
        return

    write_folder_outputs(
        out_dir,
        rows,
        xy_um_per_px=float(params["xy_um_per_px"]),
        z_step_um=float(params["z_step_um"]),
        z0_um=float(params["z0_um"]),
        analysis_sha256=analysis_sha256,
    )


def infer_params_for_folder(folder: str, *, z0_um: float) -> dict[str, float]:
    """Infer folder-level voxel parameters from metadata and/or prompts; return xy,dz,z0 values."""
    exts = {".tif", ".tiff", ".nd2"}
    cand = next(
        (
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.splitext(f.lower())[1] in exts
        ),
        None,
    )
    if not cand:
        raise RuntimeError(f"No stack found in {folder}")

    xy0, dz0 = infer_voxel_sizes_from_file(cand)
    if not xy0 or not dz0:
        raise RuntimeError(f"Missing --xy/--dz and could not infer voxel sizes from {cand}")

    return {"xy_um_per_px": float(xy0), "z_step_um": float(dz0), "z0_um": float(z0_um)}


def main() -> int:
    """CLI entry point: parse args, collect folders, run analyses, and write outputs."""
    global CORE_MARGIN_PX, INLIER_LO_PCT, INLIER_HI_PCT, LAMBDA_S_OVERRIDE_UM
    global LFILTER_LAMBDA_C_UM, BEAD_DIAMETER_NOMINAL_UM, MASK_SCALE_FRAC

    init_logging()
    analysis_sha256 = sha256_file(__file__)
    analysis_timestamp_utc = utc_now_iso()

    parser = argparse.ArgumentParser(
        description="Gel surface uniformity analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bead-um", type=float, default=BEAD_DIAMETER_NOMINAL_UM, help="Nominal bead diameter [µm]"
    )
    parser.add_argument(
        "--lambda-c-um",
        type=float,
        default=LFILTER_LAMBDA_C_UM,
        help="L-filter cutoff lambda_c [µm]",
    )
    parser.add_argument("--lambda-s-um", type=float, help="S-filter cutoff lambda_s [µm] override")
    parser.add_argument(
        "--mask-scale", type=float, default=MASK_SCALE_FRAC, help="ROI scaling factor (0.0-1.0)"
    )
    parser.add_argument(
        "--core-margin-px", type=int, default=CORE_MARGIN_PX, help="Core mask margin [px]"
    )
    parser.add_argument(
        "--inlier-lo", type=float, default=INLIER_LO_PCT, help="Low percentile for inliers"
    )
    parser.add_argument(
        "--inlier-hi", type=float, default=INLIER_HI_PCT, help="High percentile for inliers"
    )
    parser.add_argument("--log-file", type=str, help="Log file path")
    parser.add_argument("--root", type=str, help="Root folder with subfolders of stacks")
    parser.add_argument("--xy", type=float, help="XY um/px to apply to all folders")
    parser.add_argument("--dz", type=float, help="Z step um to apply to all folders")
    parser.add_argument(
        "--z0", type=float, default=0.0, help="Z0 um (offset) to apply to all folders"
    )
    parser.add_argument("--no-gui", action="store_true", help="Disable GUI prompts entirely")
    parser.add_argument("--headless", action="store_true", help="Alias of --no-gui")
    parser.add_argument(
        "--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    ns, _extra = parser.parse_known_args()

    CORE_MARGIN_PX = int(ns.core_margin_px)
    INLIER_LO_PCT = float(ns.inlier_lo)
    INLIER_HI_PCT = float(ns.inlier_hi)
    if ns.headless:
        ns.no_gui = True
    MASK_SCALE_FRAC = float(ns.mask_scale)

    # Auto-fallback to headless behavior if a GUI cannot be opened (prevents TclError crashes).
    if (not ns.no_gui) and (not gui_available()):
        logging.warning(
            "GUI is unavailable (headless environment or Tk not functional). Falling back to --no-gui behavior."
        )
        ns.no_gui = True

    logging.getLogger().setLevel(getattr(logging, ns.log_level.upper(), logging.INFO))

    if ns.log_file:
        fh = logging.FileHandler(ns.log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        fh.setLevel(logging.getLogger().level)
        logging.getLogger().addHandler(fh)

    if ns.lambda_s_um is not None:
        LAMBDA_S_OVERRIDE_UM = float(ns.lambda_s_um)
    if ns.lambda_c_um is not None:
        LFILTER_LAMBDA_C_UM = float(ns.lambda_c_um)
    if ns.bead_um is not None:
        BEAD_DIAMETER_NOMINAL_UM = float(ns.bead_um)

    args = sys.argv[1:]
    root = ns.root or (args[0] if args and os.path.isdir(args[0]) else None)

    if not root and not ns.no_gui:
        root = ask_root_folder()
    if not root:
        logging.error("No folder selected. Exiting.")
        return 1

    folder_map = list_image_files(root)
    if not folder_map:
        logging.error("No image files found under: %s", root)
        return 2

    global_params: dict[str, float] | None = None
    if ns.xy is not None and ns.dz is not None:
        global_params = {
            "xy_um_per_px": float(ns.xy),
            "z_step_um": float(ns.dz),
            "z0_um": float(ns.z0),
        }

    if ns.no_gui:
        inferred_params: dict[str, dict[str, float]] = {}
        if global_params is not None:
            inferred_params = {folder: global_params for folder in folder_map}
            logging.info("Headless mode: using global parameters for all folders.")
        else:
            logging.info("Headless mode: inferring voxel sizes for each folder.")
            for folder in folder_map:
                try:
                    inferred_params[folder] = infer_params_for_folder(folder, z0_um=float(ns.z0))
                except Exception as exc:
                    logging.error("%s", exc)
                    return 3

        for folder, files in folder_map.items():
            analyze_folder(
                folder,
                files,
                inferred_params[folder],
                analysis_sha256=analysis_sha256,
                analysis_timestamp_utc=analysis_timestamp_utc,
            )
        logging.info("Done.")
        return 0

    # GUI mode (prompt per folder)
    for folder, files in folder_map.items():
        params = ask_folder_params(folder)
        if params is None:
            logging.warning("Skipping folder (parameters not provided): %s", folder)
            continue
        analyze_folder(
            folder,
            files,
            params,
            analysis_sha256=analysis_sha256,
            analysis_timestamp_utc=analysis_timestamp_utc,
        )
    logging.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
