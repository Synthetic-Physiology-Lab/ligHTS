#!/usr/bin/env python3
"""
HOLOreader (registered) — grooves vertical + subpixel translation registration before cropping.

Registration (simple & robust, widely used):
- **Phase correlation** (FFT-based) with a 2-D Hanning window (cv2.phaseCorrelate) to get subpixel dx,dy.
- Work on rotated frames (grooves vertical). Use intersection of valid masks and Hanning window to
  avoid NaN borders and reduce edge effects. Translate with cv2.warpAffine.
- Minimal & deterministic: no ECC refinement to keep the update small unless requested.

Other steps unchanged: phase→height, pitch estimate, largest-square crop, fixed 8-bit window.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageFile, TiffImagePlugin
from tqdm import tqdm
import cv2
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from collections import defaultdict

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ----------------------------- Constants ------------------------------
DELTA_N: float = 0.00278 # updated following calibration
WAVELENGTH_UM: float = 0.635
SCALE_UM_PER_RAD: float = WAVELENGTH_UM / (2.0 * math.pi * DELTA_N)
WINDOW_MIN_UM: float = -25.0
WINDOW_MAX_UM: float = +50.0
NEAREST_SET: Tuple[int, int, int] = (40, 60, 80)
DEFAULT_PX_UM: float = 0.54


# ------------------------------ Utilities -----------------------------
def sha256_file(path: str) -> str:
    """Compute SHA-256 hex digest for a file at `path` (streamed, constant memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


__version__ = "1.0.0"
try:
    TOOL_SHA256 = sha256_file(__file__)
except Exception:  # pragma: no cover - provenance is best-effort
    TOOL_SHA256 = "unknown"


def setup_logging() -> None:
    """ Call logger """
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def ask_folder_and_scale() -> Tuple[str, float]:
    """ User selection of folder and calibration parameters for conversion """
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select folder with TIFF frames")
    if not folder:
        messagebox.showerror("Error", "No folder selected.")
        sys.exit(2)
    px_um = simpledialog.askfloat(
        "Pixel size (µm/pixel)",
        "Enter XY pixel calibration (µm per pixel):",
        minvalue=0.0001,
        initialvalue=DEFAULT_PX_UM,
    )
    if px_um is None:
        messagebox.showerror("Error", "No pixel calibration provided.")
        sys.exit(3)
    return folder, float(px_um)


def parse_filename(fname: str) -> Tuple[str, int]:
    """ Parse filename to group frames """
    base = os.path.basename(fname)
    name, _ = os.path.splitext(base)
    m = re.search(r"-\s*(\d+)\s+(\d+)", name)
    if not m:
        raise ValueError(f"Cannot parse FOV/ordinal from '{fname}'")
    return m.group(1), int(m.group(2))


def _extract_phase_text(tags: "TiffImagePlugin.ImageFileDirectory_v2") -> Optional[str]:
    """ Look for location of information on min and max phase reported values """
    preferred = [270, 40092, 40094, 40095]
    keys = list(dict.fromkeys([k for k in preferred if k in tags] + list(tags.keys())))
    for tid in keys:
        try:
            val = tags.get(tid)
            if isinstance(val, (tuple, list)) and len(val) == 1:
                val = val[0]
            if isinstance(val, (list, tuple)) and val and isinstance(val[0], int):
                try:
                    val = bytes(val)
                except Exception:
                    pass
            if isinstance(val, (bytes, bytearray)):
                try:
                    s = val.decode("utf-16le", errors="ignore")
                except Exception:
                    s = val.decode("latin-1", errors="ignore")
            else:
                s = str(val)
            s = s.replace("\\x00", "").replace("\x00", "").strip()
            if "Min" in s and "Max" in s:
                return s
        except Exception:
            continue
    return None


def read_uint16(path: str) -> Tuple[NDArray[np.uint16], Optional[str]]:
    """ Image reader """
    with Image.open(path) as im:
        if im.mode not in ("I;16", "I;16B", "I;16L"):
            im = im.convert("I;16")
        try:
            desc = _extract_phase_text(im.tag_v2)
        except Exception:
            desc = None
        arr = np.array(im, dtype=np.uint16, copy=True)
    return arr, (str(desc) if desc is not None else None)


def parse_phase_minmax(desc: Optional[str]) -> Tuple[float, float]:
    """ Parse information on min and max phase reported values """
    if not desc:
        raise ValueError("Missing ImageDescription for phase min/max.")
    text = str(desc)
    m = re.search(
        r"Min\s*\(\s*0\s*\)\s*[:=]\s*([+-]?(?:\d+(?:[.,]\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*.*?"
        r"Max\s*\(\s*65535\s*\)\s*[:=]\s*([+-]?(?:\d+(?:[.,]\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not m:
        raise ValueError("Could not parse 'Min (0)=... Max (65535)=...' from metadata.")
    a = float(m.group(1).replace(",", "."))
    b = float(m.group(2).replace(",", "."))
    if not np.isfinite(a) or not np.isfinite(b) or a == b:
        raise ValueError("Invalid per‑image phase min/max.")
    return a, b


def u16_to_height_um(
    u16: NDArray[np.uint16], min_rad: float, max_rad: float
) -> NDArray[np.float32]:
    """ phase to micron conversion """
    u = u16.astype(np.float32)
    phi = min_rad + (u / 65535.0) * (max_rad - min_rad)
    return phi * float(SCALE_UM_PER_RAD)


def rescale_height_to_u8(
    h_um: NDArray[np.float32], lo: float = WINDOW_MIN_UM, hi: float = WINDOW_MAX_UM
) -> NDArray[np.uint8]:
    """ rescaling micron height map to 8-bit """
    h = np.where(np.isfinite(h_um), h_um, lo)
    h = np.clip(h, lo, hi, out=np.empty_like(h))
    norm = (h - lo) / (hi - lo)
    return (norm * 255.0 + 0.5).astype(np.uint8)


# --------------------------- FFT angle & pitch -------------------------
def _fft_power(img: NDArray[np.float32]) -> NDArray[np.float32]:
    """ calculate fft """
    h, w = img.shape
    wy = np.hanning(h)[:, None]
    wx = np.hanning(w)[None, :]
    win = (wy * wx).astype(np.float32)
    spec = np.fft.fftshift(np.fft.fft2(np.nan_to_num(img * win, nan=0.0)))
    return np.abs(spec).astype(np.float32)


def estimate_grating_normal_deg(img: NDArray[np.float32]) -> float:
    """ find angle for groove alignment """
    P = _fft_power(img)
    h, w = P.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    rr = np.hypot(yy - cy, xx - cx)
    P = P.copy()
    P[rr < 5.0] = 0.0
    angles = np.linspace(0.0, 180.0, 720, endpoint=False, dtype=np.float32)
    r_max = int(0.48 * min(h, w))
    r_vals = np.arange(6, r_max, dtype=np.float32)
    best_ang = 0.0
    best_score = -1.0
    for a in angles:
        th = np.deg2rad(float(a))
        xs = (cx + r_vals * np.cos(th)).astype(int)
        ys = (cy + r_vals * np.sin(th)).astype(int)
        m = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        s = float(P[ys[m], xs[m]].sum())
        if s > best_score:
            best_score = s
            best_ang = float(a)
    return best_ang


# ---------------------------- Rotation (cv2) ---------------------------
def rotate_bound_nan(
    img: NDArray[np.float32], angle_deg: float
) -> Tuple[NDArray[np.float32], NDArray[np.uint8]]:
    """ Applies rotation for alignment """
    h, w = img.shape[:2]
    c = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(c, -angle_deg, 1.0)  # use requested angle
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(round((h * sin) + (w * cos)))
    new_h = int(round((h * cos) + (w * sin)))
    M[0, 2] += (new_w / 2.0) - c[0]
    M[1, 2] += (new_h / 2.0) - c[1]
    dst = cv2.warpAffine(
        img.astype(np.float32),
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan"),
    )
    ones = np.ones((h, w), dtype=np.uint8) * 255
    mask = cv2.warpAffine(
        ones,
        M,
        (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return dst, mask


# --------------------------- Registration ------------------------------
def _hanning2d(h: int, w: int) -> NDArray[np.float64]:
    """ creates hanning 2d window """
    return np.outer(np.hanning(h), np.hanning(w)).astype(np.float64)


def register_by_phasecorr(
    mov: NDArray[np.float32],
    ref: NDArray[np.float32],
    mov_mask: NDArray[np.uint8],
    ref_mask: NDArray[np.uint8],
) -> Tuple[NDArray[np.float32], NDArray[np.uint8], Tuple[float, float], float]:
    """ Frame registration """
    assert mov.shape == ref.shape
    h, w = mov.shape
    common = ((mov_mask > 0) & (ref_mask > 0)).astype(np.float64)
    if common.sum() < 64:
        return mov, mov_mask, (0.0, 0.0), 0.0
    win = _hanning2d(h, w) * common
    ref64 = np.nan_to_num(ref, nan=0.0).astype(np.float64) * win
    mov64 = np.nan_to_num(mov, nan=0.0).astype(np.float64) * win
    (dx, dy), resp = cv2.phaseCorrelate(ref64, mov64)
    if not np.isfinite(dx) or not np.isfinite(dy):
        dx = dy = 0.0
        resp = 0.0
    lim = 0.25 * min(h, w)
    if abs(dx) > lim or abs(dy) > lim or resp < 0.01:
        dx = dy = 0.0
    M = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32)
    aligned = cv2.warpAffine(
        mov,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan"),
    )
    aligned_mask = cv2.warpAffine(
        mov_mask,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return aligned, aligned_mask, (float(dx), float(dy)), float(resp)


@dataclass
class Rect:
    y: int
    x: int
    h: int
    w: int


def largest_valid_square(mask_u8: NDArray[np.uint8]) -> Rect:
    """ Determines largest area witout boundaries NaNs for cropping """
    mask = (mask_u8 > 0).astype(np.uint8)
    h, w = mask.shape
    if h == 0 or w == 0:
        raise ValueError("Empty mask.")
    dp = np.zeros((h, w), dtype=np.int32)
    best = 0
    by = bx = 0
    dp[0, :] = mask[0, :]
    dp[:, 0] = mask[:, 0]
    best = int(dp.max())
    for y in range(1, h):
        mrow = mask[y, :]
        for x in range(1, w):
            if mrow[x]:
                v = 1 + min(dp[y - 1, x], dp[y, x - 1], dp[y - 1, x - 1])
                dp[y, x] = v
                if v > best:
                    best = int(v)
                    by = y
                    bx = x
    if best <= 0:
        raise ValueError("No valid square region found.")
    y0 = by - best + 1
    x0 = bx - best + 1
    return Rect(y=int(y0), x=int(x0), h=int(best), w=int(best))


def crop_by_rect(img: NDArray[np.float32], rect: Rect) -> NDArray[np.float32]:
    """ Crops image to new boundaries """
    return img[rect.y : rect.y + rect.h, rect.x : rect.x + rect.w]


# --------------------------- I/O helpers -------------------------------
def find_tiffs(folder: str) -> List[str]:
    """ Sorts all tif file in a folder """
    exts = {".tif", ".tiff", ".TIF", ".TIFF"}
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1] in exts
    ]
    return sorted(files)


def group_by_fov(files: Sequence[str]) -> Dict[str, List[Tuple[int, str]]]:
    """ groups all images acquired on a single FOV """

    groups: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for path in files:
        try:
            fov, ordn = parse_filename(path)
        except Exception:
            continue
        groups[fov].append((ordn, path))
    clean: Dict[str, List[Tuple[int, str]]] = {}
    for fov, items in groups.items():
        items.sort()
        clean[fov] = items
    return clean


def tiffinfo_with_pixel_size(
    px_um: float, desc: str
) -> TiffImagePlugin.ImageFileDirectory_v2:
    """ extracts metadata """
    ppc = 10000.0 / px_um
    info = TiffImagePlugin.ImageFileDirectory_v2()
    info[270] = desc
    info[282] = float(ppc)
    info[283] = float(ppc)
    info[296] = 3
    info[305] = "HOLOreader_registered.py"
    return info


def build_description(
    fov: str,
    angle_deg: float,
    pitch_um: float,
    px_um: float,
    shifts: List[Tuple[float, float, float]],
) -> str:
    """ writes metadata """
    nearest = (
        min(NEAREST_SET, key=lambda x: abs(x - pitch_um))
        if np.isfinite(pitch_um)
        else -1
    )
    lines = [
        f"FOV: {fov}",
        f"Rotation applied (deg): {angle_deg:.3f}",
        f"Estimated pitch (µm): {pitch_um:.3f}",
        f"Nearest label: {nearest}",
        f"Pixel size: {px_um:.6f} µm/px (X==Y)",
        f"Phase→Height: h = φ * λ / (2πΔn), λ={WAVELENGTH_UM} µm, Δn={DELTA_N}, scale≈{SCALE_UM_PER_RAD:.5f} µm/rad",
        f"Output mapping: {WINDOW_MIN_UM}..{WINDOW_MAX_UM} µm → 0..255",
        f"Tool: holo_to_tiff.py v{__version__}",
        f"Tool SHA-256: {TOOL_SHA256}",
        "Registration (dx, dy, resp) per frame vs frame1:",
    ]
    for i, (dx, dy, resp) in enumerate(shifts, start=1):
        lines.append(f"  f{i}: dx={dx:.3f}, dy={dy:.3f}, resp={resp:.3f}")
    return "\n".join(lines)


def estimate_pitch_um_from_vertical(
    img_rot: NDArray[np.float32], px_um: float
) -> float:
    """ pitch estimation """
    mask = np.isfinite(img_rot)
    valid = mask.sum(axis=0)
    sum_cols = np.nansum(np.where(mask, img_rot, 0.0), axis=0)
    prof = np.divide(
        sum_cols, valid, out=np.zeros_like(sum_cols, dtype=np.float32), where=valid > 0
    ).astype(np.float32)
    prof -= prof[valid > 0].mean() if np.any(valid > 0) else 0.0
    n = int(prof.size)
    if n < 8:
        return float("nan")
    Y = np.fft.rfft(prof * np.hanning(n))
    mag = np.abs(Y)
    freqs = np.fft.rfftfreq(n, d=1.0)
    # Restrict the search to the physically possible pitch band so a residual
    # low-order background cannot win the argmax and be read as a huge pitch.
    k_hi = max(2, int(np.floor(n * px_um / 15.0)))   # pitch >= 15 um
    k_lo = max(1, int(np.ceil(n * px_um / 120.0)))   # pitch <= 120 um
    if k_lo >= k_hi:
        return float("nan")
    k = k_lo + int(np.argmax(mag[k_lo:k_hi]))
    f = float(freqs[k])
    if f <= 0:
        return float("nan")
    return float((1.0 / f) * px_um)


# ------------------------------ Core ----------------------------------
def process_fov(
    fov: str, items: List[Tuple[int, str]], px_um: float, out_dir: str
) -> None:
    """ Process images with full pipeline from single frames to 8-bit stacks """
    if not items:
        return
    logging.info("FOV %s: loading first frame for angle/pitch", fov)
    u16, desc = read_uint16(items[0][1])
    min_rad, max_rad = parse_phase_minmax(desc)
    h_um = u16_to_height_um(u16, min_rad, max_rad)

    ang_norm = estimate_grating_normal_deg(h_um)
    rot_deg = -ang_norm
    rot_first, mask_first = rotate_bound_nan(h_um, rot_deg)
    pitch_um = estimate_pitch_um_from_vertical(rot_first, px_um)

    rotated: List[NDArray[np.float32]] = [rot_first]
    masks: List[NDArray[np.uint8]] = [mask_first]
    for _, path in tqdm(items[1:], desc=f"Rotate {fov}", leave=False):
        u16_i, desc_i = read_uint16(path)
        mn, mx = parse_phase_minmax(desc_i)
        h_i = u16_to_height_um(u16_i, mn, mx)
        r_i, m_i = rotate_bound_nan(h_i, rot_deg)
        rotated.append(r_i)
        masks.append(m_i)

    ref = rotated[0]
    refm = masks[0]
    shifts: List[Tuple[float, float, float]] = [(0.0, 0.0, 1.0)]
    aligned: List[NDArray[np.float32]] = [ref]
    aligned_masks: List[NDArray[np.uint8]] = [refm]
    for i in range(1, len(rotated)):
        a, am, (dx, dy), resp = register_by_phasecorr(rotated[i], ref, masks[i], refm)
        aligned.append(a)
        aligned_masks.append(am)
        shifts.append((dx, dy, resp))

    # Intersection of masks AFTER registration + enforce finite pixels across all frames
    m_all = aligned_masks[0].copy()
    for m in aligned_masks[1:]:
        cv2.bitwise_and(m_all, m, dst=m_all)
    for arr in aligned:
        finite = (np.isfinite(arr)).astype(np.uint8) * 255
        cv2.bitwise_and(m_all, finite, dst=m_all)

    rect = largest_valid_square(m_all)
    logging.info(
        "FOV %s: crop square (y=%d, x=%d, size=%d)", fov, rect.y, rect.x, rect.h
    )

    frames_u8: List[Image.Image] = []
    cropped_float: List[NDArray[np.float32]] = []
    for arr in aligned:
        c = crop_by_rect(arr, rect)
        cropped_float.append(c)
        u8 = rescale_height_to_u8(c, WINDOW_MIN_UM, WINDOW_MAX_UM)
        frames_u8.append(Image.fromarray(u8))

    # Robust single-frame projection: per-pixel median (height domain)
    stack = np.stack(cropped_float, axis=0)
    fused_h = np.nanmedian(stack, axis=0).astype(np.float32)
    fused_u8 = rescale_height_to_u8(fused_h, WINDOW_MIN_UM, WINDOW_MAX_UM)
    fused_img = Image.fromarray(fused_u8)

    os.makedirs(out_dir, exist_ok=True)
    label = -1
    if np.isfinite(pitch_um):
        nearest = min(NEAREST_SET, key=lambda x: abs(x - pitch_um))
        if abs(nearest - pitch_um) <= 0.15 * nearest:
            label = nearest
        else:
            logging.warning(
                "FOV %s: pitch %.3f um matches no label in %s",
                fov, pitch_um, NEAREST_SET,
            )
    out_name = f"{fov}_{label}.tif" if label != -1 else f"{fov}_NA.tif"
    out_path = os.path.join(out_dir, out_name)
    desc_out = build_description(
        fov=fov, angle_deg=rot_deg, pitch_um=pitch_um, px_um=px_um, shifts=shifts
    )
    info = tiffinfo_with_pixel_size(px_um, desc_out)
    frames_u8[0].save(
        out_path,
        save_all=True,
        append_images=frames_u8[1:],
        compression="tiff_deflate",
        tiffinfo=info,
    )
    logging.info("FOV %s: wrote %s", fov, out_path)

    # Write single-frame fused output as well
    fused_name = f"{fov}_{label}_MEDIAN.tif" if label != -1 else f"{fov}_NA_MEDIAN.tif"
    fused_path = os.path.join(out_dir, fused_name)
    info_single = tiffinfo_with_pixel_size(
        px_um,
        desc_out
        + "\nProjection: per-pixel MEDIAN in height domain before 8-bit mapping.",
    )
    fused_img.save(fused_path, compression="tiff_deflate", tiffinfo=info_single)
    logging.info("FOV %s: wrote %s", fov, fused_path)


def main() -> None:
    """ Main function """
    setup_logging()
    folder, px_um = ask_folder_and_scale()
    all_files = find_tiffs(folder)
    by_fov = group_by_fov(all_files)
    if not by_fov:
        logging.error("No candidate stacks found.")
        sys.exit(4)
    out_dir = os.path.join(folder, "stacks_8bit_square")
    for fov, items in tqdm(by_fov.items(), desc="Process FOVs"):
        if len(items) < 5:
            logging.warning(
                "FOV %s: found %d frames (expected 5); continuing.", fov, len(items)
            )
        process_fov(fov, items, px_um, out_dir)
    logging.info("Done. Output in: %s", out_dir)


if __name__ == "__main__":
    main()
