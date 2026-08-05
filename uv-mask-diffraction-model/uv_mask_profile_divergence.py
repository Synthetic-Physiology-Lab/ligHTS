#!/usr/bin/env python3
"""
UV (365 nm) intensity profiles behind periodic photomask lines.

"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
from dataclasses import dataclass

try:
    import tkinter as tk  # type: ignore
    from tkinter import filedialog, messagebox  # type: ignore
    HAS_TK: bool = True
except Exception:  # pragma: no cover
    tk = None  # type: ignore
    filedialog = None  # type: ignore
    messagebox = None  # type: ignore
    HAS_TK = False

from typing import Iterable

import matplotlib
import numpy as np

# Non-interactive backend for script runs; comment out for interactive use.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


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


# ============================= CONSTANTS ======================================
WAVELENGTH_NM: float = 365.0
WAVELENGTH_UM: float = WAVELENGTH_NM * 1e-3

# Distance(s) after mask [µm]. Use multiple values for "evolving with distance".
Z_LIST_UM: tuple[float, ...] = (200.0,)

# Angular divergences to plot (half-angles, degrees)
DIVERGENCES_DEG: tuple[float, float, float] = (3.0, 8.0, 15.0)

# Source-direction sampling model:
# - "solid_angle": 3D cone, uniform in solid angle (recommended if you say "cone")
# - "fan_x": 1D fan of tilts in x only (closer to the original script)
CONE_MODEL: str = "solid_angle"

# Sampling density (deterministic grid)
N_THETA: int = 9  # polar samples (solid_angle) OR tilt samples (fan_x)
N_PHI: int = 17   # azimuthal samples (solid_angle only)

# Mask geometry (50% duty cycle: line_width = pitch / 2)
# These are the PERIODICITIES (pitch values) in micrometers
PITCH_VALUES_UM: tuple[int, int, int] = (40, 60, 80)

# Spatial sampling in x [µm] (used to sample one pitch)
DX_TARGET_UM: float = 0.2

# Plotting / export
FIG_W_CM: float = 9.0
FIG_H_CM: float = 6.0
DPI: int = 300
LINE_WIDTH: float = 2.0
X_HALFSPAN_UM: float = 125.0  # plot xlim = ± X_HALFSPAN_UM (fixed span in micrometers)

EXPORT_CSV: bool = True
SHOW_LEGEND: bool = False  # keep False to match "no legend" figures

# Reproducibility
NP_FLOAT: type = np.float64
NP_COMPLEX: type = np.complex128


# ============================= DATA TYPES =====================================
@dataclass(frozen=True)
class MaskSpec:
    """Periodic 1D binary line grating spec"""

    width_um: float
    pitch_um: float
    inverted: bool  # False: transparent lines on opaque background; True: opposite


# =============================== UTILITIES ====================================
def set_pub_style() -> None:
    """Arial Bold 12; falls back if Arial is not available on the system."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.weight": "bold",
            "font.size": 12,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "mathtext.default": "regular",
        }
    )


def ask_output_folder() -> str:
    """Pick an output folder (GUI when available). Defaults to CWD."""
    if not HAS_TK:
        folder = os.getcwd()
        os.makedirs(folder, exist_ok=True)
        return folder

    try:
        root = tk.Tk()  # type: ignore[union-attr]
        root.withdraw()
        messagebox.showinfo(  # type: ignore[union-attr]
            "UV intensity profiles (divergence)",
            "Select a folder to save the PNG figures (and CSVs if enabled).",
        )
        folder = filedialog.askdirectory(title="Select output folder")  # type: ignore[union-attr]
        root.destroy()
    except Exception:
        # Headless environments may import tkinter but cannot create a window.
        folder = ""

    if not folder:
        folder = os.getcwd()
    os.makedirs(folder, exist_ok=True)
    return folder


def cm_to_inch(cm: float) -> float:
    return cm / 2.54


# ============================ MASK / COEFFICIENTS =============================
def fourier_coefficients_binary_grating(
    m: np.ndarray, duty: float, inverted: bool
) -> np.ndarray:
    """
    Analytic Fourier-series coefficients c_m for a centered binary amplitude grating.
    """

    m = m.astype(NP_FLOAT)
    c = duty * np.sinc(duty * m).astype(NP_FLOAT)
    if inverted:
        c = -c
        c[m == 0] = 1.0 - duty
    return c.astype(NP_COMPLEX)


def mask_nominal_intensity_one_period(x_um: np.ndarray, spec: MaskSpec) -> np.ndarray:
    """Nominal mask-plane transmission intensity t(x)^2 over one period centered at 0."""
    phase = ((x_um + 0.5 * spec.pitch_um) % spec.pitch_um) - 0.5 * spec.pitch_um
    open_region = np.abs(phase) <= (0.5 * spec.width_um)
    t = np.where(open_region, 1.0, 0.0).astype(NP_FLOAT)
    if spec.inverted:
        t = 1.0 - t
    return (t * t).astype(NP_FLOAT)


# ============================ SOURCE SAMPLING =================================
def sample_directions(
    half_angle_deg: float,
    model: str,
    n_theta: int,
    n_phi: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministically sample illumination directions based on the model:
        - "solid_angle": uniform in solid angle over a cone (theta in [0, theta_max])
        - "fan_x": uniform in tilt angle over [-theta_max, +theta_max] with ky=0
    """
    if half_angle_deg <= 0 or n_theta < 1:
        return (
            np.array([0.0], dtype=NP_FLOAT),
            np.array([0.0], dtype=NP_FLOAT),
            np.array([1.0], dtype=NP_FLOAT),
        )

    theta_max = math.radians(float(half_angle_deg))
    model = model.strip().lower()

    if model == "fan_x":
        thetas = np.linspace(-theta_max, theta_max, int(max(1, n_theta)), dtype=NP_FLOAT)
        sin_t = np.sin(thetas)
        sx = sin_t  # sin(theta) in x-z plane
        sy = np.zeros_like(sx)
        w = np.ones_like(sx)
        return sx, sy, (w / float(w.sum())).astype(NP_FLOAT)

    # "solid_angle" (default)
    sx_list, sy_list, w_list = [], [], []

    # Include on-axis direction
    sx_list.append(0.0)
    sy_list.append(0.0)
    w_list.append(1.0)

    for i_theta in range(1, int(n_theta)):
        theta = theta_max * float(i_theta) / float(n_theta - 1) if n_theta > 1 else theta_max
        sin_t = math.sin(theta)

        # Azimuthal sampling: enforce phi and phi+pi pairing for symmetry
        n_phi_ring = max(1, int(n_phi) if i_theta == 1 else int(n_phi) * i_theta)
        phi_vals = np.linspace(0, 2.0 * math.pi, n_phi_ring + 1)[:-1]

        for phi in phi_vals:
            sx = sin_t * math.cos(phi)
            sy = sin_t * math.sin(phi)
            sx_list.append(sx)
            sy_list.append(sy)
            w_list.append(1.0)

    sx_arr = np.array(sx_list, dtype=NP_FLOAT)
    sy_arr = np.array(sy_list, dtype=NP_FLOAT)
    w_arr = np.array(w_list, dtype=NP_FLOAT)
    return sx_arr, sy_arr, (w_arr / float(w_arr.sum())).astype(NP_FLOAT)


# ============================ DIFFRACTION =====================================
def intensity_one_period_for_direction(
    pitch_um: float,
    x0_um: np.ndarray,
    m_orders: np.ndarray,
    c_m: np.ndarray,
    wavelength_um: float,
    z_um: float,
    sin_theta_cos_phi: float,
    sin_theta_sin_phi: float,
) -> np.ndarray:
    """
    Compute intensity over one period for a single plane-wave direction.

    """
    k = 2.0 * math.pi / wavelength_um
    kx0 = k * float(sin_theta_cos_phi)
    ky0 = k * float(sin_theta_sin_phi)

    kx_m = kx0 + (2.0 * math.pi * m_orders.astype(NP_FLOAT) / float(pitch_um))
    kz_m = np.sqrt((k * k - kx_m * kx_m - ky0 * ky0) + 0j).astype(NP_COMPLEX)

    # Build spectrum in FFT bin ordering (matching m_orders from fftfreq)
    n = int(x0_um.size)
    spectrum = (n * c_m * np.exp(1j * kz_m * float(z_um))).astype(NP_COMPLEX)
    periodic_part = np.fft.ifft(spectrum)  # samples at x0 in [0, pitch)

    field = periodic_part * np.exp(1j * kx0 * x0_um.astype(NP_FLOAT))
    intensity = (field.real * field.real + field.imag * field.imag).astype(NP_FLOAT)
    return intensity


def incoherent_cone_average_one_period(
    pitch_um: float,
    x0_um: np.ndarray,
    m_orders: np.ndarray,
    c_m: np.ndarray,
    wavelength_um: float,
    z_um: float,
    half_angle_deg: float,
    model: str,
    n_theta: int,
    n_phi: int,
) -> np.ndarray:
    """Weighted incoherent average intensity over a cone/fan of directions."""
    sx, sy, w = sample_directions(half_angle_deg, model=model, n_theta=n_theta, n_phi=n_phi)

    acc = np.zeros_like(x0_um, dtype=NP_FLOAT)
    wsum = float(np.sum(w))
    if not np.isfinite(wsum) or wsum <= 0:
        wsum = 1.0

    for sxi, syi, wi in zip(sx, sy, w):
        I = intensity_one_period_for_direction(
            pitch_um=pitch_um,
            x0_um=x0_um,
            m_orders=m_orders,
            c_m=c_m,
            wavelength_um=wavelength_um,
            z_um=z_um,
            sin_theta_cos_phi=float(sxi),
            sin_theta_sin_phi=float(syi),
        )
        acc += float(wi) * I

    return (acc / wsum).astype(NP_FLOAT)


# =============================== PIPELINE =====================================
def build_one_period_grid(pitch_um: float, dx_target_um: float) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Build a one-period x-grid x0 in [0, pitch) with spacing ~dx_target.
    Returns: x0_um: ndarray in [0, pitch), dx_um: actual spacing,
             m_orders: integer harmonic indices aligned to numpy FFT bins
    """
    n = int(max(64, math.ceil(float(pitch_um) / float(dx_target_um))))
    x0_um = (np.arange(n, dtype=NP_FLOAT) * (float(pitch_um) / float(n))).astype(NP_FLOAT)
    dx_um = float(pitch_um) / float(n)

    # FFT bin frequencies correspond to m/pitch cycles/µm => m = fx * pitch
    m_float = np.fft.fftfreq(n, d=dx_um) * float(pitch_um)
    m_orders = np.rint(m_float).astype(int)

    # Integrity check: should be integer-valued within numerical tolerance
    if not np.allclose(m_float, m_orders.astype(NP_FLOAT), atol=1e-9):
        raise RuntimeError("Unexpected non-integer FFT harmonic mapping; check grid construction.")

    return x0_um, dx_um, m_orders


def compute_profiles(
    spec: MaskSpec,
    z_um: float,
    divergences_deg: Iterable[float],
) -> dict[str, np.ndarray]:
    """
    Compute divergence-averaged intensity profiles and nominal mask intensity.

    Returns dict with keys: d003, d008, d015, mask_nominal, and x_um.
    """
    pitch = float(spec.pitch_um)
    x0_um, dx_um, m_orders = build_one_period_grid(pitch_um=pitch, dx_target_um=DX_TARGET_UM)

    # Fourier coefficients for 50% duty cycle
    duty = 0.5
    c_m = fourier_coefficients_binary_grating(m_orders, duty=duty, inverted=spec.inverted)

    # Compute one-period (0..pitch) intensity for each divergence
    I_periods: dict[str, np.ndarray] = {}
    for d in divergences_deg:
        I0 = incoherent_cone_average_one_period(
            pitch_um=pitch,
            x0_um=x0_um,
            m_orders=m_orders,
            c_m=c_m,
            wavelength_um=WAVELENGTH_UM,
            z_um=float(z_um),
            half_angle_deg=float(d),
            model=CONE_MODEL,
            n_theta=N_THETA,
            n_phi=N_PHI,
        )
        # Shift to x in [-pitch/2, pitch/2)
        I_shift = np.fft.fftshift(I0).astype(NP_FLOAT)
        I_periods[f"d{int(round(d)):03d}"] = I_shift

    # Build centered one-period x grid
    n = int(x0_um.size)
    x_centered = (np.arange(n, dtype=NP_FLOAT) - n // 2) * dx_um

    # Nominal mask intensity on the centered grid
    mask_nom = mask_nominal_intensity_one_period(x_centered, spec)

    # Build plot x-grid spanning multiple pitches, sampled at same dx_um.
    # Use a symmetric grid including both endpoints to avoid edge-only asymmetry in plots/CSVs.
    halfspan = X_HALFSPAN_UM
    n_plot = int(round((2.0 * halfspan) / dx_um)) + 1
    x_plot = np.linspace(-halfspan, halfspan, n_plot, dtype=NP_FLOAT)

    # Map x_plot into one-period indices
    idx = np.mod(np.rint((x_plot + 0.5 * pitch) / dx_um).astype(int), n)
    out: dict[str, np.ndarray] = {"x_um": x_plot, "mask_nominal": mask_nom[idx].astype(NP_FLOAT)}
    for k, I_shift in I_periods.items():
        out[k] = I_shift[idx].astype(NP_FLOAT)

    return out


def save_csv(out_csv: str, profs: dict[str, np.ndarray]) -> None:
    """Save tidy CSV with x and all profiles."""
    keys = ["x_um"] + sorted([k for k in profs.keys() if k.startswith("d")]) + ["mask_nominal"]
    data = np.column_stack([profs[k] for k in keys]).astype(NP_FLOAT)

    header = ",".join(keys)
    np.savetxt(out_csv, data, delimiter=",", header=header, comments="")


def plot_profile_png(
    profs: dict[str, np.ndarray],
    out_png: str,
    title: str,
) -> None:
    """Plot profile figure with publication style."""
    set_pub_style()
    fig, ax = plt.subplots(
        figsize=(cm_to_inch(FIG_W_CM), cm_to_inch(FIG_H_CM)),
        dpi=DPI,
    )

    x = profs["x_um"]
    keys = sorted([k for k in profs.keys() if k.startswith("d")])
    for k in keys:
        ax.plot(x, profs[k], linewidth=LINE_WIDTH, label=k)

    ax.plot(x, profs["mask_nominal"], linestyle=":", linewidth=LINE_WIDTH, label="mask")

    ax.set_xlabel("x [µm]")
    ax.set_ylabel("Normalised intensity")

    ax.set_xlim(float(x.min()), float(x.max()))
    ax.grid(True, linestyle="--", linewidth=0.6)

    if SHOW_LEGEND:
        ax.legend(frameon=False)

    ax.set_title(title)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_png)
    plt.close(fig)


# ================================ MAIN ========================================
def main() -> int:
    out_dir = ask_output_folder()

    run_meta = {
        "tool": "uv_mask_profile_divergence.py",
        "version": __version__,
        "tool_sha256": TOOL_SHA256,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "params": {
            "wavelength_nm": WAVELENGTH_NM,
            "z_list_um": list(Z_LIST_UM),
            "divergences_deg": list(DIVERGENCES_DEG),
            "cone_model": CONE_MODEL,
            "n_theta": N_THETA,
            "n_phi": N_PHI,
            "pitch_values_um": list(PITCH_VALUES_UM),
            "dx_target_um": DX_TARGET_UM,
        },
    }
    try:
        with open(os.path.join(out_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(run_meta, f, indent=2)
    except Exception as exc:  # provenance is best-effort
        print(f"Warning: could not write run_metadata.json: {exc}")

    for pitch_um in PITCH_VALUES_UM:
        # For 50% duty cycle: line width = pitch / 2
        width_um = float(pitch_um) / 2.0
        
        for inverted in (False, True):
            spec = MaskSpec(width_um=width_um, pitch_um=float(pitch_um), inverted=inverted)
            pol = "opaque_lines" if inverted else "transparent_lines"

            for z_um in Z_LIST_UM:
                profs = compute_profiles(spec=spec, z_um=float(z_um), divergences_deg=DIVERGENCES_DEG)

                tag = (
                    f"w{width_um:.1f}um_pitch{int(pitch_um)}um_"
                    f"z{float(z_um):.0f}um_div_{int(DIVERGENCES_DEG[0])}_"
                    f"{int(DIVERGENCES_DEG[1])}_{int(DIVERGENCES_DEG[2])}deg_{pol}"
                )

                out_png = os.path.join(out_dir, f"profile_{tag}.png")
                out_csv = os.path.join(out_dir, f"profile_{tag}.csv")

                title = f"{width_um:.1f} µm lines, pitch {int(pitch_um)} µm, z={float(z_um):.0f} µm"
                plot_profile_png(profs=profs, out_png=out_png, title=title)

                if EXPORT_CSV:
                    save_csv(out_csv=out_csv, profs=profs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
