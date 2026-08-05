from __future__ import annotations

import argparse
import csv
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.ndimage import gaussian_filter  # type: ignore
except ImportError:  # pragma: no cover
    gaussian_filter = None


LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# -----------------------------------------------------------------------------
# Fixed holographic calibration
# -----------------------------------------------------------------------------
XY_UM_PER_PX: float = 0.54
Z0_UM: float = -25.0
DZ_UM_PER_GRAY: float = 75.0 / 255.0


def height_um_to_u8(height_um: np.ndarray) -> np.ndarray:
    """Encode height (µm) into uint8 using fixed z0/dz calibration.
    Returns a uint8 array in [0, 255] with the same shape as input.
    """
    g = np.rint((height_um - Z0_UM) / DZ_UM_PER_GRAY).astype(np.int32)
    return np.clip(g, 0, 255).astype(np.uint8)


def u8_to_height_um(gray_u8: np.ndarray) -> np.ndarray:
    """Decode uint8-encoded height map into micrometers using fixed z0/dz.
    Returns a float32 array of heights in µm.
    """
    return gray_u8.astype(np.float32) * float(DZ_UM_PER_GRAY) + float(Z0_UM)


# -----------------------------------------------------------------------------
# Groove profile model (sinusoidal + per-image phase modulation)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class GrooveShapeModel:
    pm_strength: float
    pm_phase: float
    mid: float
    scale: float

    def eval_unit_ptp(self, phase: np.ndarray) -> np.ndarray:
        """Evaluate the groove waveform at phase in [0, 1).
        Returns float32 with peak-to-trough exactly 1.0 and ~[-0.5, +0.5] range.
        """
        t = phase.astype(np.float64, copy=False)
        ang = 2.0 * math.pi * t
        if self.pm_strength != 0.0:
            y = np.sin(ang + self.pm_strength * np.sin(ang + self.pm_phase))
        else:
            y = np.sin(ang)
        return ((y - self.mid) * self.scale).astype(np.float32, copy=False)


def _compute_mid_scale_for_pm(
    pm_strength: float,
    pm_phase: float,
    n: int = 4096,
) -> Tuple[float, float]:
    """Compute midpoint and scale for a phase-modulated sine.
    Returns (mid, scale) so the waveform is centered and has unit peak-to-trough.
    """
    t = np.linspace(0.0, 1.0, int(n), endpoint=False, dtype=np.float64)
    ang = 2.0 * math.pi * t
    if pm_strength != 0.0:
        y = np.sin(ang + pm_strength * np.sin(ang + pm_phase))
    else:
        y = np.sin(ang)

    ymin = float(np.min(y))
    ymax = float(np.max(y))
    ptp = ymax - ymin
    if not np.isfinite(ptp) or ptp < 1e-12:
        return float(np.mean(y)), 1.0

    return 0.5 * (ymin + ymax), 1.0 / ptp


def sample_groove_shape_model(
    rng: np.random.Generator,
    *,
    jitter_strength: float,
    jitter_clip_sigma: float,
) -> GrooveShapeModel:
    """Sample per-image groove waveform parameters.
    Returns a GrooveShapeModel with normalized unit peak-to-trough amplitude.
    """
    if jitter_strength <= 0.0:
        pm_strength = 0.0
        pm_phase = 0.0
    else:
        m = float(abs(rng.normal(0.0, float(jitter_strength))))
        if jitter_clip_sigma > 0.0:
            m = min(m, float(jitter_strength) * float(jitter_clip_sigma))
        pm_strength = m
        pm_phase = float(rng.uniform(0.0, 2.0 * math.pi))

    mid, scale = _compute_mid_scale_for_pm(pm_strength, pm_phase)
    return GrooveShapeModel(
        pm_strength=pm_strength,
        pm_phase=pm_phase,
        mid=mid,
        scale=scale,
    )


# -----------------------------------------------------------------------------
# Noise helpers (scipy gaussian_filter preferred; numpy fallback)
# -----------------------------------------------------------------------------
def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    """Create a 1D Gaussian kernel for the given sigma.
    Returns a float32 kernel normalized to sum to 1.
    """
    if sigma <= 0.0:
        return np.array([1.0], dtype=np.float32)
    radius = int(max(1, math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * (sigma * sigma)))
    return (k / float(k.sum())).astype(np.float32, copy=False)


def _gaussian_filter_fallback(img: np.ndarray, sigma_yx: Tuple[float, float]) -> np.ndarray:
    """Blur an image using a separable Gaussian implemented in NumPy.
    Returns float32 blurred image; used only when SciPy is unavailable.
    """
    sy, sx = sigma_yx
    if sy <= 0.0 and sx <= 0.0:
        return img.astype(np.float32, copy=False)

    out = img.astype(np.float32, copy=False)
    ky = _gaussian_kernel1d(float(sy))
    kx = _gaussian_kernel1d(float(sx))

    if ky.size > 1:
        pad = ky.size // 2
        tmp = np.pad(out, ((pad, pad), (0, 0)), mode="reflect")
        out = np.apply_along_axis(lambda v: np.convolve(v, ky, mode="valid"), 0, tmp)

    if kx.size > 1:
        pad = kx.size // 2
        tmp = np.pad(out, ((0, 0), (pad, pad)), mode="reflect")
        out = np.apply_along_axis(lambda v: np.convolve(v, kx, mode="valid"), 1, tmp)

    return out.astype(np.float32, copy=False)


def _gblur(img: np.ndarray, sigma_yx: Tuple[float, float]) -> np.ndarray:
    """Apply a Gaussian blur with reflect padding.
    Returns float32 blurred output using SciPy when available.
    """
    if gaussian_filter is not None:
        return gaussian_filter(img, sigma=sigma_yx, mode="reflect").astype(np.float32, copy=False)
    return _gaussian_filter_fallback(img, sigma_yx)


# -----------------------------------------------------------------------------
# Generator configuration
# -----------------------------------------------------------------------------
@dataclass
class SyntheticHoloConfig:
    """Configuration for synthetic holographic height-map generation.
    Stores sweep parameters and noise/texture controls used by the generator.
    """

    shape_px: Tuple[int, int] = (1000, 1000)
    pitches_um: Tuple[float, ...] = (20.0, 40.0, 60.0, 80.0)
    depths_um: Tuple[float, ...] = (4.0, 12.0, 30.0)

    shape_jitter_strength: float = 0.10
    shape_jitter_clip_sigma: float = 3.0

    base_height_um_mean: float = 4.0
    base_height_um_jitter: float = 1.0

    tilt_ptp_um_mean: float = 0.8
    tilt_ptp_um_jitter: float = 0.6

    pitch_cv: float = 0.010
    depth_cv: float = 0.040
    row_phase_jitter_std: float = 0.0015

    residual_low_sigma_yx: Tuple[float, float] = (24.0, 8.0)
    residual_mid_sigma_yx: Tuple[float, float] = (4.0, 2.0)
    residual_low_weight: float = 0.9
    residual_mid_weight: float = 0.55

    shading_col_sigma_px: float = 90.0
    shading_row_sigma_px: float = 70.0
    shading_std_um: float = 0.35

    white_noise_std_um: float = 0.18


# -----------------------------------------------------------------------------
# Core generator
# -----------------------------------------------------------------------------
class HolographicSyntheticGenerator:
    """Synthetic height-map generator for periodic groove patterns with holographic-like noise."""

    def __init__(self, cfg: SyntheticHoloConfig, seed: Optional[int] = None) -> None:
        """Initialize the generator with a config and optional RNG seed.
        Stores config and a NumPy Generator for reproducible sampling.
        """
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def _residual_std_um(self, pitch_um: float, depth_um: float) -> float:
        """Compute target residual texture std (µm) from pitch and depth.
        Returns a non-negative float standard deviation in micrometers.
        """
        base = 0.8 + 0.03 * float(pitch_um)
        cap = 1.0 + 0.15 * float(depth_um)
        return float(min(base, cap))

    def _make_plane(self, ny: int, nx: int, tilt_ptp_um: float) -> np.ndarray:
        """Create a random tilted plane with specified peak-to-peak amplitude (µm).
        Returns float32 plane of shape (ny, nx).
        """
        if tilt_ptp_um <= 0.0:
            return np.zeros((ny, nx), dtype=np.float32)

        span = math.sqrt((nx - 1) ** 2 + (ny - 1) ** 2)
        theta = float(self.rng.uniform(0.0, 2.0 * math.pi))
        a = (tilt_ptp_um / span) * math.cos(theta)
        b = (tilt_ptp_um / span) * math.sin(theta)

        y_idx, x_idx = np.mgrid[0:ny, 0:nx]
        xc = (nx - 1) / 2.0
        yc = (ny - 1) / 2.0
        plane = a * (x_idx - xc) + b * (y_idx - yc)
        return plane.astype(np.float32, copy=False)

    def _make_pitch_mod(self, nx: int, pitch_px: float) -> np.ndarray:
        """Create a smooth multiplicative pitch modulation along X.
        Returns float32 array of length nx with values clipped near 1.0.
        """
        if self.cfg.pitch_cv <= 0.0:
            return np.ones(nx, dtype=np.float32)

        n_periods = int(max(3, math.ceil(nx / pitch_px) + 2))
        steps = self.rng.normal(0.0, self.cfg.pitch_cv, size=n_periods).astype(np.float32)
        walk = np.cumsum(steps)
        walk -= float(walk.mean())
        walk = _gblur(walk[None, :], (0.0, 1.2)).ravel()

        xp = np.linspace(0, nx - 1, n_periods, dtype=np.float32)
        x = np.arange(nx, dtype=np.float32)
        mod = np.interp(x, xp, 1.0 + walk).astype(np.float32)
        return np.clip(mod, 0.9, 1.1)

    def _make_depth_mod(self, nx: int, pitch_px: float) -> np.ndarray:
        """Create per-period depth modulation mapped to pixels.
        Returns float32 array of length nx centered near 1.0.
        """
        if self.cfg.depth_cv <= 0.0:
            return np.ones(nx, dtype=np.float32)

        n_periods = int(max(3, math.ceil(nx / pitch_px) + 2))
        per = self.rng.normal(1.0, self.cfg.depth_cv, size=n_periods).astype(np.float32)
        per = np.clip(per, 0.6, 1.6)

        x = np.arange(nx, dtype=np.float32)
        idx = np.floor(x / float(pitch_px)).astype(np.int32)
        idx = np.clip(idx, 0, n_periods - 1)
        mod = per[idx]
        mod = _gblur(mod[None, :], (0.0, 1.0)).ravel()
        return mod.astype(np.float32, copy=False)

    def _make_residual_texture(self, ny: int, nx: int, target_std_um: float) -> np.ndarray:
        """Generate residual texture plus shading drift with target std (µm).
        Returns float32 texture of shape (ny, nx) with approx. target standard deviation.
        """
        if target_std_um <= 0.0:
            return np.zeros((ny, nx), dtype=np.float32)

        w = self.rng.normal(0.0, 1.0, size=(ny, nx)).astype(np.float32)
        low = _gblur(w, self.cfg.residual_low_sigma_yx)
        mid = _gblur(w, self.cfg.residual_mid_sigma_yx)
        tex = self.cfg.residual_low_weight * low + self.cfg.residual_mid_weight * mid

        col = self.rng.normal(0.0, 1.0, size=(1, nx)).astype(np.float32)
        row = self.rng.normal(0.0, 1.0, size=(ny, 1)).astype(np.float32)
        col = _gblur(col, (0.0, float(self.cfg.shading_col_sigma_px)))
        row = _gblur(row, (float(self.cfg.shading_row_sigma_px), 0.0))
        shade = col + row
        shade -= float(shade.mean())

        shade_std = float(shade.std())
        if not np.isfinite(shade_std) or shade_std <= 0.0:
            shade_std = 1.0
        shade = (shade / shade_std) * float(self.cfg.shading_std_um)
        tex = tex + shade

        s = float(tex.std())
        if not np.isfinite(s) or s <= 0.0:
            return np.zeros((ny, nx), dtype=np.float32)
        return (tex * (float(target_std_um) / s)).astype(np.float32, copy=False)

    def synthesize_height_um(
        self,
        *,
        pitch_um: float,
        depth_um: float,
        phase0: Optional[float] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Generate one synthetic height map (µm) and metadata.
        Returns (height_um float32 [H,W], meta dict of generation parameters).
        """
        ny, nx = self.cfg.shape_px
        pitch_px_nom = float(pitch_um) / XY_UM_PER_PX

        base = float(
            self.cfg.base_height_um_mean
            + self.rng.uniform(-self.cfg.base_height_um_jitter, self.cfg.base_height_um_jitter)
        )

        tilt_ptp = float(
            max(
                0.0,
                self.cfg.tilt_ptp_um_mean
                + self.rng.uniform(-self.cfg.tilt_ptp_um_jitter, self.cfg.tilt_ptp_um_jitter),
            )
        )
        plane = self._make_plane(ny, nx, tilt_ptp)

        if phase0 is None:
            phase0 = float(self.rng.uniform(0.0, 1.0))

        pitch_mod = self._make_pitch_mod(nx, pitch_px_nom)
        pitch_px = np.clip(
            pitch_px_nom * pitch_mod,
            pitch_px_nom * 0.92,
            pitch_px_nom * 1.08,
        ).astype(np.float32)

        dphi = (1.0 / pitch_px).astype(np.float32)
        phi = np.cumsum(dphi)
        phi = (phi + float(phase0)) % 1.0

        shape_model = sample_groove_shape_model(
            self.rng,
            jitter_strength=self.cfg.shape_jitter_strength,
            jitter_clip_sigma=self.cfg.shape_jitter_clip_sigma,
        )
        shape_1d = shape_model.eval_unit_ptp(phi)

        depth_mod = self._make_depth_mod(nx, pitch_px_nom)
        groove_1d = (float(depth_um) * depth_mod * shape_1d).astype(np.float32)

        if self.cfg.row_phase_jitter_std > 0.0:
            row_jitter = self.rng.normal(0.0, self.cfg.row_phase_jitter_std, size=ny).astype(
                np.float32
            )
        else:
            row_jitter = np.zeros(ny, dtype=np.float32)

        groove = np.empty((ny, nx), dtype=np.float32)
        for y in range(ny):
            if row_jitter[y] == 0.0:
                groove[y, :] = groove_1d
                continue
            phi_y = (phi + float(row_jitter[y])) % 1.0
            s = shape_model.eval_unit_ptp(phi_y)
            groove[y, :] = (float(depth_um) * depth_mod * s).astype(np.float32)

        resid_std = self._residual_std_um(pitch_um=float(pitch_um), depth_um=float(depth_um))
        resid = self._make_residual_texture(ny, nx, resid_std)

        if self.cfg.white_noise_std_um > 0.0:
            wn = self.rng.normal(0.0, self.cfg.white_noise_std_um, size=(ny, nx)).astype(
                np.float32
            )
        else:
            wn = 0.0

        h = (base + plane + groove + resid + wn).astype(np.float32)

        lo = float(h.min())
        hi = float(h.max())
        if lo < Z0_UM + 1.0:
            h = h + (Z0_UM + 1.0 - lo)
        if hi > 50.0 - 1.0:
            h = h - (hi - (50.0 - 1.0))

        meta = {
            "pitch_um_target": float(pitch_um),
            "depth_um_target": float(depth_um),
            "pitch_px_nominal": float(pitch_px_nom),
            "phase0_cycles": float(phase0),
            "shape_jitter_strength": float(self.cfg.shape_jitter_strength),
            "shape_pm_strength": float(shape_model.pm_strength),
            "shape_pm_phase_rad": float(shape_model.pm_phase),
            "base_height_um": float(base),
            "tilt_ptp_um": float(tilt_ptp),
            "residual_std_um": float(resid_std),
            "white_noise_std_um": float(self.cfg.white_noise_std_um),
        }
        return h, meta

    def synthesize_u8(
        self,
        *,
        pitch_um: float,
        depth_um: float,
        phase0: Optional[float] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Generate one uint8 height-coded image and metadata.
        Returns (img_u8 uint8 [H,W], meta dict of generation parameters).
        """
        h_um, meta = self.synthesize_height_um(pitch_um=pitch_um, depth_um=depth_um, phase0=phase0)
        return height_um_to_u8(h_um), meta


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------
def _write_tiff_u8(path: Path, img_u8: np.ndarray) -> None:
    """Write a uint8 image to a TIFF file.
    Creates parent directories; writes grayscale minisblack TIFF.
    """
    import tifffile  # local import to keep module import light

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(path),
        img_u8.astype(np.uint8, copy=False),
        photometric="minisblack",
        metadata=None,
    )


def _write_npz_um(path: Path, h_um: np.ndarray) -> None:
    """Write a float32 height map (µm) as a compressed NPZ.
    Creates parent directories; stores as key 'height_um'.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), height_um=h_um.astype(np.float32, copy=False))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for dataset generation.
    Returns argparse.Namespace with all configured parameters.
    """
    p = argparse.ArgumentParser(
        description="Generate synthetic 8-bit holographic height-map TIFFs for groove pitch/depth validation."
    )
    p.add_argument("--n-per-combo", type=int, default=3, help="Images per (pitch, depth) combo.")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility.")
    p.add_argument("--shape", type=str, default="1000,1000", help="Image shape as 'H,W' in pixels.")
    p.add_argument("--save-npz", action="store_true", help="Also save float32 height maps as .npz.")

    p.add_argument(
        "--shape-jitter",
        type=float,
        default=0.10,
        help="Per-image groove-shape variation strength; 0 disables modulation.",
    )
    p.add_argument(
        "--shape-jitter-clip-sigma",
        type=float,
        default=3.0,
        help="Clip shape jitter at this many sigmas; <=0 disables clipping.",
    )
    p.add_argument("--pitches", type=str, default="20,40,60,80", help="Comma-separated pitch values (µm).")
    p.add_argument("--depths", type=str, default="4,12,30", help="Comma-separated depth values (µm).")

    return p.parse_args(argv)


def _parse_csv_floats(s: str) -> Tuple[float, ...]:
    """Parse a comma-separated string into floats.
    Returns a tuple of parsed float values.
    """
    parts = [x.strip() for x in s.split(",") if x.strip()]
    return tuple(float(x) for x in parts)


def _parse_shape_hw(s: str) -> Tuple[int, int]:
    """Parse 'H,W' into (H, W) integers.
    Returns a (height, width) tuple or raises SystemExit on invalid input.
    """
    try:
        shape = tuple(int(x) for x in s.split(","))
    except ValueError as exc:
        raise SystemExit("--shape must be 'H,W' with integers, e.g. 929,929") from exc
    if len(shape) != 2:
        raise SystemExit("--shape must be 'H,W' with integers, e.g. 929,929")
    return int(shape[0]), int(shape[1])


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Generate a synthetic dataset and write TIFFs + truth CSV.
    Returns process exit code 0 on success.
    """
    args = parse_args(argv)

    out_dir = Path(__file__).resolve().parent / f"synthetic_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    shape_hw = _parse_shape_hw(args.shape)
    pitches = _parse_csv_floats(args.pitches)
    depths = _parse_csv_floats(args.depths)

    cfg = SyntheticHoloConfig(
        shape_px=shape_hw,
        pitches_um=pitches,
        depths_um=depths,
        shape_jitter_strength=float(args.shape_jitter),
        shape_jitter_clip_sigma=float(args.shape_jitter_clip_sigma),
    )
    gen = HolographicSyntheticGenerator(cfg, seed=int(args.seed))

    total_samples = len(pitches) * len(depths) * int(args.n_per_combo)
    LOG.info("Generating %d synthetic samples...", total_samples)

    truth_rows: list[dict] = []
    idx = 1
    for pitch_um in cfg.pitches_um:
        for depth_um in cfg.depths_um:
            for _ in range(int(args.n_per_combo)):
                img_u8, meta = gen.synthesize_u8(pitch_um=float(pitch_um), depth_um=float(depth_um))
                fname = f"holo_p{int(pitch_um)}_d{int(depth_um)}_{idx}.tif"
                _write_tiff_u8(out_dir / fname, img_u8)

                truth_rows.append(
                    {
                        "file": fname,
                        "pitch_um": float(pitch_um),
                        "depth_um": float(depth_um),
                        "xy_um_per_px": float(XY_UM_PER_PX),
                        "z0_um": float(Z0_UM),
                        "dz_um_per_gray": float(DZ_UM_PER_GRAY),
                        **meta,
                        "seed": int(args.seed),
                        "sample_index": int(idx),
                    }
                )

                if args.save_npz:
                    h_um = u8_to_height_um(img_u8)
                    _write_npz_um(out_dir / fname.replace(".tif", "_truth_height_um.npz"), h_um)

                LOG.info("Generated %d/%d samples", idx, total_samples)
                idx += 1

    csv_path = out_dir / "truth_index.csv"
    fieldnames = sorted(truth_rows[0].keys()) if truth_rows else ["file"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(truth_rows)

    print(f"Wrote {len(truth_rows)} TIFFs to: {out_dir}")
    print(f"Wrote ground truth CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
