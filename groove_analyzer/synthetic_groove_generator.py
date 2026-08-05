#!/usr/bin/env python3
"""
Synthetic Groove Generator - Physics-based confocal microscopy dataset generator.

Generates synthetic grooved gel surfaces with realistic confocal imaging artifacts:
PSF convolution, Poisson noise, read noise, and quantization for analyzer validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    from scipy.ndimage import gaussian_filter
    from scipy.interpolate import interp1d
except ImportError as e:
    raise RuntimeError("scipy is required: pip install scipy") from e

try:
    import tifffile
except ImportError as e:
    raise RuntimeError("tifffile is required: pip install tifffile") from e

# ============================================================================
# Configuration
# ============================================================================


@dataclass
class ImagingConfig:
    """
    Confocal microscopy calibration and noise parameters.
    Must match analyzer expectations (xy_um, dz_um) for validation consistency.
    """

    # Pixel/voxel dimensions
    xy_um: float = 1.34  # XY pixel size (µm)
    dz_um: float = 0.9  # Z step size (µm)

    # Field of view and max planes
    fov_um: float = 436.84
    nz_max: int = 280

    # Noise and PSF models
    poisson_peak_photons: int = 2400
    background_frac: float = 0.25
    read_noise_electrons: float = 5.0
    bit_depth: int = 16

    # PSF FWHM (µm) - fixed values typical of confocal system
    psf_fwhm_xy_um: float = 0.90
    psf_fwhm_z_um: float = 3.2


@dataclass
class GrooveConfig:
    """Groove geometry: sinusoidal profile with stochastic pitch/depth variation."""

    pitch_um: float = 60.0
    depth_um: float = 14.0
    eng_CV: float = 0.03
    groove_angle_deg: float = 0.0  # Default rotation from vertical


@dataclass
class SurfaceConfig:
    """Gel surface tilt and roughness for realistic imaging conditions."""

    tilt_x_deg: float = 0.0  # Default tilt about X axis
    tilt_y_deg: float = 0.0  # Default tilt about Y axis

    # Surface roughness
    roughness_rms_um: float = 0.25
    roughness_corr_len_um: float = 5.0


@dataclass
class ValidationConfig:
    """Validation dataset parameter sweep ranges and acceptance thresholds."""

    pitch_range_um: tuple[float, float] = (20.0, 80.0)
    depth_range_um: tuple[float, float] = (4.0, 30.0)
    groove_tilt_max_deg: float = 10.0
    surface_tilt_max_deg: float = 4.0

    pitch_mape_thresh_pct: float = 3.0
    depth_mape_thresh_pct: float = 5.0
    angle_err_thresh_deg: float = 1.0
    height_rmse_thresh_um: float = 1.8


# Hydrogel height targets (µm). A dataset is generated for each value.
HYDROGEL_HEIGHT_VALUES_UM: list[float] = [30.0, 60.0, 90.0]

# ============================================================================
# Logging
# ============================================================================

LOG = logging.getLogger("synth_groove_validation")
LOG.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
LOG.handlers = [_handler]

# ============================================================================
# Synthetic Surface Generation
# ============================================================================


class SyntheticGrooveSurface:
    """
    Generate grooved gel height map: h(x,y) = groove(u) + tilt(x,y) + roughness(x,y).
    Grooves are sinusoidal with stochastic pitch/depth per period (eng_CV variation).
    """

    def __init__(
        self,
        imaging: ImagingConfig,
        groove: GrooveConfig,
        surface: SurfaceConfig,
        rng: np.random.Generator,
    ):
        self.imaging = imaging
        self.groove = groove
        self.surface = surface
        self.rng = rng

        # Compute image dimensions
        self.nx = int(np.ceil(imaging.fov_um / imaging.xy_um))
        self.ny = self.nx

        # Create coordinate grids
        x = np.arange(self.nx) * imaging.xy_um
        y = np.arange(self.ny) * imaging.xy_um
        self.xx, self.yy = np.meshgrid(x, y, indexing="xy")

        # Store realized groove parameters
        self.realized_pitches: list[float] = []
        self.realized_depths: list[float] = []

    def generate_groove_profile(self) -> np.ndarray:
        """
        Build 1D sinusoidal groove profile with per-period pitch/depth variation.
        Returns 2D height map after rotating by groove_angle and interpolating to grid.
        """
        angle_rad = np.deg2rad(self.groove.groove_angle_deg)

        # u is the coordinate perpendicular to grooves
        u = self.xx * np.cos(angle_rad) + self.yy * np.sin(angle_rad)
        u = u - u.min()

        width = float(u.max() - u.min()) + 2 * self.groove.pitch_um
        pitch_nom = float(self.groove.pitch_um)
        depth_nom = float(self.groove.depth_um)
        pitch_std = pitch_nom * float(self.groove.eng_CV)
        depth_std = depth_nom * float(self.groove.eng_CV)

        # Guard against non-physical draws
        min_pitch = max(self.imaging.xy_um * 2.0, pitch_nom * 0.1)
        min_depth = max(depth_nom * 0.1, 0.0)

        # Build 1D profile with variable pitch/depth
        n_samples = int(np.ceil(width / (self.imaging.xy_um * 0.5)))  # Oversample
        u_1d = np.linspace(0, width, n_samples)
        h_1d = np.zeros_like(u_1d)

        curr_u = 0.0
        self.realized_pitches = []
        self.realized_depths = []

        while curr_u < width:
            pitch_i = float(self.rng.normal(pitch_nom, pitch_std))
            if pitch_i < min_pitch:
                pitch_i = float(min_pitch)

            depth_i = float(self.rng.normal(depth_nom, depth_std))
            if depth_i < min_depth:
                depth_i = float(min_depth)

            self.realized_pitches.append(pitch_i)
            self.realized_depths.append(depth_i)

            mask = (u_1d >= curr_u) & (u_1d < curr_u + pitch_i)
            local_u = u_1d[mask] - curr_u

            # Sinusoidal profile: minimum at segment boundaries, maximum at mid-segment
            h_1d[mask] = (depth_i / 2.0) * (1 - np.cos(2 * np.pi * local_u / pitch_i))
            curr_u += pitch_i

        interp_func = interp1d(
            u_1d, h_1d, kind="linear", bounds_error=False, fill_value=0.0
        )
        h_2d = interp_func(u.ravel()).reshape(self.ny, self.nx)
        return h_2d.astype(np.float32)

    def apply_tilt(self, h: np.ndarray) -> np.ndarray:
        """Add planar tilt: Δz = tan(tilt_x)·(x - cx) + tan(tilt_y)·(y - cy)."""
        cx = self.imaging.fov_um / 2.0
        cy = self.imaging.fov_um / 2.0

        tilt_x = np.tan(np.deg2rad(self.surface.tilt_x_deg))
        tilt_y = np.tan(np.deg2rad(self.surface.tilt_y_deg))

        tilt_plane = tilt_x * (self.xx - cx) + tilt_y * (self.yy - cy)
        return h + tilt_plane.astype(np.float32)

    def apply_roughness(self, h: np.ndarray) -> np.ndarray:
        """Add correlated surface roughness via Gaussian-filtered noise."""
        if self.surface.roughness_rms_um <= 0:
            return h

        noise = self.rng.normal(0.0, 1.0, h.shape)

        sigma_px = self.surface.roughness_corr_len_um / self.imaging.xy_um
        if sigma_px > 0.5:
            noise = gaussian_filter(noise, sigma_px)
            noise = noise / (np.std(noise) + 1e-10) * self.surface.roughness_rms_um
        else:
            noise *= self.surface.roughness_rms_um

        return h + noise.astype(np.float32)

    def generate_height_map(self) -> np.ndarray:
        """Generate complete height map: grooves + tilt + roughness."""
        h = self.generate_groove_profile()
        h = self.apply_tilt(h)
        h = self.apply_roughness(h)
        return h

    def get_realized_stats(self) -> dict[str, float]:
        """Compute mean/std of realized pitch and depth for ground truth CSV."""
        pitches = np.asarray(self.realized_pitches, dtype=np.float64)
        depths = np.asarray(self.realized_depths, dtype=np.float64)

        return {
            "pitch_realized_mean_um": (
                float(np.mean(pitches)) if pitches.size else np.nan
            ),
            "pitch_realized_std_um": (
                float(np.std(pitches, ddof=1)) if pitches.size > 1 else 0.0
            ),
            "depth_realized_mean_um": (
                float(np.mean(depths)) if depths.size else np.nan
            ),
            "depth_realized_std_um": (
                float(np.std(depths, ddof=1)) if depths.size > 1 else 0.0
            ),
            "n_periods": int(pitches.size),
        }


# ============================================================================
# Confocal Simulation
# ============================================================================


class ConfocalSimulator:
    """
    Forward model: height map → confocal z-stack with PSF, noise, quantization.
    Implements 3D Gaussian PSF with pixel-aperture integration.
    """

    def __init__(self, imaging: ImagingConfig, rng: np.random.Generator):
        self.imaging = imaging
        self.rng = rng

        # FWHM -> sigma
        sigma_xy_um = float(imaging.psf_fwhm_xy_um) / 2.355
        sigma_z_um = float(imaging.psf_fwhm_z_um) / 2.355

        # Finite pixel aperture (box integration) in quadrature.
        aperture_xy = imaging.xy_um / np.sqrt(12.0)
        aperture_z = imaging.dz_um / np.sqrt(12.0)
        sigma_xy_um = float(np.sqrt(sigma_xy_um**2 + aperture_xy**2))
        sigma_z_um = float(np.sqrt(sigma_z_um**2 + aperture_z**2))

        self.sigma_xy = sigma_xy_um
        self.sigma_z = sigma_z_um

    def simulate_zstack(
        self,
        height_map: np.ndarray,
        pre_surface_planes: int = 0,
    ) -> tuple[np.ndarray, float, float]:
        """
        Generate confocal z-stack from height map via 3D PSF convolution and noise.
        Returns (volume, z_min_um, z_max_um) where volume is quantized uint8/uint16.
        """
        ny, nx = height_map.shape
        h_valid = height_map[np.isfinite(height_map)]
        if h_valid.size == 0:
            dtype = np.uint8 if self.imaging.bit_depth <= 8 else np.uint16
            empty_vol = np.zeros((16, ny, nx), dtype=dtype)
            z_max_um = float((16 - 1) * self.imaging.dz_um)
            return empty_vol, 0.0, z_max_um

        z_min = float(np.nanmin(h_valid))
        z_max = float(np.nanmax(h_valid))

        # Margin ABOVE the surface for PSF tails
        margin_above = 3.0 * self.sigma_z

        dz = float(self.imaging.dz_um)
        pre = int(max(0, pre_surface_planes))

        # Anchor stack start (z_min) and optionally prepend empty planes
        z_min_eff = z_min - pre * dz
        z_max_eff = z_max + margin_above

        nz = int(np.ceil((z_max_eff - z_min_eff) / dz)) + 1
        nz = max(16, min(nz, int(self.imaging.nz_max)))

        # Keep z_min_eff fixed after clamping nz
        z_max_eff = z_min_eff + (nz - 1) * dz

        # Render surface into a volume
        vol = np.zeros((nz, ny, nx), dtype=np.float32)
        vol = self._render_surface(vol, height_map, z_min_eff)

        # Apply 3D PSF convolution
        sigma_xy_px = self.sigma_xy / self.imaging.xy_um
        sigma_z_px = self.sigma_z / self.imaging.dz_um
        vol = gaussian_filter(vol, sigma=[sigma_z_px, sigma_xy_px, sigma_xy_px])

        # Mask regions outside the surface (NaNs)
        mask_2d = np.isfinite(height_map)
        vol[:, ~mask_2d] = 0.0

        # Apply noise model and quantize
        vol_q = self._apply_noise(vol)
        return vol_q, z_min_eff, z_max_eff

    def _render_surface(
        self, vol: np.ndarray, height_map: np.ndarray, z_min_eff: float
    ) -> np.ndarray:
        """Render surface as a delta function in Z."""
        nz, ny, nx = vol.shape
        z_idx = ((height_map - z_min_eff) / self.imaging.dz_um).astype(np.float32)
        z_int = np.round(z_idx).astype(np.int64)

        valid = np.isfinite(height_map) & (z_int >= 0) & (z_int < nz)
        yy, xx = np.where(valid)
        vol[z_int[yy, xx], yy, xx] = 1.0
        return vol

    def _apply_noise(self, vol: np.ndarray) -> np.ndarray:
        """Apply shot noise, read noise, background, and quantization."""
        dtype = np.uint8 if self.imaging.bit_depth <= 8 else np.uint16

        vmax = float(np.max(vol))
        if vmax <= 0.0:
            return np.zeros(vol.shape, dtype=dtype)

        background = float(self.imaging.background_frac)
        vol_n = vol / vmax
        # Mix background to keep the peak at <= 1.0 (avoid saturation plateaus)
        vol_n = (1.0 - background) * vol_n + background

        peak_photons = max(1, int(self.imaging.poisson_peak_photons))
        signal = vol_n * peak_photons

        # Poisson shot noise
        signal = self.rng.poisson(lam=np.clip(signal, 0.0, None)).astype(np.float32)

        # Add read noise
        read_noise_std = float(self.imaging.read_noise_electrons)
        read_noise = self.rng.normal(0.0, read_noise_std, signal.shape)
        signal = signal + read_noise.astype(np.float32)

        # Normalize to [0,1]
        signal = np.clip(signal / peak_photons, 0.0, 1.0)

        max_val = (1 << int(self.imaging.bit_depth)) - 1
        return np.round(signal * max_val).astype(dtype)


# ============================================================================
# Dataset Generation
# ============================================================================


@dataclass
class SyntheticSample:
    """
    Ground truth metadata for single synthetic sample.
    Field names match what the validator expects.
    """

    stack_code: str
    file_path: Path
    folder_path: Path

    pitch_nom_um: float
    depth_nom_um: float
    groove_angle_deg: float
    tilt_x_deg: float
    tilt_y_deg: float

    pitch_realized_mean_um: float
    pitch_realized_std_um: float
    depth_realized_mean_um: float
    depth_realized_std_um: float
    n_periods: int

    z_min_um: float
    z_max_um: float
    hydrogel_height_um: float

    scenario: str
    seed: int


def generate_validation_dataset(
    outdir: Path,
    config: ValidationConfig,
    imaging: ImagingConfig,
    seed: int = 42,
) -> list[SyntheticSample]:
    """
    Generate complete validation dataset with systematic sweep.
    Produces HEIGHT × 5 scenarios × 4 pitches × 4 depths = 240 samples.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    rng_master = np.random.default_rng(seed)
    samples: list[SyntheticSample] = []
    sample_id = 0

    scenarios = [
        {
            "name": "clean",
            "groove_angle_range": (0.0, 0.0),
            "tilt_range": (0.0, 0.0),
            "noise_mult": 1.0,
        },
        {
            "name": "rotated",
            "groove_angle_range": (
                -config.groove_tilt_max_deg,
                config.groove_tilt_max_deg,
            ),
            "tilt_range": (0.0, 0.0),
            "noise_mult": 1.0,
        },
        {
            "name": "tilted",
            "groove_angle_range": (0.0, 0.0),
            "tilt_range": (
                -config.surface_tilt_max_deg,
                config.surface_tilt_max_deg,
            ),
            "noise_mult": 1.0,
        },
        {
            "name": "noisy",
            "groove_angle_range": (0.0, 0.0),
            "tilt_range": (0.0, 2.0),
            "noise_mult": 2.5,
        },
        {
            "name": "combined",
            "groove_angle_range": (
                -config.groove_tilt_max_deg,
                config.groove_tilt_max_deg,
            ),
            "tilt_range": (
                -config.surface_tilt_max_deg,
                config.surface_tilt_max_deg,
            ),
            "noise_mult": 1.5,
        },
    ]

    pitch_values = [20.0, 40.0, 60.0, 80.0]
    depth_values = [4.0, 12.0, 20.0, 30.0]

    # Ensure at least 8 periods visible for the largest pitch
    min_periods = 8
    max_pitch = max(pitch_values)
    min_fov = max_pitch * (min_periods + 2)
    base_fov = max(imaging.fov_um, min_fov)

    total_samples = (
        len(HYDROGEL_HEIGHT_VALUES_UM)
        * len(scenarios)
        * len(pitch_values)
        * len(depth_values)
    )
    LOG.info(f"Generating {total_samples} synthetic samples...")
    LOG.info(
        f"Base FOV: {base_fov:.0f} µm "
        f"(ensuring ≥{min_periods} periods for pitch up to {max_pitch:.0f} µm)"
    )

    for hydrogel_height_um in HYDROGEL_HEIGHT_VALUES_UM:
        for scenario in scenarios:
            for pitch in pitch_values:
                required_fov = pitch * (min_periods + 2)
                effective_fov = max(base_fov, required_fov)

                for depth in depth_values:
                    s = int(rng_master.integers(0, 2**31))
                    rng = np.random.default_rng(s)

                    groove_angle = float(rng.uniform(*scenario["groove_angle_range"]))
                    tilt_x = float(rng.uniform(*scenario["tilt_range"]))
                    tilt_y = float(rng.uniform(*scenario["tilt_range"]))

                    h_str = f"H{int(hydrogel_height_um):03d}"
                    p_str = f"P{int(pitch):02d}"
                    d_str = f"D{depth:.1f}"
                    folder_name = (
                        f"{h_str}_{p_str}_{d_str}_{scenario['name']}_"
                    ).replace(".", "p")
                    folder_path = outdir / folder_name
                    folder_path.mkdir(exist_ok=True)

                    imaging_sample = ImagingConfig(**asdict(imaging))
                    imaging_sample.fov_um = float(effective_fov)

                    sample = generate_single_sample(
                        folder_path=folder_path,
                        sample_id=sample_id,
                        pitch_um=float(pitch),
                        depth_um=float(depth),
                        hydrogel_height_um=float(hydrogel_height_um),
                        groove_angle=groove_angle,
                        tilt_x=tilt_x,
                        tilt_y=tilt_y,
                        scenario_name=str(scenario["name"]),
                        noise_mult=float(scenario["noise_mult"]),
                        imaging=imaging_sample,
                        seed=s,
                        rng=rng,
                    )

                    samples.append(sample)
                    sample_id += 1
                    LOG.info(f"Generated {sample_id}/{total_samples} samples")

    write_truth_index(outdir, samples)
    write_dataset_metadata(outdir, imaging)
    LOG.info(f"Dataset generation complete: {len(samples)} samples in {outdir}")
    return samples


def tilt_center_offset_um(tilt_x_deg: float, tilt_y_deg: float, fov_um: float) -> float:
    """
    Compute mid-FOV height after anchoring minimum corner to z=0.
    Returns offset = -min(corner_heights) for planar tilt component.
    """
    tilt_x_slope = float(np.tan(np.deg2rad(tilt_x_deg)))
    tilt_y_slope = float(np.tan(np.deg2rad(tilt_y_deg)))
    cx = float(fov_um) / 2.0
    cy = float(fov_um) / 2.0

    corners = [
        tilt_x_slope * (0.0 - cx) + tilt_y_slope * (0.0 - cy),
        tilt_x_slope * (0.0 - cx) + tilt_y_slope * (fov_um - cy),
        tilt_x_slope * (fov_um - cx) + tilt_y_slope * (0.0 - cy),
        tilt_x_slope * (fov_um - cx) + tilt_y_slope * (fov_um - cy),
    ]
    return -float(min(corners))


def check_and_adjust_tilt(
    hydrogel_height_um: float,
    tilt_x_deg: float,
    tilt_y_deg: float,
    fov_um: float,
    dz_um: float,
) -> tuple[float, float, float]:
    """
    Ensure tilt is compatible with requested hydrogel height.
    Scales tilt slopes if necessary to avoid negative pre_surface_planes.
    """
    hydrogel_height_um = float(hydrogel_height_um)
    dz_um = float(dz_um)

    offset = tilt_center_offset_um(tilt_x_deg, tilt_y_deg, fov_um)
    if offset <= hydrogel_height_um + 1e-6:
        return float(tilt_x_deg), float(tilt_y_deg), float(offset)

    # Offset scales linearly with the tilt plane slopes.
    sx = float(np.tan(np.deg2rad(tilt_x_deg)))
    sy = float(np.tan(np.deg2rad(tilt_y_deg)))

    if offset <= 0.0 or (sx == 0.0 and sy == 0.0):
        return 0.0, 0.0, 0.0

    # Target: offset <= hydrogel height (pre_surface_planes non-negative)
    target_offset = max(0.0, hydrogel_height_um)
    k = min(1.0, max(0.0, (target_offset - 1e-6) / float(offset)))

    sx2 = sx * k
    sy2 = sy * k
    tilt_x_used = float(np.rad2deg(np.arctan(sx2)))
    tilt_y_used = float(np.rad2deg(np.arctan(sy2)))
    offset2 = tilt_center_offset_um(tilt_x_used, tilt_y_used, fov_um)

    # Rounding guard: make sure the rounded pre_surface_planes is not negative.
    pre_surface_planes = int(round((hydrogel_height_um - offset2) / dz_um))
    if pre_surface_planes < 0:
        # Ensure offset2 is at least dz/2 below the target to avoid rounding to -1.
        target_offset2 = max(0.0, hydrogel_height_um - 0.5 * dz_um)
        k2 = min(1.0, max(0.0, (target_offset2 - 1e-6) / float(offset)))
        sx2 = sx * k2
        sy2 = sy * k2
        tilt_x_used = float(np.rad2deg(np.arctan(sx2)))
        tilt_y_used = float(np.rad2deg(np.arctan(sy2)))
        offset2 = tilt_center_offset_um(tilt_x_used, tilt_y_used, fov_um)

    return tilt_x_used, tilt_y_used, float(offset2)


def generate_single_sample(
    folder_path: Path,
    sample_id: int,
    pitch_um: float,
    depth_um: float,
    hydrogel_height_um: float,
    groove_angle: float,
    tilt_x: float,
    tilt_y: float,
    scenario_name: str,
    noise_mult: float,
    imaging: ImagingConfig,
    seed: int,
    rng: np.random.Generator,
) -> SyntheticSample:
    """
    Generate single synthetic sample: height map → confocal z-stack → TIFF + truth NPZ.
    Returns SyntheticSample with realized ground truth for validation CSV.
    """
    groove = GrooveConfig(
        pitch_um=pitch_um,
        depth_um=depth_um,
        groove_angle_deg=groove_angle,
    )

    # Ensure tilt does not make the requested hydrogel height impossible.
    tilt_x_used, tilt_y_used, tilt_center_offset = check_and_adjust_tilt(
        hydrogel_height_um=hydrogel_height_um,
        tilt_x_deg=tilt_x,
        tilt_y_deg=tilt_y,
        fov_um=imaging.fov_um,
        dz_um=imaging.dz_um,
    )
    if (tilt_x_used != tilt_x) or (tilt_y_used != tilt_y):
        LOG.warning(
            "Adjusted surface tilt for feasibility: "
            f"(tilt_x, tilt_y)=({tilt_x:.4f}°, {tilt_y:.4f}°) -> "
            f"({tilt_x_used:.4f}°, {tilt_y_used:.4f}°) "
            f"for hydrogel_height={hydrogel_height_um:.2f} µm "
            f"(FOV={imaging.fov_um:.1f} µm)"
        )

    # Clean scenario: low roughness to avoid perfectly flat columns
    roughness = 0.02 if scenario_name == "clean" else 0.15
    surface = SurfaceConfig(
        tilt_x_deg=tilt_x_used,
        tilt_y_deg=tilt_y_used,
        roughness_rms_um=roughness,
    )

    surface_gen = SyntheticGrooveSurface(imaging, groove, surface, rng)
    height_map = surface_gen.generate_height_map()

    # Anchor: groove bottoms start at 0 µm (z=0 reference plane)
    height_map = (height_map - float(np.nanmin(height_map))).astype(np.float32)

    # Choose how many pre-surface planes to prepend so the hydrogel height
    # (tilt-corrected mean plane height) is defined a priori by the generator.
    # tilt_center_offset already computed after feasibility adjustment
    dz = float(imaging.dz_um)
    pre_surface_planes = int(round((hydrogel_height_um - tilt_center_offset) / dz))
    if pre_surface_planes < 0:
        pre_surface_planes = 0

    imaging_adj = ImagingConfig(**asdict(imaging))
    if noise_mult != 1.0:
        photons = imaging.poisson_peak_photons / noise_mult
        imaging_adj.poisson_peak_photons = max(1, int(photons))
        imaging_adj.background_frac = float(imaging.background_frac) * noise_mult

    simulator = ConfocalSimulator(imaging_adj, rng)
    volume, z_min, z_max = simulator.simulate_zstack(
        height_map, pre_surface_planes=pre_surface_planes
    )

    stack_code = f"S{sample_id:05d}"
    file_path = folder_path / f"{stack_code}.tif"
    tifffile.imwrite(str(file_path), volume, photometric="minisblack")

    np.savez_compressed(
        folder_path / f"{stack_code}_truth_height_um.npz",
        height_um=height_map,
        height_rel_um=(height_map - z_min).astype(np.float32),
    )

    stats = surface_gen.get_realized_stats()

    return SyntheticSample(
        stack_code=stack_code,
        file_path=file_path,
        folder_path=folder_path,
        pitch_nom_um=pitch_um,
        depth_nom_um=depth_um,
        groove_angle_deg=groove_angle,
        tilt_x_deg=tilt_x_used,
        tilt_y_deg=tilt_y_used,
        pitch_realized_mean_um=stats["pitch_realized_mean_um"],
        pitch_realized_std_um=stats["pitch_realized_std_um"],
        depth_realized_mean_um=stats["depth_realized_mean_um"],
        depth_realized_std_um=stats["depth_realized_std_um"],
        n_periods=stats["n_periods"],
        z_min_um=z_min,
        z_max_um=z_max,
        hydrogel_height_um=float(hydrogel_height_um),
        scenario=scenario_name,
        seed=seed,
    )


def write_truth_index(outdir: Path, samples: list[SyntheticSample]) -> None:
    """
    Write ground truth CSV with field names matching validator expectations.
    Critical fields: pitch/depth_realized_mean_um, groove_angle_deg, etc.
    """
    rows: list[dict[str, object]] = []
    for s in samples:
        rows.append(
            {
                "stack_code": s.stack_code,
                "file_rel_path": str(s.file_path.relative_to(outdir)),
                "folder_rel_path": str(s.folder_path.relative_to(outdir)),
                "pitch_nom_um": s.pitch_nom_um,
                "depth_nom_um": s.depth_nom_um,
                "groove_angle_deg": s.groove_angle_deg,
                "tilt_x_deg": s.tilt_x_deg,
                "tilt_y_deg": s.tilt_y_deg,
                "pitch_realized_mean_um": s.pitch_realized_mean_um,
                "pitch_realized_std_um": s.pitch_realized_std_um,
                "depth_realized_mean_um": s.depth_realized_mean_um,
                "depth_realized_std_um": s.depth_realized_std_um,
                "n_periods": s.n_periods,
                "z_min_um": s.z_min_um,
                "z_max_um": s.z_max_um,
                "hydrogel_height_um": s.hydrogel_height_um,
                "scenario": s.scenario,
                "seed": s.seed,
            }
        )

    out_csv = outdir / "truth_index.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_dataset_metadata(outdir: Path, imaging: ImagingConfig) -> None:
    """Write dataset metadata JSON for validator calibration extraction."""
    metadata = {
        "xy_um_per_px": float(imaging.xy_um),
        "dz_um_per_slice": float(imaging.dz_um),
        "z0_um_offset": 0.0,
        "fov_um": float(imaging.fov_um),
        "nz_max": int(imaging.nz_max),
    }
    out_json = outdir / "dataset_metadata.json"
    out_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


# ============================================================================
# Entry point
# ============================================================================


def main_cli(output_dir: str | None = None, seed: int = 42) -> None:
    """CLI entry point: generate synthetic dataset in specified output directory."""
    if output_dir is None:
        script_dir = Path(__file__).resolve().parent
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = str(script_dir / f"groove_validation_{timestamp}")

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    imaging = ImagingConfig()
    config = ValidationConfig()

    LOG.info(f"Generating dataset in: {outdir}")
    generate_validation_dataset(outdir, config, imaging, seed=seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic groove dataset generator with physics-based confocal simulation"
        )
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        help=(
            "Output directory for dataset "
            "(default: <script-dir>/groove_validation_<timestamp>)"
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    main_cli(output_dir=args.output_dir, seed=args.seed)
