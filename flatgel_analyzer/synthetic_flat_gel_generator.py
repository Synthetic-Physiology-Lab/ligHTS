"""
Synthetic Flat Gel Surface Generator for Validation.

Generates synthetic confocal Z-stacks of flat/tilted gel surfaces with
realistic noise to validate the gel surface analysis pipeline.

Features:
  - Rounded-rectangle well geometry matching real microwell plates
  - Planar tilt with optional roughness
  - Confocal PSF convolution with realistic signal dropout (~30%)
  - Shot noise, read noise, and background fluorescence
  - 12-bit quantization

Ground truth includes:
  - Tilt angles (tilt_x_deg, tilt_y_deg) up to ±4°
  - Mean gel height (constrained to produce 50-100 µm height span)
  - Surface roughness parameters

Output files:
  - S#####.tif: Z-stack TIFF files with synthetic confocal images
  - S#####_truth.npz: Detailed ground truth (height maps, parameters)
  - truth_metrics.csv: Ground truth using analyzer-compatible column names
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    from scipy.ndimage import gaussian_filter
except ImportError as e:
    raise RuntimeError("scipy is required: pip install scipy") from e

try:
    import tifffile
except ImportError as e:
    raise RuntimeError("tifffile is required: pip install tifffile") from e


LOG = logging.getLogger("synth_flat_gel")
LOG.setLevel(logging.INFO)
if not LOG.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    LOG.addHandler(handler)


# =============================================================================
# Configuration Classes
# =============================================================================


@dataclass
class ImagingConfig:
    """Confocal microscopy imaging parameters."""

    xy_um: float = 5.36
    dz_um: float = 1.4
    fov_um: float = 4000.0
    nz_max: int = 120
    poisson_peak_photons: int = 800
    background_frac: float = 0.06
    read_noise_electrons: float = 4.0
    bit_depth: int = 12
    psf_fwhm_xy_um: float = 2.0
    psf_fwhm_z_um: float = 3.5
    signal_dropout_frac: float = 0.30


@dataclass
class WellConfig:
    """Microwell geometry configuration."""

    well_edge_mm: float = 3.3
    corner_radius_frac: float = 0.10


@dataclass
class SurfaceConfig:
    """Gel surface configuration."""

    tilt_x_deg: float = 0.0
    tilt_y_deg: float = 0.0
    gel_height_um: float = 50.0
    roughness_rms_um: float = 0.3
    roughness_corr_len_um: float = 8.0


@dataclass
class ValidationConfig:
    """Validation dataset generation parameters."""

    tilt_max_deg: float = 4.0
    height_span_min_um: float = 50.0
    height_span_max_um: float = 100.0
    gel_heights_um: tuple[float, ...] = (50.0, 75.0, 90.0)
    n_samples_per_height: int = 4


@dataclass
class SyntheticSample:
    """Ground truth for a single synthetic sample."""

    stack_code: str
    file_path: Path
    tilt_x_deg: float
    tilt_y_deg: float
    gel_height_um: float
    roughness_rms_um: float
    tilt_angle_deg: float
    z_min_um: float
    z_max_um: float
    height_span_um: float
    xy_um: float
    dz_um: float
    nz: int
    scenario: str
    seed: int
    noise_multiplier: float


# =============================================================================
# Well Mask Creation
# =============================================================================


def create_well_mask(ny: int, nx: int, xy_um: float, fov_um: float, well: WellConfig) -> np.ndarray:
    """
    Create rounded-rectangle well mask matching real microwells.

    Generates binary mask with rounded corners for realistic geometry.
    Returns uint8 array where 1=inside well, 0=outside.
    """
    well_edge_um = well.well_edge_mm * 1000.0
    corner_radius_um = well_edge_um * well.corner_radius_frac

    cx_um = fov_um / 2.0
    cy_um = fov_um / 2.0
    half_edge = well_edge_um / 2.0
    inner_half = half_edge - corner_radius_um

    x = np.arange(nx, dtype=np.float32) * xy_um
    y = np.arange(ny, dtype=np.float32) * xy_um
    xx, yy = np.meshgrid(x, y, indexing="xy")

    dx = np.abs(xx - cx_um)
    dy = np.abs(yy - cy_um)

    in_rect = (dx <= half_edge) & (dy <= half_edge)
    in_inner = (dx <= inner_half) | (dy <= inner_half)
    corner_dist = np.sqrt(np.maximum(0, dx - inner_half) ** 2 + np.maximum(0, dy - inner_half) ** 2)
    in_corners = corner_dist <= corner_radius_um

    mask = (in_rect & (in_inner | in_corners)).astype(np.uint8)

    del xx, yy, dx, dy, in_rect, in_inner, corner_dist, in_corners
    gc.collect()

    return mask


# =============================================================================
# Height Map Generation
# =============================================================================


def generate_height_map(
    ny: int,
    nx: int,
    xy_um: float,
    fov_um: float,
    well_mask: np.ndarray,
    surface: SurfaceConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate height map with tilt and roughness.

    Creates planar surface with specified tilt angles, adds correlated
    roughness noise, and applies well mask. Returns float32 array in µm.
    """
    height = np.full((ny, nx), surface.gel_height_um, dtype=np.float32)

    cx = fov_um / 2.0
    cy = fov_um / 2.0
    tilt_x_slope = np.tan(np.deg2rad(surface.tilt_x_deg))
    tilt_y_slope = np.tan(np.deg2rad(surface.tilt_y_deg))

    x = np.arange(nx, dtype=np.float32) * xy_um
    y = np.arange(ny, dtype=np.float32) * xy_um
    xx, yy = np.meshgrid(x, y, indexing="xy")

    height += tilt_x_slope * (xx - cx) + tilt_y_slope * (yy - cy)
    del xx, yy

    if surface.roughness_rms_um > 0:
        noise = rng.normal(0.0, 1.0, (ny, nx)).astype(np.float32)
        sigma_px = surface.roughness_corr_len_um / xy_um

        if sigma_px > 0.5:
            noise = gaussian_filter(noise, sigma_px)
            noise = noise / (np.std(noise) + 1e-10) * surface.roughness_rms_um
        else:
            noise *= surface.roughness_rms_um

        height += noise
        del noise

    height[well_mask == 0] = np.nan
    gc.collect()

    return height


# =============================================================================
# Z-Stack Simulation
# =============================================================================


def simulate_zstack(
    height_map: np.ndarray, imaging: ImagingConfig, rng: np.random.Generator
) -> tuple[np.ndarray, float, float]:
    """
    Generate confocal Z-stack from height map.

    Simulates confocal imaging with PSF, signal dropout (~30%), shot noise,
    read noise, and quantization. Returns (volume, z_min, z_max).
    """
    ny, nx = height_map.shape
    h_valid = height_map[np.isfinite(height_map)]

    if h_valid.size == 0:
        return np.zeros((16, ny, nx), dtype=np.uint16), 0.0, 15.0 * imaging.dz_um

    margin_below = 8.0
    margin_above = 12.0
    z_min = float(np.nanmin(h_valid)) - margin_below
    z_max = float(np.nanmax(h_valid)) + margin_above

    nz = int(np.ceil((z_max - z_min) / imaging.dz_um)) + 1
    nz = max(16, min(nz, imaging.nz_max))
    z_max = z_min + (nz - 1) * imaging.dz_um

    sigma_xy_um, sigma_z_um = compute_psf_parameters(imaging)
    sigma_xy_px = sigma_xy_um / imaging.xy_um
    sigma_z_px = sigma_z_um / imaging.dz_um

    volume = render_surface_to_volume(
        height_map, z_min, imaging.dz_um, nz, ny, nx, imaging.signal_dropout_frac, rng
    )

    volume = gaussian_filter(volume, sigma=[sigma_z_px, sigma_xy_px, sigma_xy_px])

    volume = apply_well_mask(volume, height_map)
    volume = apply_noise_model(volume, imaging, rng)
    volume = quantize_to_bits(volume, imaging.bit_depth)

    gc.collect()

    return volume, z_min, z_max


def compute_psf_parameters(imaging: ImagingConfig) -> tuple[float, float]:
    """
    Compute PSF sigma parameters from FWHM.

    Accounts for pixel integration effects.
    Returns (sigma_xy_um, sigma_z_um).
    """
    sigma_xy_um = imaging.psf_fwhm_xy_um / 2.355
    sigma_z_um = imaging.psf_fwhm_z_um / 2.355

    sigma_xy_um = np.sqrt(sigma_xy_um**2 + (imaging.xy_um / np.sqrt(12)) ** 2)
    sigma_z_um = np.sqrt(sigma_z_um**2 + (imaging.dz_um / np.sqrt(12)) ** 2)

    return sigma_xy_um, sigma_z_um


def render_surface_to_volume(
    height_map: np.ndarray,
    z_min: float,
    dz_um: float,
    nz: int,
    ny: int,
    nx: int,
    dropout_frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Render surface height map to 3D volume with signal dropout.

    Creates spatially correlated dropout mask (~30% of surface) to simulate
    realistic bead coverage. Returns float32 volume array.
    """
    volume = np.zeros((nz, ny, nx), dtype=np.float32)
    z_idx = (height_map - z_min) / dz_um

    z_int = np.zeros_like(z_idx, dtype=np.int32)
    finite_mask = np.isfinite(z_idx)
    z_int[finite_mask] = np.round(z_idx[finite_mask]).astype(np.int32)

    valid = finite_mask & (z_int >= 0) & (z_int < nz)

    dropout_noise = rng.random((ny, nx)).astype(np.float32)
    dropout_noise = gaussian_filter(dropout_noise, sigma=1.5)
    dropout_mask = dropout_noise > dropout_frac

    has_signal = valid & dropout_mask

    yy, xx = np.where(has_signal)
    intensities = rng.uniform(0.7, 1.0, len(yy)).astype(np.float32)
    volume[z_int[yy, xx], yy, xx] = intensities

    del z_idx, z_int, valid, dropout_noise, dropout_mask, has_signal
    del yy, xx, intensities, finite_mask
    gc.collect()

    return volume


def apply_well_mask(volume: np.ndarray, height_map: np.ndarray) -> np.ndarray:
    """
    Apply well mask to volume.

    Sets all voxels outside well to zero.
    Returns modified volume.
    """
    mask_2d = np.isfinite(height_map)
    volume[:, ~mask_2d] = 0.0
    return volume


def apply_noise_model(
    volume: np.ndarray, imaging: ImagingConfig, rng: np.random.Generator
) -> np.ndarray:
    """
    Apply realistic noise model to volume.

    Adds shot noise (Poisson) and read noise (Gaussian).
    Returns normalized volume array.
    """
    vmax = float(np.max(volume))

    if vmax > 0:
        volume = volume / vmax + imaging.background_frac
        peak = max(1, imaging.poisson_peak_photons)
        volume = volume * peak
        volume = rng.poisson(lam=np.clip(volume, 0, None)).astype(np.float32)
        volume += rng.normal(0, imaging.read_noise_electrons, volume.shape).astype(np.float32)
        volume = np.clip(volume / peak, 0, 1)

    return volume


def quantize_to_bits(volume: np.ndarray, bit_depth: int) -> np.ndarray:
    """
    Quantize volume to specified bit depth.

    Converts normalized float values to integer values.
    Returns uint16 array.
    """
    max_val = (1 << bit_depth) - 1
    volume = np.round(volume * max_val).astype(np.uint16)
    return volume


# =============================================================================
# Tilt Feasibility and Adjustment
# =============================================================================


def compute_height_span_for_tilt(
    tilt_x_deg: float, tilt_y_deg: float, well_diagonal_um: float
) -> float:
    """
    Compute height span (max - min) for given tilt across well.

    Maximum height difference occurs along diagonal in direction of max tilt.
    Returns height span in µm.
    """
    sx = np.tan(np.deg2rad(tilt_x_deg))
    sy = np.tan(np.deg2rad(tilt_y_deg))
    slope_mag = np.sqrt(sx**2 + sy**2)
    return slope_mag * well_diagonal_um


def compute_tilt_for_height_span(
    target_span_um: float,
    well_diagonal_um: float,
    tilt_x_deg: float,
    tilt_y_deg: float,
) -> tuple[float, float]:
    """
    Scale tilt angles to achieve target height span.

    Preserves tilt direction (ratio of x to y) while scaling magnitude.
    Returns (adjusted_tilt_x_deg, adjusted_tilt_y_deg).
    """
    current_span = compute_height_span_for_tilt(tilt_x_deg, tilt_y_deg, well_diagonal_um)

    if current_span < 1e-6:
        return tilt_x_deg, tilt_y_deg

    scale = target_span_um / current_span

    sx = np.tan(np.deg2rad(tilt_x_deg)) * scale
    sy = np.tan(np.deg2rad(tilt_y_deg)) * scale

    return np.rad2deg(np.arctan(sx)), np.rad2deg(np.arctan(sy))


def check_and_adjust_tilt(
    tilt_x_deg: float,
    tilt_y_deg: float,
    well_edge_mm: float,
    config: ValidationConfig,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """
    Ensure tilt produces height span within [50, 100] µm.

    Scales tilt to fit bounds if needed while preserving direction.
    Returns (adjusted_tilt_x, adjusted_tilt_y, actual_span).
    """
    well_diagonal_um = well_edge_mm * 1000.0 * np.sqrt(2)

    current_span = compute_height_span_for_tilt(tilt_x_deg, tilt_y_deg, well_diagonal_um)

    if config.height_span_min_um <= current_span <= config.height_span_max_um:
        return tilt_x_deg, tilt_y_deg, current_span

    if current_span > config.height_span_max_um:
        target = rng.uniform(config.height_span_min_um, config.height_span_max_um)
        tx_new, ty_new = compute_tilt_for_height_span(
            target, well_diagonal_um, tilt_x_deg, tilt_y_deg
        )
        return tx_new, ty_new, target

    elif current_span < config.height_span_min_um:
        target = rng.uniform(
            config.height_span_min_um,
            min(config.height_span_max_um, config.height_span_min_um * 1.5),
        )
        tx_new, ty_new = compute_tilt_for_height_span(
            target, well_diagonal_um, tilt_x_deg, tilt_y_deg
        )
        return tx_new, ty_new, target

    return tilt_x_deg, tilt_y_deg, current_span


# =============================================================================
# Single Sample Generation
# =============================================================================


def generate_single_sample(
    output_dir: Path,
    sample_id: int,
    tilt_x_deg: float,
    tilt_y_deg: float,
    gel_height_um: float,
    roughness_rms_um: float,
    scenario: str,
    noise_mult: float,
    imaging: ImagingConfig,
    well: WellConfig,
    config: ValidationConfig,
    seed: int,
    rng: np.random.Generator,
) -> SyntheticSample:
    """
    Generate single synthetic sample with ground truth.

    Creates TIFF stack and NPZ truth file. Adjusts tilt to ensure height span
    is within [50, 100] µm. Returns SyntheticSample with metadata.
    """
    tilt_x, tilt_y, expected_span = check_and_adjust_tilt(
        tilt_x_deg, tilt_y_deg, well.well_edge_mm, config, rng
    )

    if abs(tilt_x - tilt_x_deg) > 0.01 or abs(tilt_y - tilt_y_deg) > 0.01:
        LOG.info(
            "  Adjusted tilt: (%.2f°, %.2f°) -> (%.2f°, %.2f°) " "[target span: %.1f µm]",
            tilt_x_deg,
            tilt_y_deg,
            tilt_x,
            tilt_y,
            expected_span,
        )

    nx = int(np.ceil(imaging.fov_um / imaging.xy_um))
    ny = nx

    LOG.info("  Creating %dx%d arrays...", ny, nx)
    well_mask = create_well_mask(ny, nx, imaging.xy_um, imaging.fov_um, well)

    surface = SurfaceConfig(
        tilt_x_deg=tilt_x,
        tilt_y_deg=tilt_y,
        gel_height_um=gel_height_um,
        roughness_rms_um=roughness_rms_um,
    )

    height_map = generate_height_map(ny, nx, imaging.xy_um, imaging.fov_um, well_mask, surface, rng)

    gel_height_truth = compute_gel_height_truth(
        height_map, tilt_x, tilt_y, imaging.fov_um, imaging.xy_um, well_mask, nx, ny
    )

    h_valid = height_map[np.isfinite(height_map)]
    height_span = float(np.ptp(h_valid)) if h_valid.size > 0 else 0.0

    img_adj = adjust_imaging_for_noise(imaging, noise_mult)

    LOG.info("  Simulating Z-stack (height span: %.1f µm)...", height_span)
    volume, z_min, z_max = simulate_zstack(height_map, img_adj, rng)
    nz = volume.shape[0]

    stack_code = f"S{sample_id:05d}"
    tif_path = output_dir / f"{stack_code}.tif"

    save_tiff_stack(tif_path, volume, imaging)
    save_truth_npz(
        output_dir,
        stack_code,
        height_map,
        well_mask,
        tilt_x,
        tilt_y,
        gel_height_truth,
        roughness_rms_um,
        z_min,
        z_max,
        imaging,
    )

    del volume, height_map, well_mask
    gc.collect()

    tilt_angle = compute_combined_tilt_angle(tilt_x, tilt_y)

    return SyntheticSample(
        stack_code=stack_code,
        file_path=tif_path,
        tilt_x_deg=tilt_x,
        tilt_y_deg=tilt_y,
        gel_height_um=gel_height_truth,
        roughness_rms_um=roughness_rms_um,
        tilt_angle_deg=tilt_angle,
        z_min_um=z_min,
        z_max_um=z_max,
        height_span_um=height_span,
        scenario=scenario,
        seed=seed,
        noise_multiplier=noise_mult,
        xy_um=imaging.xy_um,
        dz_um=imaging.dz_um,
        nz=nz,
    )


def compute_gel_height_truth(
    height_map: np.ndarray,
    tilt_x: float,
    tilt_y: float,
    fov_um: float,
    xy_um: float,
    well_mask: np.ndarray,
    nx: int,
    ny: int,
) -> float:
    """
    Compute true mean gel height by removing tilt component.

    Subtracts planar tilt from height map to get base height.
    Returns mean height in µm.
    """
    x = np.arange(nx, dtype=np.float32) * xy_um
    y = np.arange(ny, dtype=np.float32) * xy_um
    xx, yy = np.meshgrid(x, y, indexing="xy")

    cx = fov_um / 2.0
    cy = fov_um / 2.0
    tilt_x_slope = np.tan(np.deg2rad(tilt_x))
    tilt_y_slope = np.tan(np.deg2rad(tilt_y))

    tilt_plane = tilt_x_slope * (xx - cx) + tilt_y_slope * (yy - cy)
    pretilt_height = height_map - tilt_plane

    well_mask_bool = well_mask.astype(bool)
    gel_height_truth = float(np.nanmean(pretilt_height[well_mask_bool]))

    del xx, yy, tilt_plane, pretilt_height, x, y

    return gel_height_truth


def adjust_imaging_for_noise(imaging: ImagingConfig, noise_mult: float) -> ImagingConfig:
    """
    Adjust imaging parameters for noise multiplier.

    Scales photon count and background fraction.
    Returns new ImagingConfig.
    """
    img_adj = ImagingConfig(**asdict(imaging))

    if noise_mult != 1.0:
        img_adj.poisson_peak_photons = max(100, int(imaging.poisson_peak_photons / noise_mult))
        img_adj.background_frac = imaging.background_frac * noise_mult

    return img_adj


def compute_combined_tilt_angle(tilt_x_deg: float, tilt_y_deg: float) -> float:
    """
    Compute combined tilt angle magnitude from x and y components.

    Returns angle in degrees.
    """
    return np.rad2deg(
        np.arctan(
            np.sqrt(np.tan(np.deg2rad(tilt_x_deg)) ** 2 + np.tan(np.deg2rad(tilt_y_deg)) ** 2)
        )
    )


def save_tiff_stack(tif_path: Path, volume: np.ndarray, imaging: ImagingConfig) -> None:
    """
    Save Z-stack to TIFF file with OME metadata.

    Includes physical size information for proper scaling.
    """
    LOG.info("  Saving %s (%s)...", tif_path.name, volume.shape)

    tifffile.imwrite(
        str(tif_path),
        volume,
        photometric="minisblack",
        ome=True,
        metadata={
            "axes": "ZYX",
            "PhysicalSizeX": imaging.xy_um,
            "PhysicalSizeY": imaging.xy_um,
            "PhysicalSizeZ": imaging.dz_um,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeYUnit": "µm",
            "PhysicalSizeZUnit": "µm",
        },
    )


def save_truth_npz(
    output_dir: Path,
    stack_code: str,
    height_map: np.ndarray,
    well_mask: np.ndarray,
    tilt_x: float,
    tilt_y: float,
    gel_height: float,
    roughness: float,
    z_min: float,
    z_max: float,
    imaging: ImagingConfig,
) -> None:
    """
    Save detailed ground truth to compressed NPZ file.

    Includes height map, masks, and all generation parameters.
    """
    np.savez_compressed(
        output_dir / f"{stack_code}_truth.npz",
        height_um=height_map.astype(np.float32),
        well_mask=well_mask,
        tilt_x_deg=np.float32(tilt_x),
        tilt_y_deg=np.float32(tilt_y),
        gel_height_um=np.float32(gel_height),
        roughness_rms_um=np.float32(roughness),
        z_min_um=np.float32(z_min),
        z_max_um=np.float32(z_max),
        xy_um=np.float32(imaging.xy_um),
        dz_um=np.float32(imaging.dz_um),
    )


# =============================================================================
# Truth Metrics Output
# =============================================================================


def write_truth_metrics(output_dir: Path, samples: list[SyntheticSample]) -> None:
    """
    Write ground truth CSV using analyzer-compatible column names.

    Converts tilt angles to slopes (µm/mm) for direct comparison with
    analyzer outputs. Creates truth_metrics.csv in output directory.
    """
    rows = []
    for sample in samples:
        tilt_x_slope = np.tan(np.deg2rad(sample.tilt_x_deg)) * 1000.0
        tilt_y_slope = np.tan(np.deg2rad(sample.tilt_y_deg)) * 1000.0

        rows.append(
            {
                "File": sample.file_path.name,
                "XY_um_per_px": round(sample.xy_um, 4),
                "Z_step_um": round(sample.dz_um, 4),
                "Z0_um": round(sample.z_min_um, 2),
                "Plane_a_um_per_mm": round(tilt_x_slope, 4),
                "Plane_b_um_per_mm": round(tilt_y_slope, 4),
                "TiltAngle_deg": round(sample.tilt_angle_deg, 4),
                "TopZ_LFilteredMean_um": round(sample.gel_height_um, 2),
                "scenario": sample.scenario,
                "seed": sample.seed,
                "noise_multiplier": sample.noise_multiplier,
            }
        )

    csv_path = output_dir / "truth_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    LOG.info("Wrote: %s", csv_path)


# =============================================================================
# Dataset Generation
# =============================================================================


def generate_validation_dataset(
    output_dir: Path,
    config: ValidationConfig,
    imaging: ImagingConfig,
    well: WellConfig,
    seed: int = 42,
) -> list[SyntheticSample]:
    """
    Generate complete validation dataset.

    Creates samples across 4 scenarios (clean, tilted, noisy, combined)
    with multiple gel heights. All samples constrained to 50-100 µm span.
    Returns list of SyntheticSample objects.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    rng_master = np.random.default_rng(seed)
    samples: list[SyntheticSample] = []
    sample_id = 0

    total_samples = len(config.gel_heights_um) * config.n_samples_per_height * 4

    log_dataset_info(total_samples, imaging, config)

    for gel_height in config.gel_heights_um:
        samples, sample_id = generate_scenario_samples(
            "clean",
            gel_height,
            samples,
            sample_id,
            total_samples,
            output_dir,
            imaging,
            well,
            config,
            rng_master,
            tilt_range=(-0.8, 0.8),
            roughness=0.15,
            noise_mult=1.0,
        )

        samples, sample_id = generate_scenario_samples(
            "tilted",
            gel_height,
            samples,
            sample_id,
            total_samples,
            output_dir,
            imaging,
            well,
            config,
            rng_master,
            tilt_range=(-config.tilt_max_deg, config.tilt_max_deg),
            roughness=0.25,
            noise_mult=1.0,
        )

        samples, sample_id = generate_scenario_samples(
            "noisy",
            gel_height,
            samples,
            sample_id,
            total_samples,
            output_dir,
            imaging,
            well,
            config,
            rng_master,
            tilt_range=(-2.5, 2.5),
            roughness=0.25,
            noise_mult=2.5,
        )

        samples, sample_id = generate_scenario_samples(
            "combined",
            gel_height,
            samples,
            sample_id,
            total_samples,
            output_dir,
            imaging,
            well,
            config,
            rng_master,
            tilt_range=(-config.tilt_max_deg, config.tilt_max_deg),
            roughness=None,
            noise_mult=1.5,
        )

    write_truth_metrics(output_dir, samples)
    LOG.info("Complete: %d samples in %s", len(samples), output_dir)

    return samples


def log_dataset_info(total: int, imaging: ImagingConfig, config: ValidationConfig) -> None:
    """Log dataset generation parameters."""
    LOG.info("Generating %d synthetic samples...", total)
    LOG.info(
        "  Image size: %dx%d px",
        int(imaging.fov_um / imaging.xy_um),
        int(imaging.fov_um / imaging.xy_um),
    )
    LOG.info("  Pixel size: %.2f µm, Z-step: %.2f µm", imaging.xy_um, imaging.dz_um)
    LOG.info(
        "  Target height span: %.1f-%.1f µm",
        config.height_span_min_um,
        config.height_span_max_um,
    )
    LOG.info("  Signal dropout: %.0f%%", imaging.signal_dropout_frac * 100)


def generate_scenario_samples(
    scenario: str,
    gel_height: float,
    samples: list[SyntheticSample],
    sample_id: int,
    total_samples: int,
    output_dir: Path,
    imaging: ImagingConfig,
    well: WellConfig,
    config: ValidationConfig,
    rng_master: np.random.Generator,
    tilt_range: tuple[float, float],
    roughness: float | None,
    noise_mult: float,
) -> tuple[list[SyntheticSample], int]:
    """
    Generate samples for one scenario.

    Creates n_samples_per_height samples with specified parameters.
    Returns (updated_samples_list, next_sample_id).
    """
    for _ in range(config.n_samples_per_height):
        seed_val = int(rng_master.integers(0, 2**31))
        rng = np.random.default_rng(seed_val)

        tx = float(rng.uniform(*tilt_range))
        ty = float(rng.uniform(*tilt_range))

        if roughness is None:
            rough = float(rng.uniform(0.1, 0.5))
        else:
            rough = roughness

        LOG.info(
            "[%d/%d] %s, gel_height=%.1f µm, tilt=(%.2f°, %.2f°)",
            sample_id + 1,
            total_samples,
            scenario,
            gel_height,
            tx,
            ty,
        )

        sample = generate_single_sample(
            output_dir,
            sample_id,
            tx,
            ty,
            gel_height,
            rough,
            scenario,
            noise_mult,
            imaging,
            well,
            config,
            seed_val,
            rng,
        )

        samples.append(sample)
        sample_id += 1

    return samples, sample_id


# =============================================================================
# CLI Entry Point
# =============================================================================


def main_cli(
    output_dir: str | None = None,
    n_samples: int | None = None,
    fov_um: float | None = None,
    xy_um: float | None = None,
    dz_um: float | None = None,
    seed: int = 42,
) -> None:
    """
    Generate validation dataset from command-line arguments.

    Creates synthetic gel stacks with specified parameters.
    """
    if output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"flat_gel_validation_{timestamp}"

    outdir = Path(output_dir)

    imaging = ImagingConfig()
    if fov_um is not None:
        imaging.fov_um = fov_um
    if xy_um is not None:
        imaging.xy_um = xy_um
    if dz_um is not None:
        imaging.dz_um = dz_um

    well = WellConfig()
    config = ValidationConfig()

    if n_samples is not None:
        config.n_samples_per_height = max(1, n_samples)

    LOG.info("Output directory: %s", outdir)
    generate_validation_dataset(outdir, config, imaging, well, seed=seed)


def main() -> None:
    """Parse command-line arguments and generate dataset."""
    parser = argparse.ArgumentParser(
        description="Synthetic flat gel surface generator for validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python synthetic_flat_gel_generator.py -o ./data -n 10
  python synthetic_flat_gel_generator.py -o ./data --fov 3500 --xy 5.0

Output:
  - S00000.tif, S00001.tif, ... : Z-stack TIFF files
  - S00000_truth.npz, ... : Ground truth (height map, well mask, parameters)
  - truth_metrics.csv : Ground truth metrics using analyzer-compatible names

All samples constrained to 50-100 µm height span.
Signal includes ~30% dropout to simulate realistic bead coverage.
        """,
    )

    parser.add_argument("--output-dir", "-o", type=str, help="Output directory")
    parser.add_argument(
        "--n-samples",
        "-n",
        type=int,
        default=4,
        help="Number of samples per scenario per height",
    )
    parser.add_argument("--fov", type=float, dest="fov_um", help="Field of view (µm)")
    parser.add_argument("--xy", type=float, dest="xy_um", help="XY pixel size (µm)")
    parser.add_argument("--dz", type=float, dest="dz_um", help="Z step size (µm)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    main_cli(
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        fov_um=args.fov_um,
        xy_um=args.xy_um,
        dz_um=args.dz_um,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
