# Gel Surface Analysis

Pipeline to estimate gel surface height from fluorescence z‑stacks.
Designed for high-SNR spinning‑disk confocal images of fluorescent beads on soft gels.


## Key features
- Reads `.tif/.tiff/.nd2` stacks as **Z×Y×X** float32 volumes.
  - ND2 reading uses `nd2` when available; falls back to `aicsimageio` if installed.
  - If additional axes exist (T/C/P/S), the analyzer deterministically selects a single index per axis (default 0).
- Subpixel axial localization by 3-point quadratic refinement around the per-pixel argmax.
- Optional well segmentation from Max-Z projection (rounded-rectangle ROI) with inward scaling (`MaskScaleFrac`) to suppress rim artifacts; exports overlays for audit.
- Within-FOV tilt removal by least-squares plane fit on a conservative core ROI (exports plane coefficients and R²).
- Explicit QC and truncation handling:
  - Reports valid pixel fractions and `TruncMode` (likely z-truncation at top/bottom).
  - If truncated, thickness/uniformity metrics are returned as NaN and QC/diagnostic images are exported.


Notes
----------

- Well segmentation assumes a rounded rectangle with high contrast on the MIP.
- ND2 reading depends on installed backends; see 'requirements.txt' for versions.
- The analyzer script is verified (100% pass summarized by the validator script) against synthetic ground truth
  (synthetic dataset produced by the generator script from a known forward model).
- This code is intended for the sole use of analyzing the datasets and extracting the hydrogel
  heights displayed in figure 2 c-f-g-h of the manuscript "LigHTS: Massively Parallel Biomimetic
  Photo-Functionalization for Imaging-Based Ultra-High-Throughput Screening" 

## Install

Using pip:
```bash

# Recommended:  install into a clean Python 3.11 virtual environment (venv or conda)
pip install -r requirements.txt

```
Using Conda:

```bash
conda env create -f environment.yml
conda activate gel-surface-analysis

```


## Basic usage

To run the script and select via GUI a folder containing stacks or subfolders of stacks for gel height extractions: 
```bash
python flat_gel_surface_analyzer.py
```

## Advanced Options

Run headless over subfolders of stacks:
```bash
python flat_gel_surface_analyzer.py --root /path/to/data --no-gui
```
Manual scale overrides:
```bash
python flat_gel_surface_analyzer.py --root data --xy 0.325 --dz 0.30 --z0 0.0 --no-gui
```
Filter and analysis controls:
```bash
--bead-um <float>           		 # Nominal bead diameter [µm]
--lambda-c-um <float>       		 # L-filter cutoff λc [µm] (default: 120.0)
--lambda-s-um <float>        		 # S-filter cutoff λs [µm] override
--mask-scale <float>         		 # ROI inward-scaling factor 0.0-1.0 (default: 0.90)
--core-margin-px <int>       		 # Core mask margin [px] (default: 10)
--inlier-lo <percent>        		 # Low percentile for inliers (default: 5)
--inlier-hi <percent>        		 # High percentile for inliers (default: 95)
--log-file <path>            		 # Log file path
--log-level {DEBUG,INFO,...} 		 # Logging level (default: INFO)
--no-gui                     		 # Disable GUI prompts
--headless                   		 # Alias of --no-gui
```
Metadata precedence: `--xy/--dz/--z0` > OME‑TIFF > AICSImage physical sizes > manual prompt (if GUI enabled).

#### To reproduce synthetic dataset validation process:

a) generate synthetic dataset via:
```
python synthetic_flat_gel_generator.py 
```

b) validate the analyzer by selecting it and the folder containing the synthetic dataset via:
```
python synthetic_flat_gel_validator.py --gui
```
The analyzer is verified against synthetic ground truth generated from a known forward model.

## Main script outputs
For each analyzed folder under `--root`, the pipeline writes:

- `gel_surface_out/summary_metrics.csv` — one row per stack
- `gel_surface_out/run_metadata.json` — provenance (script hash, platform, package versions, fixed parameters)
- Diagnostic PNGs per stack prefix (in `gel_surface_out/`):
  - `_mip_well_outline.png`
  - `_height.png`
  - `_height_raw.png` (truncation mode)
  - `_residual_surface.png`
  - `_residual_surface_pm100um.png`
  - `_residual_bandpass.png`
- Lossless numeric maps per stack prefix (float32 TIFF, µm, NaN outside the well ROI; written alongside the PNGs):
  - `_height_abs_um.tif` — absolute (pre-tilt-correction) top-surface height
  - `_height_tiltcorr_um.tif` — tilt-corrected height
  - `_residual_surface_um.tif` — long-wavelength residual surface
  - `_residual_bandpass_um.tif` — S–L band-pass residual
  - `_confidence.tif` — per-pixel confidence in [0, 1] from peak prominence
  - `_validity.tif` — validity mask (1 = valid pixel within ROI, 0 otherwise)

### `summary_metrics.csv` CSV scheme
**Primary metrics**
- `TopZ_LFilteredMean_um` — inlier mean of low-pass (“L-filtered”) top-surface height (interpretable as thickness only if `Z0_um` is a valid bottom reference)
- `TopZ_LFilteredStd_um` — inlier SD of the L-filtered height (long-wavelength within-well non-uniformity)
- `BandpassStd_um` — SD of the S–L band-pass residual (shorter-wavelength residual dispersion)

**QC**
- `Valid_frac`, `Core_valid_frac` — fraction of pixels passing QC within the ROI / core ROI
- `TruncMode` — `top`, `bottom`, or empty (not truncated)

**Tilt / plane diagnostics**
- `TiltApplied` — whether within-FOV slope removal was applied
- `TiltAngle_deg` — derived tilt magnitude
- `Plane_a_um_per_mm`, `Plane_b_um_per_mm`, `Plane_c_um`, `Plane_R2`

**Counts**
- `Well_px_n`, `Valid_px_n`, `Core_px_n`, `Core_valid_px_n`, `Z_slices`

**Acquisition / analysis metadata**
- `XY_um_per_px`, `Z_step_um`, `Z0_um`
- `LambdaC_um`, `LambdaS_um`, `BeadDiamEff_um`
- `MaskScaleFrac`, `AnalysisInfo`
- `File` — input basename




