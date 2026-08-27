# Holographic imaging registered stack generation and analysis
---------------------------------------------------------------------

Pipeline to convert phase-encoded holographic TIFF frames into registered, square-cropped 8-bit stacks from which geometrical parameters are extracted.
Designed for holographic images with vertical groove patterns and per-frame phase metadata.

Workflow
----------

1) Run holo_to_tiff.py
2) GUI: select folder containing frames and enter pixel size (µm/px).
3) Group files by FOV from names like 'Well 1-5 3.tif' → FOV=5.
4) First frame per FOV:
     - Convert 16-bit phase→height (µm) using metadata
       'Min (0) = ... Max (65535) = ...' and
         h = φ * λ / (2πΔn) with Δn=0.00278, λ=0.635 µm (physical parameters of the system)
     - Estimate dominant grating angle via 2D FFT (grating normal).
     - Rotate to make grooves vertical; estimate pitch (µm) via 1D FFT.
5) Apply the same rotation to all frames; compute the largest
   NaN-free square region common to all rotated frames and crop them.
6) Map cropped heights to 8-bit.
7) Save multi-page TIFF as '{FOV}_{40|60|80}.tif' with calibration and a median single-page TIFF.
8) Run groove_analyzer_holo.py
9) GUI: select folder containing stack and enter pixel size (µm/px), z-slice (µm/gray level), Z0 (µm level corresponding to black gray = 0 pixels)
10) Analyze all stacks, extracting surface geometry parameters
11) Saves processed data

Optional calibration against nanoindentation and confocal images
-------------------
1) create in the same folder of the code a subfolder called tiff containing the tiff version of confocal calibration data
2) create in the same folder of the code a subfolder named conf_out containing groove_recap.csv as created via groove_analyzer.py
3) create in the same folder of the code two nested folders HPI/HPI containing the raw holographic microscopy data
4) create in the same folder of the code a folder named NanoindentationChiaro containing raw nanoindentation data
5) run run_calibration.py
6) run make_figure.py to get the final image

Optional validation
---------------------
To verify groove_analyzer_holo.py credibility, run validation pipeline:
1) Generate a realistic synthetic dataset via synthetic_groove_generator_holo.py
2) Use synthetic_groove_validator_holo.py --gui to select the groove_analyzer_holo.py as the analyzer to test and the synthetic data folder as test dataset
3) Compare performance reading the VALIDATION_REPORT.md file

Optical and physical constants of the model
---------------------------------------------

- Wavelength λ = 0.635 µm
- Refractive index contrast Δn = 0.00278
- Phase → height scale  
  \[
  h = \phi \cdot \frac{\lambda}{2\pi\Delta n} \approx 36.35\ \mu m/\text{rad}
  \]
- Height window mapped to 8-bit output
- Groove pitch labels snapped to nearest of 40, 60, 80 µm

Notes
-------
- Provenance / auditability: 'groove_analyzer_holo.py' records a SHA-256 of the running
  script (alongside the version string) in 'run_metadata.json', in the per-file
  '_results.csv'/'groove_recap.csv' rows, and in '_analysis_summary.txt'. 'holo_to_tiff.py'
  embeds its own version and SHA-256 in the ImageDescription metadata of every output TIFF.

- The analyzer script is verified (100% pass summarized by the validator script)
  against synthetic ground truth (synthetic dataset produced by the generator script
  from a known forward model)

- The analyzer code is intended for the sole use of analyzing the datasets and extracting the hydrogel
  geometries displayed in figure 6 e-f of the manuscript "LigHTS: Massively Parallel Biomimetic 
  Photo-Functionalization for Imaging-Based Ultra-High-Throughput Screening" 

- The 'holo_to_tiff.py' code to convert from holographic images to tiff stacks and obtain the median is also used to
  isolate cells for subsequent segmentation, tracking, and extract migration parameters reported in figure 6 d
  and supplementary figures and videos of the manuscript "LigHTS: Massively Parallel Biomimetic 
  Photo-Functionalization for Imaging-Based Ultra-High-Throughput Screening" 

## Input requirements

- Folder containing .tif / .tiff files
- Each file must contain phase min/max metadata in the ImageDescription (or equivalent TIFF tags)
- Pixel size is provided manually at runtime (µm / pixel)


## Install

Using pip:
```bash
Recommended:  install into a clean Python 3.10 virtual environment (venv or conda)
pip install -r requirements.txt
```

Using Conda:

```bash
conda env create -f environment.yml
conda activate holoreader
```

## Basic usage

To run the TIFF stack conversion:
```bash
python holo_to_tiff.py
```

To run the geometrical analysis on folders containing TIFF stacks and divided per pitch:
```bash
python groove_analyzer_holo.py
```
## Advanced Options
To calibrate the holographic reconstructor against nanoindentation and confocal datasets, after creating a folder tree as described before:
a) run the ancillary code via:
```
python run_calibration.py
```
b) generate figure panel via:
```
python make_figure.py
```

To reproduce synthetic dataset validation process and obtain metrics as reported in VALIDATION_REPORT.md:

a) generate syntethic dataset via:
```
python synthetic_groove_generator_holo.py
```

b) validate the analyzer by selecting it and the folder containing the synthetic dataset via:
```
python synthetic_groove_validator_holo.py --gui
```

## Analyzer outputs

For each input TIFF 'FILE.tif', the script creates a per-file subfolder `FILE_proc/` (plots are written unless '--no-plots' is set), containing:

- '_height_map.(png|pdf)' – processed height map
- '_avg_profile_annotated.(png|pdf)' – average row profile with detected peaks/valleys annotated
- '_pitch_hist.(png|pdf)' – pitch distribution histogram (when enough periods are detected)
- '_depth_hist.(png|pdf)' – depth distribution histogram (when enough periods are detected)
- '_analysis_summary.txt' – per-file metric summary (plain text)
- '_results.csv' – per-file metric summary (CSV)
- '_run_metadata.json' – per-file provenance (script version, platform, parameters)

Per-period arrays are also written whenever the corresponding values are available:

- '_pitches_um.csv' – raw per-period pitch values (µm)
- '_depths_um.csv' – raw per-period depth values (µm)

In the root directory (the selected folder, or the input file's parent):
- 'groove_recap.csv' – one-row-per-file summary of all key metrics
- 'run_metadata.json' – run-level provenance (script version, platform, calibration)

Note: each input TIFF gets its own `FILE_proc/` subfolder for the per-file outputs listed above; the root-level files sit alongside those subfolders.