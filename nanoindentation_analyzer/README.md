# GelMA nanoindentation analysis

Analysis suite for Optics11 Chiaro spherical nanoindentation of GelMA hydrogels. Feeded a
single characterisation session (all gels measured together, e.g. 12 gels = 4 replicates × 3
concentrations, each gel a sub-folder of raw `.txt` force curves) it extracts an apparent
reduced indentation modulus per curve, aggregates by gel and by concentration, and produces the
statistics and figures.

## Method

The script extracts apparent reduced indentation modulus from a soft-matter Hertz sphere model:

- Depth axis ζ = piezo − cantilever, which removes cantilever compliance.
- Robust pre-contact baseline (line + MAD scatter), subtracted.
- Objective contact point that maximises R² over a fixed 0–4 µm window (coarse 40 nm scan, refined
  to 5 nm).
- Zero-intercept Hertz sphere `F = (4/3) E_app √R δ^1.5` fitted over 0–4 µm; the slope gives
  `E_app`, the reported apparent reduced modulus.
- Late-window (1.5–4 µm) self-consistency diagnostic.
- Pre-registered QC gates and a within-gel outlier screen (modified z-score on log10 E).
- Each gel represents the experimental unit: the gel median feeds the per-condition summary and statistics.

## Install

Using pip:
```bash
# Recommended: install into a clean Python 3.11 virtual environment (venv or conda)
pip install -r requirements.txt
```

Using Conda:
```bash
conda env create -f environment.yml
conda activate gelma-indentation
```

## Basic usage

Run the script and pick the session folder from the GUI:
```bash
python gelma_indentation_suite.py
```

## Advanced Options

Run headless by passing the session folder and an output folder:
```bash
python gelma_indentation_suite.py --raw <session_folder> --out <output_folder>
```

## Input requirements

- One session folder with one sub-folder per gel.
- Gel sub-folders named `GelMA_<conc>mgmL_gel<n>[_...]`, with concentration in mg/mL. Percent-w/v
  labels (7.5 / 10 / 15) and the raw `matrix_scan.._<pct>_<rep>` instrument names are also recognised.
- Curve files keep the instrument name `<label> S-1 X-<col> Y-<row> I-01.txt`.

## Outputs

### Per gel (one sub-folder per matrix scan)
- `per_curve_metrics.csv` — instrument-extracted and recalculated values per curve, with per-QC-gate
  pass/fail, thresholds, and an outlier flag
- `<sample_id>_curve_fits.eps` / `.jpg` — every force curve with its contact point and Hertz fit
- `<sample_id>_spatial_maps.eps` / `.jpg` — X–Y grid maps of apparent modulus, contact point, and
  QC / FOV technical success
- `matrix_report.txt`

### Per session (top-level folder)
- `master_per_curve.csv` — every curve of the session, one row each
- `per_matrix_summary.csv` — one row per gel
- `per_condition_summary.csv` — one row per concentration
- `statistics.csv` — ANOVA and pairwise tests (Welch/Holm, exact permutation, Hedges g)
- `superplot.eps` / `.jpg` — per-curve, gel-median and condition-mean figure
- `modulus_map_atlas.eps` / `.jpg` — every gel's modulus map, tiled by condition
- `contactpoint_map_atlas.eps` / `.jpg` — every gel's contact-point map
- `sensitivity_analysis.eps` / `.jpg` and `sensitivity_analysis.csv` — stability of the result to the
  contact rule, fit window and processing choices
- `session_report.txt`


