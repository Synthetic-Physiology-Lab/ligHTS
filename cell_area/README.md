# Segmentation for cell area extraction in LigHTS
This repository contains the image-segmentation and feature-extraction workflow used to quantify projected single-cell 
area from 20x confocal images of HT1080 Actin‑EGFP/Tubulin‑RFP cells on glass and LigHTS GelMA films.

Scope note: the repository prepares a merged long-format table for downstream statistics and plotting. The final statistical
inference described in the manuscript is performed separately.
Scripts assume to run with active GPU and installed CUDA
## Install

Using pip:
Recommended:  install into a clean Python 3.10 virtual environment (venv or conda)
```bash
pip install -r requirements.txt

```

Using Conda:

```bash
conda env create -f environment.yml
conda activate cell_area

```

## Input layout

All three scripts glob relative to the current working directory, so run them from the folder
that holds the experiment folders:

```
<your data root>/            <- run the scripts from here
|-- 250330_filtered/         <- one folder per experiment
|   |-- 3_15_flat_....nd2
|   |-- 7_CNTRL_....nd2
|   |-- figures/             <- written by segment_cellpose_sam_masks.py
|   |-- CSVs/                <- written by segment_cellpose_sam_masks.py
|   `-- histograms/          <- written by plot_histograms.py
|-- 250331_filtered/
`-- 250403_filtered/
```

The naming is part of the contract and is not configurable:

- **Experiment folder** must end in `_filtered` (`segment_cellpose_sam_masks.py` globs
  `*_filtered/*.nd2`) and must contain `250330`, `250331` or `250403`. `merge_csvs.py` maps those to
  `experiment_idx` 1, 2 and 3 and raises `ValueError: Experiment ... not found!` for anything else.
- **ND2 / CSV file name** must start with the FOV number followed by `_` (`merge_csvs.py` reads
  `FOV = int(name.split("_")[0])`) and must contain one of the condition tokens below. A file with
  none of these tokens is silently ignored by `plot_histograms.py` and `merge_csvs.py`.

  | token in the file name | condition | `condition_index` |
  |---|---|---:|
  | `15_flat` | Stiffer gels | 3 |
  | `75_flat` | Softer gels | 2 |
  | `CNTRL` | Control | 1 |

## Basic usage
We use the new Cellpose SAM and segment the tubulin and actin channels together to get reliable cell masks that are analyzed to extract and plot cell area for statistical analysis.

The `segment_cellpose_sam_masks.py` script *expects folders that end with _filtered.
From the folder, the script:
- loads ND2 via AICSImageIO and extracts the actin + tubulin channels (channels 1 and 2 of the 0-indexed 4-channel stack),
  - segments whole-cell masks using CellposeSAM
  - writes per-image outputs:
    - `figures/<stem>.png` QC panel (image vs. masks),
    - `figures/masks_2ch_<stem>.tif` label mask TIFF,
    - `CSVs/<stem>.csv` per-cell `label` and `area` (µm²; pixel spacing from ND2 metadata).

The `plot_histograms.py` reads the per-image CSVs and writes histograms for quick distribution checks.

The `merge_csvs.py` concatenates all per-image CSVs into a single dataset for downstream statistics and plotting.


1. **Segmentation**:
   ```bash
   python /path/to/cell_area/segment_cellpose_sam_masks.py
   ```

2. **Plot histograms**:
   ```bash
   python /path/to/cell_area/plot_histograms.py
   ```

3. **Merge CSVs**:
   ```bash
   python /path/to/cell_area/merge_csvs.py

   ```

## Outputs

`segment_cellpose_sam_masks.py` writes, per ND2 file, into the experiment folder:
`figures/<stem>.png` (QC panel), `figures/masks_2ch_<stem>.tif` (label mask) and
`CSVs/<stem>.csv` (per-cell `label` and `area` in um^2).

`plot_histograms.py` writes:

- `<experiment>_filtered/histograms/<stem>.pdf` - one area histogram per input CSV
- `histogram_area_15_flat.pdf`, `histogram_area_75_flat.pdf`, `histogram_area_CNTRL.pdf` -
  pooled histogram per condition, in the working directory
- `overview_experiment.pdf` - box plot of the three conditions with the individual cells overlaid
- the median area of each condition, printed to stdout

`merge_csvs.py` writes `data_ligHTS_cell_area.csv`, one row per segmented cell:

| column | meaning |
|---|---|
| `label` | mask label within its source image |
| `area` | projected cell area (um^2) |
| `analysis_info` | segmentation provenance string (`name\|version\|sha256\|timestamp`) |
| `condition_index` | 3 = stiffer gels, 2 = softer gels, 1 = control |
| `experiment_idx` | 1, 2 or 3, from the date in the experiment folder name |
| `FOV` | FOV number taken from the file name, then renumbered to a dense 1..N sequence across the whole merged table |
| `condition` | `Stiffer gels`, `Softer gels` or `Control` |
| `file_name` | source CSV file name, without the `.csv` suffix |
| `experiment` | experiment folder name |
| `merge_info` | merge provenance string |

## Notes

- If you have a GPU but it's not being detected, make sure you have the correct CUDA toolkit installed



