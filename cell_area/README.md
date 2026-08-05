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

### Provenance / auditability
Each script records the SHA-256 of the exact code version that produced the outputs:
`segment_cellpose_sam_masks.py` adds an `analysis_info` column (`name|version|sha256|timestamp`) to every
per-cell CSV and writes `segmentation_provenance.json`; `merge_csvs.py` adds a `merge_info` column to
`data_ligHTS_cell_area.csv` and writes `merge_provenance.json`; `plot_histograms.py` writes
`histograms_provenance.json`.

1. **Segmentation**:
   ```bash
   python segment_cellpose_sam_masks.py
   ```

2. **Plot histograms**:
   ```bash
   python plot_histograms.py
   ```

3. **Merge CSVs**:
   ```bash
   python merge_csvs.py

   ```
## Notes

- If you have a GPU but it's not being detected, make sure you have the correct CUDA toolkit installed



