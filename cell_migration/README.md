# Migration Analysis
This repository provides a reproducible workflow for segmenting, tracking, and extracting migration parameters of
 Nikon .nd2, .tif or .tiff time‑lapse microscopy datasets, that should be named in the form:
ID_SUBSTRATE_TAG_(other), where SUBSTRATE indicates the gel characteristics or CNTRL if the experiment is performed on glass,
while TAG represents the different conditions in each experiment.
The code called "rename_for_analysis.py" was used to rename files not compliant with this structure for figure 6d.

It is divided in two sub-routines:
 - [segment_track_stack.py](segment_track_stack.py) harnesses
[Cellpose](https://github.com/MouseLand/cellpose) for segmentation and
[LapTrack](https://github.com/yfukai/laptrack) for linking detections across
frames. The workflow outputs compressed label stacks, a CSV with tracking
metadata and an annotated video overlaying contours and track tails. Currently runs on CPU-only, if a GPU is available remove the no mkl requirement from the environment and update Cellpose model to GPU=True
 - [migration_analysis.py](migration_analysis.py) takes as input one or more folders containing CSV files with tracking
metadata, and performs migration quantification, directionality (nematic order) analysis, and velocity computation, producing both numerical datasets and figures


## Install

Using pip:
```bash
# Recommended:  install into a clean Python 3.11 virtual environment (venv or conda)
pip install -r requirements.txt
```

Using Conda:
```bash
conda env create -f environment.yml
conda activate stack-cell-tracking
```

## Basic usage

To process all image files within a directory use:
```bash
python segment_track_stack.py
```
To select one or multiple folders for migration analysis use:
```bash
python migration_analysis.py
```

## Additional options
[stack_cell_tracking_cli.py](stack_cell_tracking_cli.py) runs [segment_track_stack.py](segment_track_stack.py)
headless and exposes the pipeline parameters as flags (defaults are read from the pipeline module).
It accepts 1-channel stacks (segmented directly) or 4-channel stacks (channels joined into a
cytoplasm and a nuclei image before segmentation). If `--folder` is omitted a GUI folder picker is used.

```bash
python stack_cell_tracking_cli.py
    --folder /path/to/stack_files   # folder path (GUI picker if omitted)
    --diameter 30                   # approximate cell diameter in pixels
    --um_per_px 1.34                # micron-per-pixel scale
    --time_gap_min 15               # time gap between frames (minutes)
    --gpu / --no-gpu                # run Cellpose on GPU (default: CPU; falls back to CPU if unavailable)

  # tracking
    --link_dist 23                  # max distance between consecutive frames (px)
    --gap_dist 46                   # max distance covered over gaps (px)
    --gap_frames 1                  # max frame gap for re-linking a track
    --tail 50                       # frames of track permanence in the movie

  # segmentation
    --cellprob_threshold -2.0       # Cellpose cell-probability threshold
    --flow_threshold 0.8            # Cellpose flow-error threshold
    --min_size 8                    # minimum mask size (px)
    --model_type cyto3              # Cellpose model
    --cyto_channels 1,2             # 4-channel input: channels joined as cytoplasm
    --nuclei_channels 0,3           # 4-channel input: channels joined as nuclei
```


## Main script outputs
 1.  [segment_track_stack.py](segment_track_stack.py)

For each stack in the folder, the script saves the outputs in the same folder:

- '_cellpose_labels.tif' — CellPose output of segmented cells
- '_tracking.csv' — Tracking information for each cell in the stack
- '_tracking.mov'  — Quality control video with original images overlapped with segmented cell contours and assigned tracks
- 'segment_track_provenance.json' — provenance record (SHA-256 of the script, git commit, interpreter/platform, and run parameters) for auditability

 2. [migration_analysis.py](migration_analysis.py)

For every valid CSV inside each folder:

- '_track_movement.tiff' - Per-file track-movement TIFF plots
- '_nematic.tiff'- Per-file nematic order histogram TIFFs

For each folder in input:

- '_recap.csv' - Summary statistics of each FOV in the folder in CSV
- '_Summary_WNO.png' - Per-folder statistics on directionality parameter (Nematic Order)
- '_Summary_MeanVel.png' - Per-folder statistics on migration velocity 
- '_combined_track_plot_TAG.tiff' - Trajectories from all CSVs in the folder grouped by condition (TAG)
- 'migration_analysis_provenance.json' - provenance record (SHA-256 of the script, git commit, interpreter/platform, and run parameters) for auditability

Global metrics across all folders:

- 'Summary_recap.csv' - CSV containing all calculated parameters with one row per FOV across all experiments
- 'Tidy_recap.csv' - CSV containing all calculated parameters with one row per track across all folders
- 'Combined_Summary_WNO.png' - Plot comparing directionality parameter per condition across all experiments
- 'Combined_Summary_MeanVel.png' - Plot comparing migration velocity per condition across all experiments
- 'migration_analysis_combined_provenance.json' - provenance record for the combined cross-folder run (SHA-256 of the script, git commit, interpreter/platform, parameters)

## Migration metric definitions and QC (migration_analysis.py)

Track QC:
- Keep tracks spanning at least 8 frames.
- Discard entire tracks with any temporal gap > 2 frames.
- Exclude tracks with zero total displacement.
- Skip CSV files with fewer than 10 valid tracks after QC.

Directionality (WNO):
- Step angle is computed from consecutive centroid displacements.
- Step-wise nematic contribution is cos(2·(Angle − 90°)) relative to the reference axis (groove axis aligned to image vertical).
- Per-track WNO is displacement-weighted across steps.
- Per-FOV WNO is a weighted mean across tracks using per-track weights displacement x number of frame in which the cell is identified.

Velocity:
- Per-track mean velocity is total displacement divided by elapsed time, using the user-provided micron-per-pixel scale and time gap between frames.
- Per-FOV mean velocity is the mean across tracks.

Note: inferential statistics and multiple-comparison correction are not executed inside migration_analysis.py; the script exports per-FOV recap tables used for downstream statistical testing.