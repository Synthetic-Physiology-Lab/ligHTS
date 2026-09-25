# Nanoindentation-anchored calibration of confocal and holographic groove depth (SI Note S15, Figure S11, Figure 6f)

This folder produces the two conversion constants used in the manuscript and the Figure 6f data:

    d = zeta * a      confocal apparent optical depth a  -> physical depth d
    d = C * phi       HoloMonitor native phase contrast phi -> physical depth d

Both constants are fixed on one 40 µm-pitch, 75 mg/mL GelMA specimen, measured by three methods: nanoindentation (the physical anchor), confocal z-stacking, and holographic phase imaging (HPI).

Every depth is measured with **one estimator** (`scripts/groovekit.py`):
1. FFT orientation, refined by maximising the profile contrast;
2. a 10 % trimmed-mean collapse along the grooves;
3. a quadratic detrend;
4. a 6-harmonic fit at the refined pitch;
5. **the peak-to-trough of the fundamental**, reported as the depth.

C is valid only for phases measured with this estimator.

**Phase unit.** HoloMonitor AppSuite stores phase in units of one wavelength of optical path (optical path length = phi * lambda). C therefore corresponds to an effective gel-to-medium index contrast delta_n_eff = lambda / C.

## Inputs (raw data: BioImage Archive / Zenodo, see Data Availability)

| Input | Content |
|---|---|
| [BIA] `c06_multimodal-calibration/` (`--raw`) | the calibration specimen, in two subfolders: `e02_holographic/w001_field<nn>_f<nn>.tiff` (5 fields × 7 frames, 16-bit, phase range in tag 40092, on which holo_to_tiff runs to extract the medians used in step2) and `e01_confocal/zstack<nn>.nd2` (on which groove_analyzer runs to extract the csv used in step3. The csv is also available in Zenodo |
| [Zenodo] `Nanoindentation_Calibration/matrix_scan18/` (`--nano-dir`) | use only the 40 × 40 Chiaro map X-11…X-50, 1600 curves (`*I-01.txt`, 7 µm grid). The first ten columns of the 50 × 40 raster (X-01…X-10, stage-settling offset) are discarded. **Step 1 does not do this trimming** — the folder must already contain only the 1600 curves. |
| [Zenodo] `NanoindentationCalibration/groove_recap_260805_confo.csv` (`--confocal-recap`) | `groove_analyzer.py` recap of the plate confocal stacks (apparent depths)|
| [BIA] `c05_holographic-phase-imaging/e01_groove-qc` (`--plates`)| dataset needed to regenerate figure 6f and perform step2 and step3 |

## Run (from the raw data to Figures)

```bash
# 0) per-field registered median maps of the plates (median over frames with registration response >= 0.3)
python ../holo_to_tiff.py --input <RAW> --out <stacks_8bit_square>
# 1) calibration specimen
python scripts/step1_measure_calibration.py --raw <c06_multimodal-calibration> --nano-dir <Nanoindentation_Calibration/matrix_scan18>
# 2) plates, same estimator
python scripts/step2_measure_plates.py --plates <stacks_8bit_square>
# 3) constants, transfer test, Figure 6f tables
python scripts/step3_lock_constants.py --confocal-recap <groove_recap_260805_confo.csv>
# 4) Figure 6f panel
python scripts/figure6f.py
# 5) Figure S11 panels
python scripts/figureS11.py
```

Step 2 also accepts the archived 8-bit maps of the original submission (`1536_HOLO_processed/*_MEDIAN.tif`). Their encoding is read from their metadata.

## Outputs (generated in a `outputs/` folder)

| File | Content |
|---|---|
| `calibration_constants.json` | d_nano, a, phi; zeta, C, Lambda with uncertainties; delta_n_eff; optical bounds; transfer test |
| `plate_fields.csv` | per-field phase, pitch and QC for the 380 pitch-labelled plate fields |
| `figure6f_source_data.csv` | per-field HPI phase and physical depth |
| `group_summary.csv` | HPI vs confocal per pitch, with expanded uncertainty (k = 2) |
| `transfer_test.csv`, `uncertainty_budget.csv` | Lambda per pitch; uncertainty terms |
| `../figures/Figure6f_physical_groove_depth.*` | Figure 6f panel |
| `../figures/FigureS11.*`| Figure S11 panels |

## Reference results

| Quantity | Value |
|---|---|
| d_nano | 23.60 ± 0.67 µm (1600 curves) |
| a (confocal) | 15.61 ± 1.19 µm |
| phi (HPI) | 0.7335 ± 0.0395 |
| **zeta** | **1.51 ± 0.12** |
| **C** | **32.17 ± 1.96 µm per native phase unit** |
| delta_n_eff | 0.0197 |
| Lambda (specimen) | 21.28 ± 1.99 |
| Lambda 60 / 80 µm plates | 21.98 ± 0.17 / 20.56 ± 0.13 |
| Lambda anchor | 21.10 |

**Figure 6f**:

| Pitch | HPI | Confocal × ζ | HPI − confocal | U (k = 2) |
|---|---|---|---|---|
| 40 µm | 11.30 µm | 9.05 µm | +24.8 % | 22.5 % |
| 60 µm | 18.78 µm | 19.40 µm | −3.2 % | 19.6 % |
| 80 µm | 27.76 µm | 26.82 µm | +3.5 % | 19.2 % |

## Relation to the other scripts

- `../holo_to_tiff.py` writes the registered 8-bit maps with an uncalibrated storage scale that step 2 decodes. The median uses only frames that registered.
- The depths reported by `../groove_analyzer_holo.py` (crest minus adjacent valleys) are not calibrated.

