# UV mask diffraction (1D periodic grating, scalar model)

## Summary
- Provides a **first-order scalar diffraction** model to compute 1D intensity profiles behind an **infinite periodic binary amplitude line grating** under **finite source divergence**.
- Intended for **qualitative/illustrative optics-only reasoning** (e.g., why a binary mask can yield a smoother, approximately sinusoidal profile at finite z), **not** for quantitative exposure prediction.
- Outputs publication-oriented **PNG** plots and matching **CSV** profiles from a single Python script.

### Key features
- Scalar Helmholtz propagation using an exact transfer function (non-paraxial).
- Periodic-grating formulation via diffraction orders (Fourier series).
- Incoherent averaging over illumination directions (cone in solid angle, or optional 1D fan).

### Out of scope
This code **does not** model or validate:
- Photoresist/photopolymer chemistry, kinetics, oxygen inhibition, diffusion, shrinkage, or any material response.
- Thick-mask effects (waveguiding, near-field coupling inside the mask), mask surface relief, or polarization.
- Scattering/absorption in media, partial coherence beyond the implemented angular incoherence, or system-specific imaging optics.


## Repository contents
- `uv_mask_profile_divergence.py` — main script (GUI output-folder picker when available).
- `MODEL_AND_OUTPUTS.md` — model overview, parameters to change, and output schema.

## Installation
Tested with Python 3.9+ in a clean virtual environment.

```bash
pip install -r requirements.txt
```

## Run
```bash
python uv_mask_profile_divergence.py
```

- A GUI prompts you to select an output folder when Tkinter is available.
- In headless environments (no GUI), outputs are written to the current working directory.

## Outputs
For each combination of line width / pitch / polarity / z, the script writes:
- `profile_<tag>.png` — intensity profiles for the configured divergences + nominal mask transmission.
- `profile_<tag>.csv` — matching numerical profiles.

Once per run, the script also writes:
- `run_metadata.json` — provenance for auditability: tool version, SHA-256 of the script, timestamp, and the model parameters used.

Notes:
- Intensities are **relative to the incident plane-wave amplitude** (optics-only). Values can exceed 1.0 due to constructive interference.
- If you need max-normalised curves (e.g., for shape-only comparisons), normalise in post-processing (e.g., divide each column by its maximum).

The CSV columns are:
- `x_um`: x-coordinate (µm)
- `d003`, `d008`, `d015`, ...: intensity for each divergence half-angle (degrees) as configured
- `mask_nominal`: nominal mask-plane transmission intensity (0 or 1)

## How to change the model parameters
Edit the constants near the top of `uv_mask_profile_divergence.py`, including:
- `Z_LIST_UM`, `DIVERGENCES_DEG`
- `CONE_MODEL` (`solid_angle` or `fan_x`)
- `N_THETA`, `N_PHI` (direction sampling)
- `PITCH_VALUES_UM` (line width is derived as pitch / 2), `DX_TARGET_UM`