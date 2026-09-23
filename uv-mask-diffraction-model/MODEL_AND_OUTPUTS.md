# Model and outputs

## Core computation
The script computes the 1D intensity profile behind a periodic line grating by decomposing the mask transmission into diffraction orders and propagating each order with the exact scalar Helmholtz propagator:

- Grating period: `pitch_um`
- Harmonic index: `m`
- Spatial frequency: `2π m / pitch`
- For a given incidence direction `(θ, φ)`:
  - `kx0 = k sin(θ) cos(φ)`
  - `ky0 = k sin(θ) sin(φ)`
  - `kx_m = kx0 + 2π m / pitch`
  - `kz_m = sqrt(k^2 - kx_m^2 - ky0^2)`

The field is assembled as:

`E(x,z) = exp(i kx0 x) * Σ_m c_m exp(i kz_m z) exp(i 2π m x / pitch)`

Intensity is computed as `Intensity = |E|^2`.

## Mask model (Fourier coefficients)
The default mask is a centered binary transmission function with duty cycle 0.5. The Fourier coefficients for the transparent fraction `duty` are:

`c_m = duty * sinc(duty * m)`  (NumPy definition: `sinc(x)=sin(πx)/(πx)`)

An `inverted` option is provided (swap transparent/opaque), implemented by coefficient sign changes.

## Source divergence model
Two deterministic sampling options are implemented:
- `CONE_MODEL = "solid_angle"`: samples a 3D cone with uniform weighting in solid angle (uniform in `u = cos θ`, uniform in `φ`), and **enforces azimuthal pairing** to avoid numerical left/right asymmetry.
- `CONE_MODEL = "fan_x"`: samples a 1D fan of tilts in x only (ky = 0).

The source is treated as **incoherent across directions**, so the final intensity is the weighted average of per-direction intensities.

## Outputs
### PNG
- One figure per condition.
- Figure size is set to ~9 × 6 cm, with bold 12 pt fonts.

### CSV
A CSV file is saved alongside each PNG. Columns:
- `x_um`: x-coordinate (µm)
- `dXXX`: intensity profile for divergence half-angle `XXX` degrees
- `mask_nominal`: nominal binary transmission intensity at the mask plane

All intensity values are **relative** (not calibrated absolute irradiance) and can exceed 1.0 due to constructive interference.

## Where to edit parameters
The following constants near the top of `uv_mask_profile_divergence.py` control most behavior:
- Wavelength: `WAVELENGTH_NM`
- Distances: `Z_LIST_UM`
- Divergence half-angles: `DIVERGENCES_DEG`
- Cone model + sampling: `CONE_MODEL`, `N_THETA`, `N_PHI`
- Spatial sampling: `DX_TARGET_UM`
