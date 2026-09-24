"""Multimodal groove calibration: nanoindentation, confocal, holographic phase imaging.

The three techniques report different physical quantities for the same grooved
hydrogel surface, and two conversion constants tie them together:

* ``zeta``  converts confocal *nominal optical* depth to physical depth. A dry
  objective focused through a medium of different refractive index images an
  interface at a shifted axial position, so the reconstructed depth is not the
  physical one.
* ``delta_n`` converts holographic *optical path difference* to physical
  thickness. Transmission phase is a path integral, so the measured OPD equals
  ``(n_gel - n_medium) * d``.

Nanoindentation is the reference because it is a mechanical contact measurement
referenced to a calibrated piezo: it involves no refractive index, no point
spread function and no phase-to-height model. Its two known systematics, tip
convolution and contact-point detection on a compliant surface, both cause
under-reading, so the reference depth is a lower bound.

Order of operations matters and is enforced here: the nanoindentation surface is
reconstructed first, then the confocal factor is derived against it, and only
then is the holographic index difference derived. No constant is assumed a
priori. 
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy import ndimage, optimize, signal

# --------------------------------------------------------------------------- #
# Acquisition constants, all taken from instrument metadata
# --------------------------------------------------------------------------- #

NOMINAL_PITCH_UM = 40.0

# Nanoindentation (Optics11 Chiaro), from the per-point file headers.
NANO_STEP_UM = 7.0
NANO_PROBE_RADIUS_UM = 3.5
# The acquired grid is 41 x 40. The first raster column carries a stage-settling
# transient, so it is dropped to give a square 40 x 40 field.
NANO_SETTLING_COLUMN = 10

# Confocal (Nikon spinning disk), from the ND2 metadata.
CONFOCAL_XY_UM = 1.343986
CONFOCAL_DZ_UM = 0.900
CONFOCAL_NA = 0.75
CONFOCAL_IMMERSION_INDEX = 1.000  # air objective

# Holographic phase imaging, from the TIFF resolution tag and the source laser.
HPI_XY_UM = 25400.0 / 45872.308  # ResolutionUnit is inch; 25400 um per inch
HPI_WAVELENGTH_UM = 0.635
HPI_PHASE_TAG = 40092  # private tag carrying "Min (0) = .. Max (65535) = .."

# Medium refractive index at the source wavelength and 37 C (PBS).
MEDIUM_INDEX = 1.3316

# Earlier hard-coded value in the acquisition script, retained only for
# comparison; holo_to_tiff.py has been since corrected to use the calibrated DELTA_N.
LEGACY_DELTA_N = 0.0064
LEGACY_HPI_XY_UM = 0.54


# --------------------------------------------------------------------------- #
# Surface metrology (ISO 25178-2 / ISO 16610)
# --------------------------------------------------------------------------- #


@dataclass
class GrooveMetrology:
    """Depth, pitch and the diagnostics that show the measurement is sound."""

    depth_mean_um: float
    depth_sd_um: float
    n_periods: int
    pitch_um: float
    misalignment_deg: float
    naive_amplitude: float
    locked_amplitude: float
    form_transfer_at_pitch: float
    injected_recovery: float
    profile: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))


def gaussian_cutoff_sigma(cutoff_px: float) -> float:
    """Gaussian regression-filter sigma for a given cutoff (ISO 16610-21)."""
    return cutoff_px * math.sqrt(math.log(2.0)) / (math.pi * math.sqrt(2.0))


# A Gaussian regression filter needs the profile to span several nesting
# lengths; below this ratio the filter is dominated by its own edge behaviour
# and begins to absorb the groove itself.
MIN_NESTING_SPANS = 4.0


def robust_form_removal(profile: np.ndarray, nesting_px: float) -> np.ndarray:
    """Remove long-wavelength form from a profile, leaving the groove intact.

    A robust Gaussian regression filter (ISO 16610-31) is used when the profile
    is long enough to support one, with a nesting index several times the groove
    pitch so that its transmission at the groove wavelength is negligible.

    Short profiles are the case that matters here: the nanoindentation field
    spans only about seven grooves, and on such a profile the regression filter
    attenuates the groove by more than two per cent whatever nesting index is
    chosen, because its edge behaviour dominates. In that regime the form is
    genuinely only a tilt, so a robust first-order fit is both sufficient and
    unbiased. Which branch was taken is observable in the reported transfer.
    """
    values = np.asarray(profile, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 8:
        return values - np.nanmedian(values)
    filled = np.interp(np.arange(values.size), np.flatnonzero(finite), values[finite])

    if values.size < MIN_NESTING_SPANS * nesting_px:
        index = np.arange(values.size, dtype=float)
        coefficients = np.polyfit(index, filled, 1)
        for _ in range(3):  # iteratively reweighted, so outliers cannot tilt the fit
            residual = filled - np.polyval(coefficients, index)
            scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
            if scale <= 0:
                break
            weights = np.clip(1.0 - (residual / (4.0 * scale)) ** 2, 0.0, 1.0) ** 2
            coefficients = np.polyfit(index, filled, 1, w=weights)
        out = np.where(finite, filled - np.polyval(coefficients, index), np.nan)
        return out - np.nanmedian(out)

    sigma = gaussian_cutoff_sigma(nesting_px)
    form = ndimage.gaussian_filter1d(filled, sigma, mode="nearest")
    # One robust reweighting pass limits the pull of outliers on the form.
    residual = filled - form
    scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
    if scale > 0:
        weights = np.clip(1.0 - (residual / (3.0 * scale)) ** 2, 0.0, 1.0) ** 2
        form = ndimage.gaussian_filter1d(
            filled * weights, sigma, mode="nearest"
        ) / np.maximum(ndimage.gaussian_filter1d(weights, sigma, mode="nearest"), 1e-9)
    out = np.where(finite, filled - form, np.nan)
    return out - np.nanmedian(out)


def fundamental(profile: np.ndarray, period_px: float) -> tuple[float, float]:
    """Amplitude and phase of the groove fundamental in a profile."""
    values = np.asarray(profile, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 4:
        return float("nan"), float("nan")
    index = np.flatnonzero(finite)
    kernel = np.exp(-2j * np.pi * index / period_px)
    coefficient = np.sum(values[finite] * kernel) / finite.sum()
    return 2.0 * abs(coefficient), float(np.angle(coefficient))


def complex_fundamental(profile: np.ndarray, period_px: float) -> complex:
    """Complex Fourier coefficient of the groove fundamental."""
    values = np.asarray(profile, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 4:
        return complex("nan")
    index = np.flatnonzero(finite)
    return complex(
        np.sum(values[finite] * np.exp(-2j * np.pi * index / period_px)) / finite.sum()
    )


def fourier_shift(profile: np.ndarray, shift_px: float) -> np.ndarray:
    """Shift a profile by a sub-pixel amount without changing its amplitude.

    The Fourier shift theorem is exact for band-limited data, unlike
    interpolation, which smooths and therefore reduces groove amplitude.
    """
    values = np.asarray(profile, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 4:
        return values
    filled = np.interp(np.arange(values.size), np.flatnonzero(finite), values[finite])
    frequencies = np.fft.fftfreq(filled.size)
    shifted = np.fft.ifft(
        np.fft.fft(filled) * np.exp(-2j * np.pi * frequencies * shift_px)
    )
    return np.real(shifted)


def groove_wavevector(
    field_2d: np.ndarray, step_um: float, pitch_um: float
) -> tuple[float, float]:
    """Groove pitch and tilt from the two-dimensional spectrum.

    Estimating the tilt line by line fails on a coarsely sampled field: with only
    a few samples per period the per-line phase is dominated by noise and the
    fitted angle can even come out with the wrong sign, which then makes the
    phase-locked average *worse* than a plain average. Fitting the wavevector to
    the whole field at once uses every sample simultaneously and is stable. The
    spectral peak is refined off the discrete grid, because on a field only a few
    tens of samples wide the frequency quantisation is coarse enough to miss a
    tilt of one or two degrees.

    Returns the pitch in micrometres and the tilt of the grooves in degrees.
    """
    valid = np.isfinite(field_2d)
    if valid.sum() < 16:
        return float("nan"), 0.0
    centred = np.where(valid, field_2d - np.nanmean(field_2d), 0.0)
    rows, columns = np.mgrid[0 : field_2d.shape[0], 0 : field_2d.shape[1]]
    x_um = columns[valid] * step_um
    y_um = rows[valid] * step_um
    heights = centred[valid]

    def negative_amplitude(wavevector: np.ndarray) -> float:
        phase = 2.0 * np.pi * (wavevector[0] * x_um + wavevector[1] * y_um)
        return -abs(np.sum(heights * np.exp(-1j * phase))) / heights.size

    # Coarse peak from the discrete spectrum, which is fast, then an off-grid
    # refinement, which is what actually resolves a tilt of a degree or two on a
    # small field. The refinement is evaluated on a bounded random subsample so
    # that the cost does not grow with the size of the field.
    filled = np.where(valid, centred, 0.0)
    window = np.hanning(filled.shape[0])[:, None] * np.hanning(filled.shape[1])[None, :]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(filled * window)))
    freq_y = np.fft.fftshift(np.fft.fftfreq(filled.shape[0], d=step_um))
    freq_x = np.fft.fftshift(np.fft.fftfreq(filled.shape[1], d=step_um))
    mesh_x, mesh_y = np.meshgrid(freq_x, freq_y)
    radius = np.hypot(mesh_x, mesh_y)
    nominal = 1.0 / pitch_um
    band = (radius > 0.4 * nominal) & (radius < 2.0 * nominal)
    if not band.any():
        return float("nan"), 0.0
    peak = np.unravel_index(np.argmax(np.where(band, spectrum, 0.0)), spectrum.shape)
    best = (float(mesh_x[peak]), float(mesh_y[peak]))

    if heights.size > 40000:
        selection = np.random.default_rng(0).choice(heights.size, 40000, replace=False)
        x_um, y_um, heights = x_um[selection], y_um[selection], heights[selection]

    # The refinement is confined to one frequency bin around the coarse peak.
    # Left unbounded it can walk to an unrelated high-frequency noise maximum,
    # because the objective has many local optima.
    bin_x = 1.0 / (filled.shape[1] * step_um)
    bin_y = 1.0 / (filled.shape[0] * step_um)
    refined = optimize.minimize(
        negative_amplitude,
        np.asarray(best),
        method="Nelder-Mead",
        bounds=[
            (best[0] - bin_x, best[0] + bin_x),
            (best[1] - bin_y, best[1] + bin_y),
        ],
        options={"xatol": 1e-9, "fatol": 1e-14, "maxiter": 4000},
    )
    kx, ky = refined.x if refined.success else np.asarray(best)
    if not np.hypot(kx, ky):
        return float("nan"), 0.0
    # Lines of constant phase satisfy kx*x + ky*y = const, so the groove runs
    # along dx/dy = -ky/kx: the tilt is the negative of the wavevector angle.
    # A real field also has conjugate spectral peaks, so k and -k are equivalent
    # and the angle is folded into (-90, 90] to give one unambiguous tilt.
    angle_deg = -float(np.degrees(np.arctan2(ky, kx)))
    angle_deg = (angle_deg + 90.0) % 180.0 - 90.0
    return float(1.0 / np.hypot(kx, ky)), angle_deg


def phase_locked_average(
    field_2d: np.ndarray, period_px: float, angle_deg: float
) -> np.ndarray:
    """Average lines after shifting each to a common groove phase.

    This recovers the full groove amplitude from a misaligned image without
    rotating and resampling it, which would blur the surface.

    The Fourier shift is circular, so on a tall field a tilt of a few degrees
    wraps tens of pixels of the opposite edge into each line. Those wrapped
    samples are masked before averaging; leaving them in makes the phase-locked
    average *worse* than a plain average, which is exactly the failure the
    amplitude-gain diagnostic is there to catch.
    """
    shift_per_row = math.tan(math.radians(angle_deg))
    width = field_2d.shape[1]
    stack = []
    for row in range(field_2d.shape[0]):
        shift = -shift_per_row * row
        line = fourier_shift(field_2d[row], shift)
        wrapped = math.ceil(abs(shift))
        if wrapped:
            if wrapped >= width:
                continue
            line = line.astype(float).copy()
            if shift > 0:
                line[:wrapped] = np.nan
            else:
                line[width - wrapped :] = np.nan
        stack.append(line)
    if not stack:
        return np.nanmedian(field_2d, axis=0)
    return np.nanmedian(np.vstack(stack), axis=0)


def fourier_upsample(profile: np.ndarray, factor: int) -> np.ndarray:
    """Band-limited interpolation by zero-padding the spectrum.

    The nanoindentation profile has fewer than six samples per groove, so the
    true crest and valley almost never fall on a sample and a depth read from
    the discrete extrema is systematically too shallow. Zero-padding the
    spectrum is the exact interpolator for band-limited data: it adds no
    amplitude and, unlike spline or linear interpolation, does not smooth the
    peaks it is meant to locate.
    """
    values = np.asarray(profile, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 8 or factor <= 1:
        return values
    filled = np.interp(np.arange(values.size), np.flatnonzero(finite), values[finite])
    return signal.resample(filled, filled.size * int(factor))


def period_depths(profile: np.ndarray, period_px: float) -> np.ndarray:
    """Crest-to-adjacent-valleys depth for every complete groove period."""
    distance = max(round(0.6 * period_px), 1)
    peaks, _ = signal.find_peaks(profile, distance=distance)
    valleys, _ = signal.find_peaks(-profile, distance=distance)
    depths = []
    for peak in peaks:
        left = valleys[valleys < peak]
        right = valleys[valleys > peak]
        if left.size and right.size:
            floor = 0.5 * (profile[left[-1]] + profile[right[0]])
            depths.append(profile[peak] - floor)
    return np.asarray(depths, dtype=float)


def analyse_surface(
    field_2d: np.ndarray,
    step_um: float,
    pitch_um: float = NOMINAL_PITCH_UM,
    nesting_factor: float = 5.0,
) -> GrooveMetrology:
    """Measure groove depth and pitch with the full ISO chain and diagnostics.

    Depth is read from the unfiltered, form-removed profile; filtering is used
    only to locate the crests and valleys. A sinusoid of known amplitude is
    injected at the groove wavelength and re-measured through the same chain, so
    the reported recovery shows directly that no step in the analysis attenuates
    the quantity being measured.
    """
    spectral_pitch, angle = groove_wavevector(field_2d, step_um, pitch_um)
    accepted = bool(
        np.isfinite(spectral_pitch)
        and 0.5 * pitch_um < spectral_pitch < 2.0 * pitch_um
    )
    if accepted:
        pitch_um = spectral_pitch
    period_px = pitch_um / step_um
    naive = np.nanmedian(field_2d, axis=0)

    # The spectral tilt is defined up to a sign, because a real field has
    # conjugate spectral peaks and because an upstream transpose or rotation
    # flips the handedness of the image axes. Rather than assume a convention,
    # both signs are tried and the one that actually recovers more groove
    # amplitude is kept. Correct alignment can only increase the amplitude, so a
    # locked profile that is worse than the plain average is rejected outright;
    # the gain that was achieved is reported alongside the depth.
    candidates = [(fundamental(naive, period_px)[0], 0.0, naive)]
    for signed_angle in (angle, -angle):
        candidate = phase_locked_average(field_2d, period_px, signed_angle)
        candidates.append(
            (fundamental(candidate, period_px)[0], signed_angle, candidate)
        )
    best_amplitude, angle, locked = max(candidates, key=lambda item: item[0])
    del best_amplitude

    nesting_px = nesting_factor * period_px
    profile = robust_form_removal(locked, nesting_px)

    # Transmission of the form filter at the groove wavelength.
    probe = np.sin(2.0 * np.pi * np.arange(profile.size) / period_px)
    transfer = (
        fundamental(robust_form_removal(probe, nesting_px), period_px)[0]
        / fundamental(probe, period_px)[0]
    )

    # Injected-signal recovery through the identical chain. The comparison is
    # made on the complex fundamental, because the injected sinusoid and the
    # groove need not share a phase and their amplitudes do not simply add.
    injected_amplitude = float(np.nanstd(profile)) or 1.0
    injected = profile + injected_amplitude * probe
    recovery = float(
        abs(
            complex_fundamental(robust_form_removal(injected, nesting_px), period_px)
            - complex_fundamental(profile, period_px)
        )
        / (0.5 * injected_amplitude)
    )

    # Locate the extrema on a band-limited interpolation of the profile, so the
    # depth is not biased low by the coarse sampling of the grid.
    upsample = max(math.ceil(8.0 / period_px), 1)
    depths = period_depths(fourier_upsample(profile, upsample), period_px * upsample)
    measured_pitch = spectral_pitch if accepted else float("nan")
    return GrooveMetrology(
        depth_mean_um=float(np.mean(depths)) if depths.size else float("nan"),
        depth_sd_um=float(np.std(depths, ddof=1)) if depths.size > 1 else float("nan"),
        n_periods=int(depths.size),
        pitch_um=measured_pitch,
        misalignment_deg=angle,
        naive_amplitude=fundamental(naive, period_px)[0],
        locked_amplitude=fundamental(locked, period_px)[0],
        form_transfer_at_pitch=float(transfer),
        injected_recovery=recovery,
        profile=profile,
    )


def measure_pitch(profile: np.ndarray, period_px: float, step_um: float) -> float:
    """Groove pitch from the mean crest-to-crest spacing."""
    peaks, _ = signal.find_peaks(profile, distance=max(round(0.6 * period_px), 1))
    if peaks.size < 2:
        return float("nan")
    return float(np.mean(np.diff(peaks)) * step_um)


# --------------------------------------------------------------------------- #
# Nanoindentation: the physical reference
# --------------------------------------------------------------------------- #

_HEADER_PATTERN = re.compile(r"X-(\d+)\s+Y-(\d+)\s+I-(\d+)", re.IGNORECASE)


def _header_value(text: str, key: str) -> float:
    match = re.search(re.escape(key) + r"[^\-\d]*([+-]?[\d.]+(?:[eE][+-]?\d+)?)", text)
    return float(match.group(1)) if match else float("nan")


def read_nano_points(folder: Path) -> pd.DataFrame:
    """Read every Chiaro indentation header into a per-point table."""
    records = []
    for path in sorted(folder.rglob("*.txt")):
        match = _HEADER_PATTERN.search(path.name)
        if match is None:
            continue
        text = path.read_text(encoding="latin-1")
        # The numeric block begins at a line starting "Time (s)"; splitting on the
        # bare word would truncate at the acquisition timestamp on the first line.
        header = re.split(r"^Time \(s\)", text, maxsplit=1, flags=re.MULTILINE)[0]
        status = re.search(r"Status\s*\t?\s*(\w+)", header)
        records.append(
            {
                "xi": int(match.group(1)),
                "yi": int(match.group(2)),
                "x_um": _header_value(header, "X-position (um)"),
                "y_um": _header_value(header, "Y-position (um)"),
                "z_surface_um": _header_value(header, "Z surface (um)"),
                "modulus_pa": _header_value(header, "E[eff] (Pa)"),
                "status": status.group(1) if status else "UNKNOWN",
            }
        )
    return pd.DataFrame(records)


def nano_square_grid(
    points: pd.DataFrame, drop_column: int = NANO_SETTLING_COLUMN
) -> np.ndarray:
    """Rasterise the surface onto a square grid, dropping the settling column.

    The acquisition is 41 columns by 40 rows. The first raster column shows a
    stage-settling offset, so removing it both squares the field and removes the
    least trustworthy data. The grid is indexed from the first *present* column,
    so the returned array contains no empty border rows or columns.
    """
    kept = points[points.xi != drop_column]
    columns = np.sort(kept.xi.unique())
    rows = np.sort(kept.yi.unique())
    grid = np.full((rows.size, columns.size), np.nan)
    column_index = {value: i for i, value in enumerate(columns)}
    row_index = {value: i for i, value in enumerate(rows)}
    accepted = kept[kept.status.str.upper() == "OK"]
    for xi, yi, height in zip(accepted.xi, accepted.yi, accepted.z_surface_um):
        grid[row_index[yi], column_index[xi]] = height
    return grid


def nano_metrology(grid: np.ndarray) -> GrooveMetrology:
    """Groove metrology on the nanoindentation surface.

    The stage rasters along the grooves, so the grid arrives with the grooves
    horizontal while both optical modalities have them vertical. It is rotated so
    that all three share one convention: grooves along axis 0, groove-normal
    direction along axis 1.
    """
    return analyse_surface(np.rot90(grid), NANO_STEP_UM)


# --------------------------------------------------------------------------- #
# Confocal: optical-to-physical axial factor
# --------------------------------------------------------------------------- #


def axial_factor_bounds(
    numerical_aperture: float, n1: float, n2: float
) -> dict[str, float]:
    """Geometrical-optics limits on the axial re-scaling factor.

    The paraxial ratio is a lower bound and the marginal-ray ratio an upper
    bound; wave-optical treatments place the realised value between them, near
    the intermediate Lyakin-Stallinga estimate. A measured factor outside this
    band would indicate an error rather than a physical result.
    """
    paraxial = n2 / n1
    marginal = math.tan(math.asin(min(numerical_aperture / n1, 1.0))) / math.tan(
        math.asin(min(numerical_aperture / n2, 1.0))
    )
    lyakin = math.sqrt(
        (n2**2 - numerical_aperture**2 / 2.0) / (n1**2 - numerical_aperture**2 / 2.0)
    )
    return {"paraxial": paraxial, "lyakin_stallinga": lyakin, "marginal_ray": marginal}


# --------------------------------------------------------------------------- #
# Holographic phase imaging: optical path difference
# --------------------------------------------------------------------------- #


def read_phase_range(path: Path) -> tuple[float, float]:
    """Read the phase values that the 16-bit range maps onto, in radians.

    The acquisition software records the mapping in the private Windows
    XPComment tag, which is UTF-16LE and uses a comma decimal separator.
    Decoding it as a single-byte encoding interleaves NUL characters, which
    silently turns every number into a string of zeros and yields a zero phase
    range and therefore a uniformly zero optical path. The encoding is handled
    explicitly, and the bracketed count limits are removed before parsing so
    they cannot be mistaken for data.
    """
    with Image.open(path) as image:
        raw = image.tag_v2.get(HPI_PHASE_TAG)
    if raw is None:
        raise ValueError(f"No phase-range tag in {path.name}")
    text = (
        raw.decode("utf-16-le", errors="ignore") if isinstance(raw, bytes) else str(raw)
    )
    text = text.replace("\x00", "").strip()
    body = re.sub(r"\(\s*\d+\s*\)", "", text)
    numbers = re.findall(r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?", body)
    if len(numbers) < 2:
        raise ValueError(f"Unparsable phase range in {path.name}: {text!r}")
    low, high = (float(value.replace(",", ".")) for value in numbers[:2])
    if not (math.isfinite(low) and math.isfinite(high) and high > low):
        raise ValueError(f"Invalid phase range in {path.name}: {low} to {high}")
    return low, high


def load_optical_path_difference(path: Path) -> np.ndarray:
    """Convert one 16-bit phase image to optical path difference in nanometres.

    Transmission phase is a path integral, so ``OPD = phi * lambda / (2 pi)``
    holds independently of any refractive index. Keeping the result as OPD, and
    converting to thickness only later, is what allows the index difference to be
    calibrated instead of assumed. The conversion is done in floating point from
    the raw 16-bit data, never through an 8-bit intermediate.
    """
    counts = tifffile.imread(path).astype(np.float64)
    low, high = read_phase_range(path)
    phase = low + counts * (high - low) / 65535.0
    return phase * HPI_WAVELENGTH_UM / (2.0 * np.pi) * 1000.0


def load_hpi_field(folder: Path) -> np.ndarray:
    """Combine the repeat frames of one field by pixelwise median.

    The median rejects the frame-to-frame flicker of the source without
    broadening the groove edges, which a mean would do if any frame were shifted.
    """
    frames = [load_optical_path_difference(p) for p in sorted(folder.glob("*.tif*"))]
    if not frames:
        raise ValueError(f"No TIFF frames in {folder}")
    return np.median(np.stack(frames), axis=0)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


@dataclass
class Calibration:
    """The two conversion constants with their supporting numbers."""

    depth_nano_um: float
    depth_nano_sd_um: float
    depth_confocal_um: float
    optical_path_nm: float
    zeta: float
    zeta_ci: tuple[float, float]
    delta_n: float
    delta_n_ci: tuple[float, float]
    bounds: dict[str, float]

    @property
    def gel_index(self) -> float:
        """Refractive index of the gel at the source wavelength."""
        return MEDIUM_INDEX + self.delta_n


def bootstrap_ratio(
    numerator: float,
    numerator_sd: float,
    denominators: np.ndarray,
    n_resamples: int = 20000,
    seed: int = 20260715,
) -> tuple[float, tuple[float, float]]:
    """Confidence interval for a ratio whose two terms carry separate errors.

    The reference depth contributes its own sampling uncertainty and the optical
    fields contribute theirs, so both are resampled together rather than one
    being treated as exact.
    """
    rng = np.random.default_rng(seed)
    reference = rng.normal(numerator, numerator_sd, n_resamples)
    resampled = rng.choice(
        denominators, (n_resamples, denominators.size), replace=True
    ).mean(axis=1)
    ratios = reference / resampled
    return float(np.median(ratios)), (
        float(np.percentile(ratios, 2.5)),
        float(np.percentile(ratios, 97.5)),
    )


def calibrate(
    nano: GrooveMetrology,
    confocal_depths_um: np.ndarray,
    optical_path_nm: np.ndarray,
) -> Calibration:
    """Derive both conversion constants against the mechanical reference."""
    zeta, zeta_ci = bootstrap_ratio(
        nano.depth_mean_um,
        nano.depth_sd_um / math.sqrt(max(nano.n_periods, 1)),
        confocal_depths_um,
    )
    # delta_n = OPD / thickness, with OPD in micrometres.
    rng = np.random.default_rng(20260716)
    reference = rng.normal(
        nano.depth_mean_um,
        nano.depth_sd_um / math.sqrt(max(nano.n_periods, 1)),
        20000,
    )
    resampled = rng.choice(
        optical_path_nm, (20000, optical_path_nm.size), replace=True
    ).mean(axis=1)
    ratios = (resampled / 1000.0) / reference
    return Calibration(
        depth_nano_um=nano.depth_mean_um,
        depth_nano_sd_um=nano.depth_sd_um,
        depth_confocal_um=float(np.mean(confocal_depths_um)),
        optical_path_nm=float(np.mean(optical_path_nm)),
        zeta=zeta,
        zeta_ci=zeta_ci,
        delta_n=float(np.median(ratios)),
        delta_n_ci=(
            float(np.percentile(ratios, 2.5)),
            float(np.percentile(ratios, 97.5)),
        ),
        bounds=axial_factor_bounds(
            CONFOCAL_NA,
            CONFOCAL_IMMERSION_INDEX,
            MEDIUM_INDEX + float(np.median(ratios)),
        ),
    )


def nano_independent_delta_n_bounds(
    optical_path_nm: np.ndarray,
    confocal_depths_um: np.ndarray,
    bounds: dict[str, float],
) -> tuple[float, float]:
    """Bound the index difference using the two optical modalities alone.

    The product ``zeta * delta_n`` equals ``OPD / d_confocal`` and uses no
    mechanical measurement. Constraining ``zeta`` to its physical band therefore
    brackets ``delta_n`` independently of the nanoindentation reference, which
    breaks the shared dependence of the two constants on that single reference.
    """
    product = (
        float(np.mean(optical_path_nm)) / 1000.0 / float(np.mean(confocal_depths_um))
    )
    return product / bounds["marginal_ray"], product / bounds["paraxial"]


def align_to_common_phase(
    profile: np.ndarray, step_um: float, pitch_um: float, crest_at_um: float = 20.0
) -> np.ndarray:
    """Refer a profile to a common groove phase for overlay.

    The three modalities image different regions and are not spatially
    registered, so the absolute groove phase carries no information. The shift is
    derived from the measured fundamental and applied with the Fourier shift
    theorem, so it is objective and amplitude-exact. No inversion or mirroring is
    applied, leaving any real difference in profile asymmetry visible.
    """
    period_px = pitch_um / step_um
    _, phase = fundamental(profile, period_px)
    shift = (crest_at_um + phase * pitch_um / (2.0 * np.pi)) / step_um
    return fourier_shift(profile, shift)


def floor_reference(profile: np.ndarray, period_px: float) -> np.ndarray:
    """Set zero at the mean groove floor of that modality.

    The three techniques share no absolute datum, so only height differences are
    comparable and each profile is referred to its own floor.
    """
    valleys, _ = signal.find_peaks(-profile, distance=max(round(0.6 * period_px), 1))
    floor = (
        float(np.mean(profile[valleys])) if valleys.size else float(np.nanmin(profile))
    )
    return profile - floor


__all__ = [
    "Calibration",
    "GrooveMetrology",
    "align_to_common_phase",
    "analyse_surface",
    "axial_factor_bounds",
    "calibrate",
    "floor_reference",
    "load_hpi_field",
    "load_optical_path_difference",
    "nano_independent_delta_n_bounds",
    "nano_metrology",
    "nano_square_grid",
    "read_nano_points",
]
