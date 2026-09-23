"""Build the six-panel multimodal calibration figure.

Panels a-c show the native surface reconstructions of the three modalities, each
in its own measured quantity and with no cross-modal correction. Panel d shows
the two modalities that share a height axis before any optical-to-physical
correction, panel e places the confocal axial factor against the limits set by
geometrical optics, and panel f overlays all three after calibration.

Run ``python3 make_figure.py`` after ``run_calibration.py``. The HPI panel is
drawn from one field folder under ``HPI/HPI``; pass ``--field NAME`` to choose
it, otherwise the first one alphabetically is used.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from cmcrameri import cm as cmc

import multimodal_calibration as M

BASE = Path(__file__).resolve().parent
FIELD_UM = 273.0  # common field of view, set by the square nanoindentation map

SIZE_INDEX, SIZE_LABEL, SIZE_TICK = 11.0, 9.5, 8.5
COLOUR_NANO, COLOUR_CONFOCAL, COLOUR_HPI = "#000000", "#0072B2", "#D55E00"
COLOUR_MODELS = ("#0072B2", "#009E73", "#CC79A7")
FAILED_COLOUR = "0.6"


def set_style() -> None:
    """Apply the manuscript style: Arial, three type sizes, vector-safe output."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": SIZE_TICK,
            "axes.labelsize": SIZE_LABEL,
            "axes.titlesize": SIZE_LABEL,
            "xtick.labelsize": SIZE_TICK,
            "ytick.labelsize": SIZE_TICK,
            "legend.fontsize": SIZE_TICK,
            "axes.linewidth": 0.7,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def symmetric_limits(
    field: np.ndarray, percentile: float = 99.0
) -> tuple[float, float]:
    """Colour limits symmetric about zero so the diverging map stays centred."""
    finite = np.asarray(field)[np.isfinite(field)]
    span = float(np.percentile(np.abs(finite), percentile)) if finite.size else 1.0
    return -span, span


def levelled(field: np.ndarray, step_um: float) -> np.ndarray:
    """Remove tilt line by line so the groove is displayed about a zero mean."""
    period_px = M.NOMINAL_PITCH_UM / step_um
    out = np.full_like(field, np.nan, dtype=float)
    for row in range(field.shape[0]):
        if np.isfinite(field[row]).sum() > 8:
            out[row] = M.robust_form_removal(field[row], 5.0 * period_px)
    return out - np.nanmedian(out)


def centre_crop(field: np.ndarray, step_um: float, extent_um: float) -> np.ndarray:
    """Take a centred square window of the requested physical size."""
    want = round(extent_um / step_um)
    if want >= min(field.shape):
        return field
    row0 = (field.shape[0] - want) // 2
    col0 = (field.shape[1] - want) // 2
    return field[row0 : row0 + want, col0 : col0 + want]


def draw_map(
    ax: plt.Axes,
    field: np.ndarray,
    extent_um: float,
    measurand: str,
    unit: str,
    title: str,
) -> None:
    """Render one native map, with a scale bar and a labelled colour bar."""
    vmin, vmax = symmetric_limits(field)
    cmap = cmc.vik.copy()
    cmap.set_bad(FAILED_COLOUR)  # failed contacts are not a data value
    image = ax.imshow(
        np.ma.masked_invalid(field),
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=(0.0, extent_um, 0.0, extent_um),
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(title, fontsize=SIZE_LABEL, fontweight="bold", pad=3)
    ax.plot(
        [extent_um * 0.06, extent_um * 0.06 + 50.0],
        [extent_um * 0.07] * 2,
        color="k",
        lw=2.0,
        solid_capstyle="butt",
    )
    ax.text(
        extent_um * 0.06 + 25.0,
        extent_um * 0.105,
        "50 µm",
        ha="center",
        va="bottom",
        fontsize=SIZE_TICK,
    )
    bar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    bar.set_label(f"{measurand} ({unit})", fontsize=SIZE_LABEL)
    bar.ax.tick_params(labelsize=SIZE_TICK)
    bar.outline.set_linewidth(0.6)


def default_hpi_field(base: Path) -> Path:
    """Return the first field folder under HPI/HPI, or explain what is missing."""
    root = base / "HPI" / "HPI"
    fields = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    if not fields:
        raise SystemExit(
            f"No field folders found in {root}. Expected one subfolder per field, "
            "each holding that field's repeat TIFF frames (see README)."
        )
    return fields[0]


def load_maps(hpi_field: Path) -> dict[str, tuple[np.ndarray, float]]:
    """Return one representative native map per modality, on a common field."""
    grid = np.load(BASE / "nano_grid.npy")
    nano = levelled(np.rot90(grid), M.NANO_STEP_UM)

    volume = tifffile.imread(min((BASE / "tiff").glob("*.tif")))
    peak = np.argmax(volume, axis=0).astype(float) * M.CONFOCAL_DZ_UM
    peak[np.max(volume, axis=0) < np.median(volume) * 1.05] = np.nan
    confocal = levelled(centre_crop(peak, M.CONFOCAL_XY_UM, FIELD_UM), M.CONFOCAL_XY_UM)

    opd = M.load_hpi_field(hpi_field)
    hpi = levelled(centre_crop(opd, M.HPI_XY_UM, FIELD_UM), M.HPI_XY_UM)
    return {
        "nano": (nano, nano.shape[1] * M.NANO_STEP_UM),
        "confocal": (confocal, confocal.shape[1] * M.CONFOCAL_XY_UM),
        "hpi": (hpi, hpi.shape[1] * M.HPI_XY_UM),
    }


def load_profiles(calibration: dict) -> dict[str, tuple[np.ndarray, float]]:
    """Return the phase-referenced, floor-zeroed profile of each modality."""
    profiles: dict[str, tuple[np.ndarray, float]] = {}

    nano = pickle.load((BASE / "nano.pkl").open("rb"))
    profiles["nano"] = (nano.profile, M.NANO_STEP_UM)

    confocal_stack = []
    for path in sorted((BASE / "tiff").glob("*.tif")):
        volume = tifffile.imread(path)
        peak = np.argmax(volume, axis=0).astype(float) * M.CONFOCAL_DZ_UM
        peak[np.max(volume, axis=0) < np.median(volume) * 1.05] = np.nan
        result = M.analyse_surface(
            centre_crop(peak, M.CONFOCAL_XY_UM, FIELD_UM), M.CONFOCAL_XY_UM
        )
        confocal_stack.append(
            M.align_to_common_phase(
                result.profile, M.CONFOCAL_XY_UM, M.NOMINAL_PITCH_UM
            )
        )
    length = min(p.size for p in confocal_stack)
    profiles["confocal"] = (
        np.nanmean([p[:length] for p in confocal_stack], axis=0),
        M.CONFOCAL_XY_UM,
    )

    hpi_stack = []
    for folder in sorted((BASE / "HPI" / "HPI").iterdir()):
        if not folder.is_dir():
            continue
        result = M.analyse_surface(
            centre_crop(M.load_hpi_field(folder), M.HPI_XY_UM, FIELD_UM), M.HPI_XY_UM
        )
        hpi_stack.append(
            M.align_to_common_phase(result.profile, M.HPI_XY_UM, M.NOMINAL_PITCH_UM)
        )
    length = min(p.size for p in hpi_stack)
    profiles["hpi"] = (np.nanmean([p[:length] for p in hpi_stack], axis=0), M.HPI_XY_UM)

    profiles["nano"] = (
        M.align_to_common_phase(nano.profile, M.NANO_STEP_UM, M.NOMINAL_PITCH_UM),
        M.NANO_STEP_UM,
    )
    return profiles


def main() -> None:
    """Assemble and write the figure in vector and raster formats."""
    parser = argparse.ArgumentParser(
        description="Build the six-panel multimodal calibration figure."
    )
    parser.add_argument(
        "--field",
        default=None,
        help="Field folder under HPI/HPI to draw the HPI panel from "
        "(default: the first one alphabetically).",
    )
    arguments = parser.parse_args()
    if arguments.field:
        hpi_field = BASE / "HPI" / "HPI" / arguments.field
        if not hpi_field.is_dir():
            raise SystemExit(f"Field folder not found: {hpi_field}")
    else:
        hpi_field = default_hpi_field(BASE)

    set_style()
    calibration = json.loads((BASE / "calibration.json").read_text())
    maps = load_maps(hpi_field)
    profiles = load_profiles(calibration)

    figure = plt.figure(figsize=(7.09, 4.9))
    grid = figure.add_gridspec(
        2, 3, hspace=0.70, wspace=0.72, left=0.10, right=0.965, top=0.92, bottom=0.11
    )
    axes = [figure.add_subplot(grid[i // 3, i % 3]) for i in range(6)]

    draw_map(
        axes[0],
        maps["nano"][0],
        maps["nano"][1],
        "Physical height",
        "µm",
        "Nanoindentation",
    )
    draw_map(
        axes[1],
        maps["confocal"][0],
        maps["confocal"][1],
        "Optical height",
        "µm",
        "Confocal z-stacking",
    )
    draw_map(
        axes[2], maps["hpi"][0], maps["hpi"][1], "Optical path difference", "nm", "HPI"
    )

    # (d) Before any optical-to-physical correction.
    axis = axes[3]
    for key, colour, label in (
        ("confocal", COLOUR_CONFOCAL, "Confocal, optical"),
        ("nano", COLOUR_NANO, "Nano, physical"),
    ):
        profile, step = profiles[key]
        floored = M.floor_reference(profile, M.NOMINAL_PITCH_UM / step)
        axis.plot(
            np.arange(floored.size) * step, floored, lw=1.4, color=colour, label=label
        )
    axis.set_xlim(0, FIELD_UM)
    axis.set_xticks([0, 100, 200])
    axis.set_xlabel("Position across grooves (µm)")
    axis.set_ylabel("Uncorrected height (µm)", labelpad=2)
    _floor_axis(axis, profiles, ("confocal", "nano"), {}, headroom=0.55)
    axis.legend(frameon=False, loc="upper center", handlelength=1.3, borderaxespad=0.2)

    # (e) Axial factor against the geometrical-optics limits.
    axis = axes[4]
    bounds = calibration["bounds"]
    axis.axhspan(bounds["paraxial"], bounds["marginal_ray"], color="0.92", zorder=0)
    for value, label, colour, style in (
        (bounds["marginal_ray"], "Marginal ray", COLOUR_MODELS[2], "--"),
        (bounds["lyakin_stallinga"], "Lyakin / Stallinga", COLOUR_MODELS[1], "-."),
        (bounds["paraxial"], "Paraxial", COLOUR_MODELS[0], "--"),
    ):
        axis.axhline(value, color=colour, lw=1.1, ls=style)
        axis.text(
            0.04,
            value + 0.012,
            label,
            transform=axis.get_yaxis_transform(),
            fontsize=SIZE_TICK,
            color=colour,
            va="bottom",
        )
    low, high = calibration["zeta_ci"]
    zeta = calibration["zeta"]
    axis.errorbar(
        [0.7],
        [zeta],
        yerr=[[zeta - low], [high - zeta]],
        fmt="o",
        ms=6,
        capsize=4,
        lw=1.6,
        color=COLOUR_HPI,
        zorder=5,
    )
    axis.annotate(
        f"ζ = {zeta:.3f}",
        xy=(0.7, zeta),
        xytext=(0.78, zeta),
        fontsize=SIZE_TICK,
        fontweight="bold",
        color=COLOUR_HPI,
        ha="left",
        va="center",
    )
    axis.set_xlim(0, 1)
    axis.set_xticks([])
    axis.set_ylim(1.29, 1.73)
    axis.set_yticks([1.3, 1.4, 1.5, 1.6, 1.7])
    axis.set_ylabel("Optical-to-physical\nfactor ζ", labelpad=2)

    # (f) After the modality-specific conversions.
    axis = axes[5]
    scales = {
        "nano": (1.0, "Nanoindentation"),
        "confocal": (zeta, f"Confocal × {zeta:.3f}"),
        "hpi": (
            1.0 / calibration["delta_n"] / 1000.0,
            f"HPI ÷ {calibration['delta_n']:.5f}",
        ),
    }
    for key, colour in (
        ("nano", COLOUR_NANO),
        ("confocal", COLOUR_CONFOCAL),
        ("hpi", COLOUR_HPI),
    ):
        profile, step = profiles[key]
        factor, label = scales[key]
        floored = M.floor_reference(profile * factor, M.NOMINAL_PITCH_UM / step)
        axis.plot(
            np.arange(floored.size) * step,
            floored,
            "o-" if key == "nano" else "-",
            ms=2.6,
            lw=1.3,
            color=colour,
            label=label,
        )
    axis.set_xlim(0, FIELD_UM)
    axis.set_xticks([0, 100, 200])
    axis.set_xlabel("Position across grooves (µm)")
    axis.set_ylabel("Physical height (µm)", labelpad=2)
    _floor_axis(axis, profiles, ("nano", "confocal", "hpi"), scales, headroom=1.0)
    axis.legend(frameon=False, loc="upper center", handlelength=1.3, borderaxespad=0.2)

    for index, axis in enumerate(axes):
        box = axis.get_position()
        figure.text(
            box.x0 - 0.052,
            box.y1 + 0.055,
            "abcdef"[index],
            fontsize=SIZE_INDEX,
            fontweight="bold",
            va="top",
            ha="left",
        )

    stem = BASE / "Figure_multimodal_calibration"
    figure.savefig(f"{stem}.pdf")
    figure.savefig(f"{stem}.eps", format="eps")
    figure.savefig(f"{stem}.jpg", dpi=600, pil_kwargs={"quality": 95})
    plt.close(figure)


def _floor_axis(
    axis: plt.Axes,
    profiles: dict[str, tuple[np.ndarray, float]],
    keys: tuple[str, ...],
    scales: dict[str, tuple[float, str]],
    headroom: float = 1.0,
) -> None:
    """Set the y range from the groove floor to a margin above the highest trace."""
    peaks = []
    for key in keys:
        profile, step = profiles[key]
        factor = scales.get(key, (1.0, ""))[0]
        floored = M.floor_reference(profile * factor, M.NOMINAL_PITCH_UM / step)
        peaks.append(float(np.nanmax(floored)))
    # Five micrometres below the groove floor and above the highest trace, plus
    # room for the legend so it cannot sit on top of the data.
    top = max(peaks) + 5.0
    axis.set_ylim(-5.0, top + 0.42 * (top + 5.0) * headroom)


if __name__ == "__main__":
    main()
