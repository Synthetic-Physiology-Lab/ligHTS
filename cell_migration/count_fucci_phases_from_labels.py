"""Optional add-on -- FUCCI cell-cycle phase fractions from saved labels.

Runs *after* ``segment_track_stack.py`` and requires no change to it. Instead of
re-segmenting the nuclei, it reuses the cell masks already saved as
``{stem}_cellpose_labels.tif`` and re-reads the matching original stack
(``{stem}.nd2`` / ``.tif`` / ``.tiff``, still in the folder) only to sample the
FUCCI reporter intensities.

For every stack, at the first / middle / last frame (or every frame with
``--all-frames``):

1. for each labelled cell measure its mean intensity in the two FUCCI nuclear
   channels (defaults: C0 = S/G2/M reporter, C3 = G1 reporter -- the same
   ``NUCLEI_CHANNELS = (0, 3)`` that segment_track_stack merges for the nuclei
   image of a 4-channel stack);
2. assign the cell to the dominant reporter:
     * C0 dominant  -> S/G2/M phase (geminin),
     * G1 channel dominant -> G1 phase (Cdt1);
3. express each phase as a percentage of the cells in that frame.

Cells whose intensity in BOTH channels is below ``--min-nuclear-intensity`` are
counted in the total but left unclassified (early-G1 double-negative), so the
two phase percentages need not sum to exactly 100.

Because it scores the whole-cell masks from the tracking step (not a dedicated
nuclear segmentation), the classification relies on the FUCCI reporters being
nuclear-localised so that whichever reporter dominates the cell also dominates
its nucleus; this is the same dominant-reporter rule as ``count_fucci_phases.py``
but without a second Cellpose run.

Grouping
--------
``replicate_id`` is the stack stem. ``group`` defaults to that stem (one group
per field of view); pass ``--group-regex`` with a single capture group to
aggregate replicates into conditions for the mean +/- SEM summary and figure.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tifffile  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
try:  # provenance is best-effort; never block analysis
    from utils.provenance import write_provenance
except Exception:  # pragma: no cover
    write_provenance = None

LABELS_SUFFIX = "_cellpose_labels.tif"
ORIGINAL_EXTENSIONS = (".nd2", ".tif", ".tiff")
OUT_PLOT = "fucci_phase_fractions.png"
OUT_SUMMARY_CSV = "fucci_phase_fractions.csv"
OUT_REPLICATE_CSV = "fucci_phase_per_replicate.csv"
MAGENTA = "#e5007e"  # C0 -> S/G2/M
CYAN = "#00aec7"  # G1 channel -> G1
NAMED_TIMEPOINTS = ("first", "middle", "last")

# FUCCI nuclear channels default to segment_track_stack's NUCLEI_CHANNELS.
DEFAULT_C0_CHANNEL = 0  # S/G2/M reporter (geminin)
DEFAULT_G1_CHANNEL = 3  # G1 reporter (Cdt1)


# --- stack reader (mirrors segment_track_stack.py; kept local so this ------
# --- add-on does not import the heavy Cellpose/Torch pipeline) -------------


def _guess_axes(arr: np.ndarray) -> str:
    """Infer an axis string when the file carries no usable metadata."""
    if arr.ndim == 2:
        return "YX"
    if arr.ndim == 3:
        return "TYX"
    if arr.ndim == 4:
        return "TCYX" if arr.shape[1] <= 4 else "TYXC"
    raise ValueError(f"Unsupported array with {arr.ndim} dimensions.")


def _to_tcyx(arr: np.ndarray, axes) -> np.ndarray:
    """Standardise any stack to the (T, C, Y, X) layout."""
    axes = "".join(axes).upper().replace("S", "C")
    if "T" not in axes:
        axes = axes.replace("Q", "T", 1).replace("I", "T", 1)
    if (
        len(axes) != arr.ndim
        or not {"Y", "X"} <= set(axes)
        or len(set(axes)) != len(axes)
    ):
        axes = _guess_axes(arr)
    for i in range(len(axes) - 1, -1, -1):
        if axes[i] not in "TCYX":
            if arr.shape[i] != 1:
                raise ValueError(f"Unsupported non-singleton axis '{axes[i]}'.")
            arr = np.squeeze(arr, axis=i)
            axes = axes[:i] + axes[i + 1 :]
    for ax in ("C", "T"):
        if ax not in axes:
            arr, axes = arr[np.newaxis], ax + axes
    return np.transpose(arr, [axes.index(a) for a in "TCYX"])


def read_stack(in_file: Path) -> np.ndarray:
    """Read ND2/TIF/TIFF and return a (T, C, Y, X) array."""
    if in_file.suffix.lower() == ".nd2":
        import nd2

        with nd2.ND2File(str(in_file)) as handle:
            return _to_tcyx(handle.asarray(), list(handle.sizes.keys()))
    with tifffile.TiffFile(str(in_file)) as handle:
        series = handle.series[0]
        return _to_tcyx(series.asarray(), series.axes)


# --- GUI helpers (match count_fucci_phases.py) -----------------------------


def gui_select_folder(title: str) -> str:
    """Return a folder chosen via Tk, or "" if no GUI is available."""
    try:
        import tkinter
        from tkinter import filedialog
    except Exception:  # pragma: no cover
        return ""
    try:
        root = tkinter.Tk()
        root.withdraw()
        chosen = filedialog.askdirectory(title=title)
        root.destroy()
    except Exception:  # pragma: no cover
        return ""
    return chosen or ""


def resolve_dir(cli_value, title: str, must_exist: bool) -> Path:
    """Return a directory from the CLI or a Tk chooser; exit on failure."""
    if cli_value is not None:
        path = Path(cli_value).expanduser().resolve()
    else:
        chosen = gui_select_folder(title)
        if not chosen:
            print(f"FATAL: no folder for {title!r}", file=sys.stderr)
            sys.exit(2)
        path = Path(chosen).expanduser().resolve()
    if must_exist and not path.is_dir():
        print(f"FATAL: not a directory: {path}", file=sys.stderr)
        sys.exit(2)
    return path


def find_original(folder: Path, stem: str) -> Path | None:
    """Return the source stack for a stem, skipping pipeline outputs."""
    for ext in ORIGINAL_EXTENSIONS:
        candidate = folder / f"{stem}{ext}"
        if candidate.name.endswith(LABELS_SUFFIX):
            continue
        if candidate.is_file():
            return candidate
    return None


def find_pairs(folder: Path) -> list:
    """Return [(stem, labels_path, original_path)] for matched stacks."""
    pairs = []
    for labels_path in sorted(folder.glob(f"*{LABELS_SUFFIX}")):
        stem = labels_path.name[: -len(LABELS_SUFFIX)]
        original = find_original(folder, stem)
        if original is None:
            print(
                f"  skip {labels_path.name}: no original stack for {stem!r}",
                file=sys.stderr,
            )
            continue
        pairs.append((stem, labels_path, original))
    return pairs


def derive_group(stem: str, pattern) -> str:
    """Return the experimental group for a stem via optional regex capture."""
    if pattern is None:
        return stem
    match = pattern.search(stem)
    if match and match.groups():
        return match.group(1)
    return stem


def load_labels(path: Path) -> np.ndarray:
    """Read a saved label stack and return it as a (T, Y, X) array."""
    array = np.asarray(tifffile.imread(str(path)))
    if array.ndim == 2:
        array = array[np.newaxis]
    if array.ndim != 3:
        raise ValueError(
            f"{path.name}: expected a (T,Y,X) label stack, got shape {array.shape}"
        )
    return array


def pick_frame_indices(n_frames: int, all_frames: bool) -> dict:
    """Return {timepoint label: frame index} for the requested frames."""
    if all_frames:
        return {index: index for index in range(n_frames)}
    return {"first": 0, "middle": n_frames // 2, "last": n_frames - 1}


def classify_frame(
    labels: np.ndarray, c0_plane: np.ndarray, g1_plane: np.ndarray, floor: float
) -> tuple:
    """Return (n_total, n_c0_sg2m, n_g1) for one frame by dominant reporter."""
    ids = np.unique(labels)
    ids = ids[ids != 0]
    if ids.size == 0:
        return 0, 0, 0
    mean_c0 = ndi.mean(c0_plane, labels, ids)
    mean_g1 = ndi.mean(g1_plane, labels, ids)
    keep = (mean_c0 >= floor) | (mean_g1 >= floor)
    n_c0 = int(np.count_nonzero(keep & (mean_c0 > mean_g1)))
    n_g1 = int(np.count_nonzero(keep & (mean_g1 > mean_c0)))
    return int(ids.size), n_c0, n_g1


def process_stack(
    stem: str,
    group: str,
    labels: np.ndarray,
    stack: np.ndarray,
    c0_ch: int,
    g1_ch: int,
    floor: float,
    all_frames: bool,
) -> list:
    """Return per-timepoint phase counts for one stack."""
    n_frames_img, n_ch = stack.shape[0], stack.shape[1]
    for channel in (c0_ch, g1_ch):
        if not 0 <= channel < n_ch:
            raise ValueError(
                f"{stem}: channel {channel} out of range for a "
                f"{n_ch}-channel stack (valid 0..{n_ch - 1})"
            )
    n_frames = min(n_frames_img, labels.shape[0])
    rows = []
    for point, index in pick_frame_indices(n_frames, all_frames).items():
        c0_plane = stack[index, c0_ch].astype(np.float32)
        g1_plane = stack[index, g1_ch].astype(np.float32)
        total, n_c0, n_g1 = classify_frame(labels[index], c0_plane, g1_plane, floor)
        rows.append(
            {
                "replicate_id": stem,
                "group": group,
                "timepoint": point,
                "frame": index,
                "n_total": total,
                "n_c0_sg2m": n_c0,
                "n_g1": n_g1,
                "pct_c0_sg2m": 100.0 * n_c0 / total if total else np.nan,
                "pct_g1": 100.0 * n_g1 / total if total else np.nan,
            }
        )
    return rows


def _sem(values: np.ndarray) -> float:
    """Standard error of the mean; 0 for a single value."""
    n = len(values)
    return float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else 0.0


def summarise(reps: pd.DataFrame) -> pd.DataFrame:
    """Group x timepoint mean +/- SEM of each phase percentage."""
    out = []
    for (group, point), sub in reps.groupby(["group", "timepoint"]):
        out.append(
            {
                "group": group,
                "timepoint": point,
                "n_replicates": len(sub),
                "pct_c0_sg2m_mean": sub["pct_c0_sg2m"].mean(),
                "pct_c0_sg2m_sem": _sem(sub["pct_c0_sg2m"].dropna().to_numpy()),
                "pct_g1_mean": sub["pct_g1"].mean(),
                "pct_g1_sem": _sem(sub["pct_g1"].dropna().to_numpy()),
            }
        )
    return pd.DataFrame(out)


def _timepoint_axis(summary: pd.DataFrame, all_frames: bool):
    """Return (ordered timepoint labels, x positions, tick labels)."""
    if all_frames:
        order = sorted(summary["timepoint"].unique())
        return order, np.asarray(order, dtype=float), [str(t) for t in order]
    order = [t for t in NAMED_TIMEPOINTS if t in set(summary["timepoint"])]
    return order, np.arange(len(order)), [t.capitalize() for t in order]


def make_figure(summary: pd.DataFrame, out_path: Path, all_frames: bool) -> None:
    """One panel per group: magenta C0 (S/G2/M) and cyan G1 over time."""
    order, x, tick_labels = _timepoint_axis(summary, all_frames)
    groups = sorted(summary["group"].unique())
    n_cols = min(4, max(1, len(groups)))
    n_rows = int(np.ceil(len(groups) / n_cols))
    figure, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.0 * n_cols, 2.6 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for position, axis in enumerate(axes.flat):
        if position >= len(groups):
            axis.set_axis_off()
            continue
        group = groups[position]
        sub = summary[summary["group"] == group].set_index("timepoint")
        sub = sub.reindex(order)
        axis.errorbar(
            x,
            sub["pct_c0_sg2m_mean"],
            yerr=sub["pct_c0_sg2m_sem"],
            marker="o",
            ms=5,
            lw=1.8,
            capsize=2,
            color=MAGENTA,
        )
        axis.errorbar(
            x,
            sub["pct_g1_mean"],
            yerr=sub["pct_g1_sem"],
            marker="s",
            ms=5,
            lw=1.8,
            capsize=2,
            color=CYAN,
        )
        axis.set_title(group, fontsize=9)
        axis.set_ylim(0, 100)
        axis.set_xticks(x)
        axis.set_xticklabels(tick_labels, fontsize=8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="0.93", lw=0.6)
        if position % n_cols == 0:
            axis.set_ylabel("% of cells", fontsize=8.5)
    legend = [
        Line2D([0], [0], marker="o", color=MAGENTA, lw=1.8, label="C0 → S/G2/M"),
        Line2D([0], [0], marker="s", color=CYAN, lw=1.8, label="G1 channel → G1"),
    ]
    figure.legend(handles=legend, fontsize=9, frameon=False, loc="upper right", ncol=2)
    figure.suptitle("FUCCI phase fractions (replicate mean ± SEM)", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the configured argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="folder with segment_track_stack outputs and the "
        "original stacks (GUI picker if omitted)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="where to write the phase tables and figure (GUI picker if omitted)",
    )
    parser.add_argument(
        "--c0-channel",
        type=int,
        default=DEFAULT_C0_CHANNEL,
        help="FUCCI channel scored as S/G2/M (default 0)",
    )
    parser.add_argument(
        "--g1-channel",
        type=int,
        default=DEFAULT_G1_CHANNEL,
        help="FUCCI channel scored as G1 (default 3)",
    )
    parser.add_argument(
        "--min-nuclear-intensity",
        type=float,
        default=0.0,
        help="both-channel intensity floor below which a cell "
        "is counted but left unclassified",
    )
    parser.add_argument(
        "--all-frames",
        action="store_true",
        help="score every frame instead of first/middle/last",
    )
    parser.add_argument(
        "--group-regex",
        default=None,
        help="regex with one capture group applied to the "
        "stack stem to assign an experimental group; "
        "default is one group per field of view",
    )
    return parser


def main() -> int:
    """Count FUCCI phases per group over time; write CSVs and a figure."""
    args = build_arg_parser().parse_args()
    input_dir = resolve_dir(
        args.input_dir, "Select segment_track_stack output folder", True
    )
    output_dir = resolve_dir(args.output_dir, "Select output folder", False)
    output_dir.mkdir(parents=True, exist_ok=True)

    group_pattern = re.compile(args.group_regex) if args.group_regex else None

    pairs = find_pairs(input_dir)
    if not pairs:
        print(
            f"FATAL: no '{LABELS_SUFFIX}' + original-stack pairs in {input_dir}",
            file=sys.stderr,
        )
        return 2

    records = []
    for number, (stem, labels_path, original) in enumerate(pairs, start=1):
        print(f"[{number}/{len(pairs)}] {stem}", flush=True)
        labels = load_labels(labels_path)
        stack = read_stack(original)
        group = derive_group(stem, group_pattern)
        records.extend(
            process_stack(
                stem,
                group,
                labels,
                stack,
                args.c0_channel,
                args.g1_channel,
                args.min_nuclear_intensity,
                args.all_frames,
            )
        )

    reps = pd.DataFrame(records)
    reps.to_csv(output_dir / OUT_REPLICATE_CSV, index=False)
    summary = summarise(reps)
    summary.to_csv(output_dir / OUT_SUMMARY_CSV, index=False)
    make_figure(summary, output_dir / OUT_PLOT, args.all_frames)

    if write_provenance is not None:
        write_provenance(
            output_dir,
            params={
                "input_dir": str(input_dir),
                "c0_channel": args.c0_channel,
                "g1_channel": args.g1_channel,
                "min_nuclear_intensity": args.min_nuclear_intensity,
                "all_frames": bool(args.all_frames),
                "group_regex": args.group_regex,
                "reused_masks": "segment_track_stack cellpose labels",
                "n_stacks": len(pairs),
            },
            script_path=__file__,
            filename="count_fucci_phases_provenance.json",
        )

    print(f"wrote {OUT_SUMMARY_CSV}, {OUT_REPLICATE_CSV} and {OUT_PLOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
