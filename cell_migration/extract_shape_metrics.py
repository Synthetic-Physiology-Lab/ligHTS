"""Optional add-on -- extended migration/morphology metrics from saved labels.

Runs *after* ``segment_track_stack.py`` and requires no change to it. For every
``{stem}_cellpose_labels.tif`` + ``{stem}_tracking.csv`` pair produced by the
segmentation/tracking step, this script:

1. re-reads the saved Cellpose label stack and measures, per object per frame,
   the shape descriptors that the base ``_tracking.csv`` does not carry
   (area, eccentricity, orientation, major/minor axis lengths);
2. joins those objects to their ``track_id`` through the ``(frame, label)`` key
   that both files share (``label`` in the tracking CSV is the integer pixel
   value of the mask in the label stack);
3. converts everything to physical units with ``--um-per-px`` and derives the
   per-step displacement / velocity / nematic metrics;
4. writes, per replicate, an enriched tracking CSV into
   ``<output-dir>/enriched_tracking/{stem}_tracking.csv`` 

Shape-descriptor definitions
----------------------------
* ``area_um2``            = pixel area x (um_per_px ** 2)
* ``perimeter_um``        = region perimeter x um_per_px
* ``major/minor_axis_length_um`` = axis lengths x um_per_px
* ``eccentricity``        = skimage region eccentricity (0 = circle .. ->1)
* ``aspect_ratio``        = major_axis / minor_axis (>= 1; NaN if minor == 0)
* ``polarization_index``  = 1 - minor_axis / major_axis, in [0, 1); an
                            elongation measure, 0 for a round cell and
                            approaching 1 for a highly polarised (elongated) one
* ``orientation_rad/deg`` = skimage major-axis orientation

Grouping
--------
``replicate_id`` is the stack stem. ``group`` defaults to that same stem (each
field of view is its own group). Pass ``--group-regex`` with a single capture
group to aggregate replicates into experimental conditions; the base tracking
step imposes no naming scheme, so grouping is left explicit here.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops_table

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
try:  # provenance is best-effort; never block analysis
    from utils.provenance import write_provenance
except Exception:  # pragma: no cover
    write_provenance = None

LABELS_SUFFIX = "_cellpose_labels.tif"
TRACKING_SUFFIX = "_tracking.csv"

# Defaults mirror stack_cell_tracking_cli.py.
DEFAULT_UM_PER_PX = 1.34
DEFAULT_INTERVAL_MIN = 15.0
REFERENCE_AXIS_DEG = 90.0
MIN_TRACK_FRAMES = 8
MAX_ALLOWED_GAP = 2
SENSITIVITY_THRESHOLDS = (5, 8, 12)

# skimage >= 0.19 canonical names (scikit-image 0.24 is pinned in this repo).
REGION_PROPERTIES = (
    "label",
    "area",
    "centroid",
    "eccentricity",
    "orientation",
    "axis_major_length",
    "axis_minor_length",
    "perimeter",
)


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


def find_pairs(folder: Path) -> list:
    """Return [(stem, labels_path, tracking_path)] for matched output files."""
    pairs = []
    for labels_path in sorted(folder.glob(f"*{LABELS_SUFFIX}")):
        stem = labels_path.name[: -len(LABELS_SUFFIX)]
        tracking_path = folder / f"{stem}{TRACKING_SUFFIX}"
        if not tracking_path.is_file():
            print(
                f"  skip {labels_path.name}: no matching {stem}{TRACKING_SUFFIX}",
                file=sys.stderr,
            )
            continue
        pairs.append((stem, labels_path, tracking_path))
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


def load_tracking(path: Path) -> pd.DataFrame:
    """Return a tidy (frame, label, track_id) table from a tracking CSV."""
    table = pd.read_csv(path)
    table = table.loc[:, ~table.columns.duplicated()]
    required = {"frame", "label", "track_id"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(
            f"{path.name}: tracking CSV missing columns {sorted(missing)}; "
            "expected the output of segment_track_stack.py"
        )
    tidy = table[["frame", "label", "track_id"]].copy()
    tidy["frame"] = tidy["frame"].astype(int)
    tidy["label"] = tidy["label"].astype(int)
    return tidy


def measure_labels(labels: np.ndarray, um_per_px: float) -> pd.DataFrame:
    """Measure shape descriptors for every object in every frame."""
    per_frame = []
    for frame_idx in range(labels.shape[0]):
        plane = labels[frame_idx]
        if not plane.any():
            continue
        props = pd.DataFrame(regionprops_table(plane, properties=REGION_PROPERTIES))
        props["frame"] = int(frame_idx)
        per_frame.append(props)
    if not per_frame:
        return pd.DataFrame()
    props = pd.concat(per_frame, ignore_index=True)

    major = props["axis_major_length"].to_numpy(dtype=float)
    minor = props["axis_minor_length"].to_numpy(dtype=float)
    orientation_rad = props["orientation"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        # polarization_index = 1 - minor/major (0 for a round cell, ->1 elongated)
        polarization = np.where(
            major > 0, 1.0 - minor / np.where(major > 0, major, 1), np.nan
        )
        aspect_ratio = np.where(
            minor > 0, major / np.where(minor > 0, minor, 1), np.nan
        )

    out = pd.DataFrame(
        {
            "frame": props["frame"].astype(int),
            "label": props["label"].astype(int),
            "object_label": props["label"].astype(int),
            "centroid_y_um": props["centroid-0"].to_numpy(dtype=float) * um_per_px,
            "centroid_x_um": props["centroid-1"].to_numpy(dtype=float) * um_per_px,
            "area_um2": props["area"].to_numpy(dtype=float) * (um_per_px**2),
            "perimeter_um": props["perimeter"].to_numpy(dtype=float) * um_per_px,
            "major_axis_length_um": major * um_per_px,
            "minor_axis_length_um": minor * um_per_px,
            "eccentricity": props["eccentricity"].to_numpy(dtype=float),
            "orientation_rad": orientation_rad,
            "orientation_deg": np.degrees(orientation_rad),
            "aspect_ratio": aspect_ratio,
            "polarization_index": polarization,
        }
    )
    return out


def build_enriched_frames(
    stem: str, group: str, labels: np.ndarray, tracking: pd.DataFrame, um_per_px: float
) -> pd.DataFrame:
    """Join measured objects to their track_id and add identity columns."""
    measured = measure_labels(labels, um_per_px)
    if measured.empty:
        return measured
    merged = measured.merge(tracking, on=["frame", "label"], how="inner")
    merged.insert(0, "source_file", stem)
    merged.insert(1, "replicate_id", stem)
    merged.insert(2, "group", group)
    return merged.drop(columns=["label"])


# --- ported migration metrics, sourced here from the labels ---------------


def major_axis_angle_deg(orientation_rad: np.ndarray) -> np.ndarray:
    """Convert skimage orientation to the migration-angle frame (degrees)."""
    return np.degrees(np.arctan2(-np.cos(orientation_rad), -np.sin(orientation_rad)))


def add_frame_metrics(data: pd.DataFrame, interval_min: float) -> pd.DataFrame:
    """Add per-step displacement, velocity, angle and nematic component."""
    data = data.sort_values(["track_uid", "frame"]).reset_index(drop=True)
    grouped = data.groupby("track_uid", sort=False)

    delta_x = grouped["centroid_x_um"].diff()
    delta_y = grouped["centroid_y_um"].diff()
    delta_frame = grouped["frame"].diff()

    data["step_distance_um"] = np.hypot(delta_x, delta_y)
    data["step_elapsed_min"] = delta_frame * interval_min
    data["instantaneous_velocity_um_per_min"] = (
        data["step_distance_um"] / data["step_elapsed_min"]
    )
    data["migration_angle_deg"] = np.degrees(np.arctan2(delta_y, delta_x)).mod(360)
    data["nematic_component"] = np.cos(
        2 * np.radians(data["migration_angle_deg"] - REFERENCE_AXIS_DEG)
    )
    axis_deg = major_axis_angle_deg(data["orientation_rad"].to_numpy(dtype=float))
    data["orientation_migration_alignment"] = np.cos(
        2 * np.radians(data["migration_angle_deg"] - axis_deg)
    )
    data["time_min"] = data["frame"] * interval_min
    return data


def _weighted_nematic(sub: pd.DataFrame) -> tuple:
    """Return (S_axis, S_magnitude) for one track's displacement steps."""
    weight = sub["step_distance_um"].to_numpy(dtype=float)
    angle = np.radians(sub["migration_angle_deg"].to_numpy(dtype=float))
    keep = np.isfinite(weight) & np.isfinite(angle) & (weight > 0)
    weight, angle = weight[keep], angle[keep]
    total = weight.sum()
    if total <= 0:
        return np.nan, np.nan
    s_axis = float(
        (weight * np.cos(2 * (angle - np.radians(REFERENCE_AXIS_DEG)))).sum() / total
    )
    s_magnitude = float(np.abs((weight * np.exp(2j * angle)).sum()) / total)
    return s_axis, s_magnitude


def build_track_table(
    frames: pd.DataFrame, interval_min: float, min_frames: int
) -> pd.DataFrame:
    """Aggregate frame-level rows into one row per track with QC verdicts."""
    records = []
    for track_uid, sub in frames.groupby("track_uid", sort=True):
        sub = sub.sort_values("frame")
        first, last = int(sub["frame"].iloc[0]), int(sub["frame"].iloc[-1])
        observed = int(len(sub))
        gaps = sub["frame"].diff().dropna()
        max_gap = int(gaps.max()) if len(gaps) else 1
        elapsed = (last - first) * interval_min
        # Velocity's time base is (observed frames - 1) steps, matching
        # migration_analysis.py; equals `elapsed` when the track has no gaps.
        stepping_min = (observed - 1) * interval_min
        path_length = float(sub["step_distance_um"].sum(skipna=True))
        net = float(
            np.hypot(
                sub["centroid_x_um"].iloc[-1] - sub["centroid_x_um"].iloc[0],
                sub["centroid_y_um"].iloc[-1] - sub["centroid_y_um"].iloc[0],
            )
        )
        s_axis, s_magnitude = _weighted_nematic(sub)

        status, reason = "retained", ""
        if observed < min_frames:
            status, reason = "excluded", f"track shorter than {min_frames} frames"
        elif max_gap > MAX_ALLOWED_GAP:
            status, reason = (
                "excluded",
                (f"frame gap of {max_gap} exceeds {MAX_ALLOWED_GAP}"),
            )
        elif path_length <= 0:
            status, reason = "excluded", "zero total displacement"

        records.append(
            {
                "replicate_id": sub["replicate_id"].iloc[0],
                "group": sub["group"].iloc[0],
                "source_file": sub["source_file"].iloc[0],
                "track_uid": track_uid,
                "track_id": int(sub["track_id"].iloc[0]),
                "first_frame": first,
                "last_frame": last,
                "observed_frames": observed,
                "max_frame_gap": max_gap,
                "elapsed_time_min": elapsed,
                "total_path_length_um": path_length,
                "net_displacement_um": net,
                "mean_velocity_um_per_min": (
                    path_length / stepping_min if stepping_min > 0 else np.nan
                ),
                "persistence_ratio": (net / path_length if path_length > 0 else np.nan),
                "weighted_nematic_order_axis_referenced": s_axis,
                "weighted_nematic_order_magnitude": s_magnitude,
                "mean_area_um2": float(sub["area_um2"].mean()),
                "median_area_um2": float(sub["area_um2"].median()),
                "mean_polarization_index": float(sub["polarization_index"].mean()),
                "mean_aspect_ratio": float(sub["aspect_ratio"].mean()),
                "mean_eccentricity": float(sub["eccentricity"].mean()),
                "mean_orientation_migration_alignment": float(
                    sub["orientation_migration_alignment"].mean()
                ),
                "exclusion_status": status,
                "exclusion_reason": reason,
            }
        )
    return pd.DataFrame(records)


def _weighted_mean(values: pd.Series, weights: np.ndarray) -> float:
    """Weighted mean, ignoring non-finite values or non-positive weights."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    keep = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not keep.any():
        return float("nan")
    return float(np.average(v[keep], weights=w[keep]))


def build_replicate_table(tracks: pd.DataFrame) -> pd.DataFrame:
    """Collapse retained tracks to one row per replicate."""
    kept = tracks[tracks["exclusion_status"] == "retained"]

    def _summarise(sub: pd.DataFrame) -> pd.Series:
        # Weight each track by path length x number of steps, matching the FOV
        # aggregation of the weighted nematic order in migration_analysis.py.
        weight = sub["total_path_length_um"].to_numpy(dtype=float) * (
            sub["observed_frames"].to_numpy(dtype=float) - 1.0
        )
        return pd.Series(
            {
                "number_of_retained_tracks": int(sub["track_uid"].nunique()),
                "weighted_nematic_order": _weighted_mean(
                    sub["weighted_nematic_order_axis_referenced"], weight
                ),
                "weighted_nematic_order_magnitude": _weighted_mean(
                    sub["weighted_nematic_order_magnitude"], weight
                ),
                "mean_cell_migration_velocity_um_per_min": float(
                    sub["mean_velocity_um_per_min"].mean()
                ),
                "mean_cell_area_um2": float(sub["mean_area_um2"].mean()),
                "mean_morphological_polarization": float(
                    sub["mean_polarization_index"].mean()
                ),
                "mean_persistence_ratio": float(sub["persistence_ratio"].mean()),
                "mean_orientation_migration_alignment": float(
                    sub["mean_orientation_migration_alignment"].mean()
                ),
            }
        )

    grouped = kept.groupby(["replicate_id", "group"], sort=False)
    return grouped.apply(_summarise).reset_index()


ENRICHED_COLUMNS = [
    "source_file",
    "replicate_id",
    "group",
    "frame",
    "track_id",
    "object_label",
    "centroid_x_um",
    "centroid_y_um",
    "area_um2",
    "perimeter_um",
    "major_axis_length_um",
    "minor_axis_length_um",
    "polarization_index",
    "aspect_ratio",
    "eccentricity",
    "orientation_rad",
    "orientation_deg",
]


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
        help="folder with segment_track_stack outputs (GUI picker if omitted)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="where to write the metric tables (GUI picker if omitted)",
    )
    parser.add_argument(
        "--um-per-px",
        type=float,
        default=DEFAULT_UM_PER_PX,
        help="micron-per-pixel scale",
    )
    parser.add_argument(
        "--frame-interval-min",
        type=float,
        default=DEFAULT_INTERVAL_MIN,
        help="time gap between frames in minutes",
    )
    parser.add_argument(
        "--min-track-frames",
        type=int,
        default=MIN_TRACK_FRAMES,
        help="minimum frames for a track to be retained",
    )
    parser.add_argument(
        "--reference-axis-deg",
        type=float,
        default=REFERENCE_AXIS_DEG,
        help="axis for the signed nematic order, in degrees",
    )
    parser.add_argument(
        "--max-allowed-gap",
        type=int,
        default=MAX_ALLOWED_GAP,
        help="largest tolerated frame gap within a track",
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
    """Extract label-derived migration/morphology metrics; write CSVs."""
    global REFERENCE_AXIS_DEG, MAX_ALLOWED_GAP
    args = build_arg_parser().parse_args()
    REFERENCE_AXIS_DEG = args.reference_axis_deg
    MAX_ALLOWED_GAP = args.max_allowed_gap

    input_dir = resolve_dir(
        args.input_dir, "Select segment_track_stack output folder", True
    )
    output_dir = resolve_dir(args.output_dir, "Select output folder", False)
    output_dir.mkdir(parents=True, exist_ok=True)

    group_pattern = re.compile(args.group_regex) if args.group_regex else None

    pairs = find_pairs(input_dir)
    if not pairs:
        print(
            f"FATAL: no '{LABELS_SUFFIX}' + '{TRACKING_SUFFIX}' pairs in {input_dir}",
            file=sys.stderr,
        )
        return 2

    enriched_dir = output_dir / "enriched_tracking"
    enriched_dir.mkdir(parents=True, exist_ok=True)
    enriched_parts = []
    for number, (stem, labels_path, tracking_path) in enumerate(pairs, start=1):
        print(f"[{number}/{len(pairs)}] {stem}", flush=True)
        labels = load_labels(labels_path)
        tracking = load_tracking(tracking_path)
        group = derive_group(stem, group_pattern)
        enriched = build_enriched_frames(stem, group, labels, tracking, args.um_per_px)
        if enriched.empty:
            print(f"  no objects measured for {stem}", file=sys.stderr)
            continue
        enriched.reindex(columns=ENRICHED_COLUMNS).to_csv(
            enriched_dir / f"{stem}_tracking.csv", index=False
        )
        enriched["track_uid"] = (
            enriched["replicate_id"].astype(str)
            + ":"
            + enriched["track_id"].astype(int).astype(str)
        )
        enriched_parts.append(enriched)

    if not enriched_parts:
        print("FATAL: no objects measured in any stack", file=sys.stderr)
        return 2

    frames = pd.concat(enriched_parts, ignore_index=True)
    frames = add_frame_metrics(frames, args.frame_interval_min)

    frame_columns = [
        "replicate_id",
        "group",
        "source_file",
        "frame",
        "time_min",
        "track_id",
        "object_label",
        "centroid_x_um",
        "centroid_y_um",
        "step_distance_um",
        "instantaneous_velocity_um_per_min",
        "migration_angle_deg",
        "nematic_component",
        "area_um2",
        "polarization_index",
        "aspect_ratio",
        "eccentricity",
        "orientation_deg",
        "orientation_migration_alignment",
    ]
    frames.rename(columns={"centroid_x_um": "x_um", "centroid_y_um": "y_um"}).reindex(
        columns=[
            c.replace("centroid_x_um", "x_um").replace("centroid_y_um", "y_um")
            for c in frame_columns
        ]
    ).to_csv(output_dir / "frame_cell_metrics.csv", index=False)

    tracks = build_track_table(frames, args.frame_interval_min, args.min_track_frames)
    tracks.to_csv(output_dir / "track_metrics.csv", index=False)
    tracks[tracks["exclusion_status"] == "excluded"].to_csv(
        output_dir / "exclusion_ledger.csv", index=False
    )

    replicates = build_replicate_table(tracks)
    replicates.to_csv(output_dir / "replicate_summary.csv", index=False)

    sensitivity = []
    for threshold in SENSITIVITY_THRESHOLDS:
        alt = build_track_table(frames, args.frame_interval_min, threshold)
        summary = build_replicate_table(alt)
        summary.insert(0, "min_track_frames", threshold)
        sensitivity.append(summary)
    pd.concat(sensitivity, ignore_index=True).to_csv(
        output_dir / "sensitivity_analysis.csv", index=False
    )

    if write_provenance is not None:
        write_provenance(
            output_dir,
            params={
                "input_dir": str(input_dir),
                "um_per_px": args.um_per_px,
                "frame_interval_min": args.frame_interval_min,
                "min_track_frames": args.min_track_frames,
                "reference_axis_deg": REFERENCE_AXIS_DEG,
                "max_allowed_gap": MAX_ALLOWED_GAP,
                "group_regex": args.group_regex,
                "polarization_index": "1 - minor_axis/major_axis",
                "n_stacks": len(pairs),
            },
            script_path=__file__,
            filename="extract_shape_metrics_provenance.json",
        )

    print(replicates.to_string(index=False))
    print(
        "\nretained tracks:",
        int((tracks["exclusion_status"] == "retained").sum()),
        "of",
        len(tracks),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
