# CLI for image cell tracking with argparse help.

import argparse
import importlib
import sys
from pathlib import Path

BASE_MODULE = "segment_track_stack"

# CLI flag -> module attribute overridden before the run
OVERRIDES = {
    "link_dist": "frame_linking_distance_pixels",
    "gap_dist": "segment_gap_closing_distance_pixels",
    "gap_frames": "segment_gap_closing_max_frame_gap",
    "tail": "tail_length",
    "cellprob_threshold": "CELLPROB_THRESHOLD",
    "flow_threshold": "FLOW_THRESHOLD",
    "min_size": "MIN_SIZE",
    "gpu": "USE_GPU",
    "model_type": "MODEL_TYPE",
    "cyto_channels": "CYTO_CHANNELS",
    "nuclei_channels": "NUCLEI_CHANNELS",
}


def channel_list(text):
    """ Parses a comma-separated list of channel indices in [0, 3] """
    try:
        idx = tuple(int(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' is not a comma-separated int list.")
    if not idx or len(set(idx)) != len(idx) or not all(0 <= v <= 3 for v in idx):
        raise argparse.ArgumentTypeError(f"'{text}' must be unique indices in 0-3.")
    return idx


def parse_args(mod):
    """ Argument parser; defaults are read from the pipeline module """
    p = argparse.ArgumentParser(
        description="stack segmentation and tracking (1- or 4-channel stacks)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--folder", type=str, help="Folder containing stack files")
    p.add_argument("--diameter", type=float, default=30.0,
                   help="Approximate cell diameter in pixels")
    p.add_argument("--um_per_px", type=float, default=1.34,
                   help="micron-per-pixel scale")
    p.add_argument("--time_gap_min", type=float, default=15.0,
                   help="time gap between frames in minutes")
    p.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=mod.USE_GPU,
                   help="run Cellpose on the GPU (falls back to CPU if unavailable)")

    g = p.add_argument_group("tracking")
    g.add_argument("--link_dist", type=int, default=mod.frame_linking_distance_pixels,
                   help="max distance between consecutive frames, px")
    g.add_argument("--gap_dist", type=int,
                   default=mod.segment_gap_closing_distance_pixels,
                   help="max distance covered over gaps, px")
    g.add_argument("--gap_frames", type=int,
                   default=mod.segment_gap_closing_max_frame_gap,
                   help="max frame gap for re-linking a track")
    g.add_argument("--tail", type=int, default=mod.tail_length,
                   help="frames of track permanence in the movie")

    s = p.add_argument_group("segmentation")
    s.add_argument("--cellprob_threshold", type=float, default=mod.CELLPROB_THRESHOLD,
                   help="Cellpose cell probability threshold")
    s.add_argument("--flow_threshold", type=float, default=mod.FLOW_THRESHOLD,
                   help="Cellpose flow error threshold")
    s.add_argument("--min_size", type=int, default=mod.MIN_SIZE,
                   help="minimum mask size in px")
    s.add_argument("--model_type", type=str, default=mod.MODEL_TYPE,
                   help="Cellpose model")
    s.add_argument("--cyto_channels", type=channel_list,
                   default=mod.CYTO_CHANNELS,
                   help="4-channel input: channels joined as cytoplasm")
    s.add_argument("--nuclei_channels", type=channel_list,
                   default=mod.NUCLEI_CHANNELS,
                   help="4-channel input: channels joined as nuclei")
    return p.parse_args()


def pick_folder():
    """ GUI folder selection used when --folder is omitted """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select Folder Containing Stacks")
    root.destroy()
    return folder or None


def resolve_folder(arg_folder):
    """ Returns a validated folder path or None """
    folder = arg_folder or pick_folder()
    if not folder:
        print("No folder selected.", file=sys.stderr)
        return None
    path = Path(folder).expanduser()
    if not path.is_dir():
        print(f"Not a directory: {path}", file=sys.stderr)
        return None
    return path


def apply_overrides(mod, args):
    """ Pushes CLI values onto the pipeline module and echoes them """
    for flag, attr in OVERRIDES.items():
        setattr(mod, attr, getattr(args, flag))
        print(f"  {attr} = {getattr(mod, attr)}")


def main():
    """ Main function """
    try:
        mod = importlib.import_module(BASE_MODULE)
    except Exception as e:
        print(f"Error: could not import {BASE_MODULE}: {e}", file=sys.stderr)
        return 2

    args = parse_args(mod)
    folder = resolve_folder(args.folder)
    if folder is None:
        return 2
    if set(args.cyto_channels) & set(args.nuclei_channels):
        print("Error: cyto and nuclei channel sets overlap.", file=sys.stderr)
        return 2

    print(f"Folder: {folder}")
    print(f"  diameter = {args.diameter} px, scale = {args.um_per_px} um/px, "
          f"dt = {args.time_gap_min} min")
    apply_overrides(mod, args)

    try:
        mod.segment_and_track(
            str(folder), args.diameter, args.um_per_px, args.time_gap_min
        )
    except Exception as e:
        print(f"Error: run failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
