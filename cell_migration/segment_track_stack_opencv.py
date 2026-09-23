""" OpenCV rendering variant of segment_track_stack.

Segmentation (Cellpose) and tracking (LapTrack) are identical to
segment_track_stack.py; only the tracking-movie generation differs: frames are
drawn directly onto the image arrays with OpenCV instead of being rasterized by
matplotlib. This is much faster and renders at the image's native resolution,
but the visual appearance (anti-aliasing, line/font rendering, contour style)
is NOT pixel-identical to the matplotlib version.

Requires opencv (add `opencv-python-headless` to environment.yml).
"""

import random
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm
from skimage.measure import regionprops_table
from scipy.ndimage import find_objects
import cv2
from matplotlib import colormaps
import imageio as iio
import nd2
from cellpose import models
from laptrack import LapTrack
import torch

try:
    from utils.provenance import write_provenance
except Exception:  # provenance is best-effort; never block analysis
    write_provenance = None

SEED = 123
np.random.seed(SEED)
random.seed(SEED)
try:
    torch.manual_seed(SEED)
except Exception:
    pass

frame_linking_distance_pixels = 23
segment_gap_closing_distance_pixels = 46
segment_gap_closing_max_frame_gap = 1
tail_length = 50

# --- Multi-channel configuration -------------------------------------------
CYTO_CHANNELS = (1, 2)          # joined -> Cellpose cytoplasm channel
NUCLEI_CHANNELS = (0, 3)        # joined -> Cellpose nuclei channel
DISPLAY_COLORS = np.array(      # RGB weights for the 4-channel movie overlay
    [[0.0, 0.35, 1.0],          # C0 - blue
     [0.0, 1.00, 0.0],          # C1 - green
     [1.0, 0.15, 0.0],          # C2 - red
     [0.8, 0.0, 1.0]],          # C3 - violet
    dtype=np.float32,
)
DISPLAY_PERCENTILES = (1.0, 99.5)
LIMIT_FRAME_SAMPLES = 8         # frames sampled to estimate intensity limits

# --- Cellpose configuration -------------------------------------------------
MODEL_TYPE = "cyto3"
USE_GPU = False                 # CLI --gpu overrides this
CELLPROB_THRESHOLD = -2.0
FLOW_THRESHOLD = 0.8
MIN_SIZE = 8

# --- Movie overlay style ----------------------------------------------------
TRACK_ALPHA = 0.8               # opacity of the track tails
CONTOUR_THICKNESS = 1
TRACK_THICKNESS = 1
MARKER_RADIUS = 3


def prompt_int_with_default(prompt, default):
    """ Precompiles the user interface with a default number """
    while True:
        s = input(f"{prompt} [{default}]: ").strip()
        if s == "":
            return default
        try:
            return int(s)
        except ValueError:
            print("Invalid integer, please try again.")


def _guess_axes(arr):
    """ Infers an axis string when the file carries no usable metadata """
    if arr.ndim == 2:
        return "YX"
    if arr.ndim == 3:
        return "TYX"
    if arr.ndim == 4:
        return "TCYX" if arr.shape[1] <= 4 else "TYXC"
    raise ValueError(f"Unsupported array with {arr.ndim} dimensions.")


def _to_tcyx(arr, axes):
    """ Standardizes any stack to the (T, C, Y, X) layout """
    axes = "".join(axes).upper().replace("S", "C")
    if "T" not in axes:                        # generic labels from headerless TIFFs
        axes = axes.replace("Q", "T", 1).replace("I", "T", 1)
    if len(axes) != arr.ndim or not {"Y", "X"} <= set(axes) or len(set(axes)) != len(axes):
        axes = _guess_axes(arr)
    for i in range(len(axes) - 1, -1, -1):     # drop singleton extra axes (Z, P, ...)
        if axes[i] not in "TCYX":
            if arr.shape[i] != 1:
                raise ValueError(f"Unsupported non-singleton axis '{axes[i]}'.")
            arr = np.squeeze(arr, axis=i)
            axes = axes[:i] + axes[i + 1:]
    for ax in ("C", "T"):
        if ax not in axes:
            arr, axes = arr[np.newaxis], ax + axes
    return np.transpose(arr, [axes.index(a) for a in "TCYX"])


def _read_stack(in_file):
    """ Reads ND2/TIF/TIFF and returns a (T, C, Y, X) array """
    if in_file.suffix.lower() == ".nd2":
        with nd2.ND2File(str(in_file)) as f:
            return _to_tcyx(f.asarray(), list(f.sizes.keys()))
    with tifffile.TiffFile(str(in_file)) as f:
        series = f.series[0]
        return _to_tcyx(series.asarray(), series.axes)


def _stack_limits(stack):
    """ Per-channel display/normalization limits from sub-sampled frames """
    step = max(1, stack.shape[0] // LIMIT_FRAME_SAMPLES)
    sample = stack[::step]
    return np.stack(
        [np.percentile(sample[:, c], DISPLAY_PERCENTILES) for c in range(stack.shape[1])]
    ).astype(np.float32)


def _rescale(plane, limits):
    """ Linear rescale of one plane to [0, 1] using precomputed limits """
    lo, hi = float(limits[0]), float(limits[1])
    return np.clip((plane.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def _merge_channels(frame, indices, limits):
    """ Joins channels by maximum intensity after per-channel rescaling """
    merged = _rescale(frame[indices[0]], limits[indices[0]])
    for c in indices[1:]:
        np.maximum(merged, _rescale(frame[c], limits[c]), out=merged)
    return merged


def _segmentation_input(frame, limits):
    """ Builds the Cellpose input plane(s) and the matching channel spec """
    if frame.shape[0] == 1:
        return frame[0], [0, 0]
    img = np.zeros(frame.shape[1:] + (3,), dtype=np.float32)
    img[..., 0] = _merge_channels(frame, CYTO_CHANNELS, limits)
    img[..., 1] = _merge_channels(frame, NUCLEI_CHANNELS, limits)
    return img, [1, 2]


def _composite_rgb(frame, limits):
    """ Additive RGB composite of all channels for the tracking movie """
    rgb = np.zeros(frame.shape[1:] + (3,), dtype=np.float32)
    for c in range(frame.shape[0]):
        rgb += _rescale(frame[c], limits[c])[..., None] * DISPLAY_COLORS[c]
    return np.clip(rgb, 0.0, 1.0)


def _resolve_gpu():
    """ Validates the GPU request against the available torch backend """
    if not USE_GPU:
        return False
    if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        return True
    print("Warning: GPU requested but no CUDA device is visible; running on CPU.")
    return False


def _build_model():
    """ Builds the Cellpose model on the resolved device """
    gpu = _resolve_gpu()
    print(f"Model: {MODEL_TYPE} on {'GPU' if gpu else 'CPU'}")
    return models.Cellpose(gpu=gpu, model_type=MODEL_TYPE)


def _eval_masks(model, img, diameter, channels):
    """ Runs Cellpose with the fixed thresholds, tolerant to API changes """
    kwargs = dict(
        diameter=diameter,
        channels=channels,
        flow_threshold=FLOW_THRESHOLD,
        cellprob_threshold=CELLPROB_THRESHOLD,
        min_size=MIN_SIZE,
    )
    try:
        out = model.eval(img, **kwargs)
    except TypeError:
        kwargs.pop("channels")
        out = model.eval(img, **kwargs)
    return np.asarray(out[0], dtype=np.uint16)


def _cmap_rgb8(cmap, i, n):
    """ RGB uint8 tuple sampled from a matplotlib colormap """
    return tuple(int(round(v * 255)) for v in cmap(i / max(1, n - 1))[:3])


def _base_frame(frame_channels, num_channels, limits, vmin, vmax):
    """ Builds the 8-bit RGB background for one movie frame """
    if num_channels == 1:
        norm = np.clip(
            (frame_channels[0].astype(np.float32) - vmin) / max(vmax - vmin, 1e-6),
            0.0, 1.0,
        )
        gray = (norm * 255.0).astype(np.uint8)
        rgb = np.repeat(gray[:, :, None], 3, axis=2)
    else:
        rgb = (_composite_rgb(frame_channels, limits) * 255.0).astype(np.uint8)
    return np.ascontiguousarray(rgb)


def _draw_contours(frame, frame_labels, label_colors):
    """ Draws each cell's outline; bounding-box crop keeps it fast """
    for idx, sl in enumerate(find_objects(frame_labels)):
        if sl is None:
            continue
        label_id = idx + 1
        y_sl, x_sl = sl
        sub = (frame_labels[y_sl, x_sl] == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        color = label_colors.get(label_id, (255, 255, 255))
        cv2.drawContours(
            frame, contours, -1, color, CONTOUR_THICKNESS,
            lineType=cv2.LINE_AA, offset=(int(x_sl.start), int(y_sl.start)),
        )


def _draw_tracks(frame, t, track_ids, track_arrays, track_colors):
    """ Draws track tails (alpha-blended) and current-position markers """
    overlay = frame.copy()
    drew_tail = False
    for track_id in track_ids:
        times, xs, ys = track_arrays[track_id]
        in_tail = (times <= t) & (times >= t - tail_length)
        if not in_tail.any():
            continue
        pts = np.column_stack((xs[in_tail], ys[in_tail]))
        pts = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        if len(pts) >= 2:
            cv2.polylines(
                overlay, [pts], False, track_colors[track_id],
                TRACK_THICKNESS, lineType=cv2.LINE_AA,
            )
            drew_tail = True
    if drew_tail:
        cv2.addWeighted(overlay, TRACK_ALPHA, frame, 1.0 - TRACK_ALPHA, 0.0, dst=frame)

    # Current positions on top, at full opacity (matches the opaque matplotlib marker)
    for track_id in track_ids:
        times, xs, ys = track_arrays[track_id]
        at_t = times == t
        if at_t.any():
            cx = int(round(float(xs[at_t][0])))
            cy = int(round(float(ys[at_t][0])))
            cv2.circle(
                frame, (cx, cy), MARKER_RADIUS, track_colors[track_id],
                TRACK_THICKNESS, lineType=cv2.LINE_AA,
            )


def _draw_scalebar_and_time(frame, t, width, height, um_per_px, time_gap_min):
    """ White 100 µm scale bar and a timestamp with a dark background box """
    margin = int(0.02 * height)
    bar_px = 100.0 / float(um_per_px)  # 100 µm converted to pixels
    bar_th = max(6, int(0.006 * height))  # ~0.6% of height
    x_end = width - margin
    x_start = int(round(x_end - bar_px))
    y_bottom = height - margin
    y_top = y_bottom - bar_th
    cv2.rectangle(
        frame, (x_start, y_top), (int(round(x_end)), y_bottom),
        (255, 255, 255), thickness=-1,
    )

    label = f"{int(round(t * float(time_gap_min)))} min"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, height / 1000.0)
    thickness = max(1, int(round(height / 600.0)))
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    pad = max(2, int(round(0.004 * height)))
    x0, y0 = margin, margin
    cv2.rectangle(
        frame, (x0, y0), (x0 + tw + 2 * pad, y0 + th + baseline + 2 * pad),
        (0, 0, 0), thickness=-1,
    )
    cv2.putText(
        frame, label, (x0 + pad, y0 + th + pad), font, font_scale,
        (255, 255, 255), thickness, lineType=cv2.LINE_AA,
    )


def segment_and_track(folder, diameter, um_per_px, time_gap_min):
    """ Segments via CellPose and tracks migration via LapTrack """
    model = _build_model()
    folder = Path(folder)
    # Iterate over ND2 and TIFF files
    input_files = []
    input_files += sorted(Path(folder).glob("*.nd2"))
    input_files += sorted(Path(folder).glob("*.tif"))
    input_files += sorted(Path(folder).glob("*.tiff"))

    if not input_files:
        print("No ND2/TIF/TIFF files found.")
        return

    for in_file in input_files:
        print(f"Processing {in_file.name}")
        image_stack = _read_stack(in_file)
        num_frames, num_channels, height, width = image_stack.shape
        if num_channels not in (1, 4):
            print(f"Skipped: {num_channels} channels found, expected 1 or 4.")
            continue
        print(f"Loaded {num_frames} frames, {num_channels} channel(s), {height}x{width} px")

        limits = _stack_limits(image_stack)
        labels_stack = np.zeros((num_frames, height, width), dtype=np.uint16)

        for frame_idx in tqdm(range(num_frames), desc="Segmenting Frames"):
            seg_img, channels = _segmentation_input(image_stack[frame_idx], limits)
            labels_stack[frame_idx] = _eval_masks(model, seg_img, diameter, channels)

        output_tif = folder / f"{in_file.stem}_cellpose_labels.tif"
        tifffile.imwrite(
            str(output_tif), labels_stack, dtype=np.uint16, compression="zlib"
        )
        print(f"Saved {output_tif}")

        regionprops_frames = []
        for frame, label in enumerate(labels_stack):
            df = pd.DataFrame(
                regionprops_table(label, properties=["label", "centroid"])
            )
            df["frame"] = frame
            regionprops_frames.append(df)
        regionprops_df = pd.concat(regionprops_frames)

        if "frame_y" in regionprops_df.columns:
            regionprops_df.rename(columns={"frame_y": "frame"}, inplace=True)

        lt = LapTrack(
            cutoff=frame_linking_distance_pixels**2,
            gap_closing_cutoff=segment_gap_closing_distance_pixels**2,
            gap_closing_max_frame_count=segment_gap_closing_max_frame_gap,
        )
        track_df, _, _ = lt.predict_dataframe(
            regionprops_df.copy(),
            coordinate_cols=["centroid-0", "centroid-1"],
            only_coordinate_cols=False,
        )
        track_df = track_df.reset_index()
        output_csv = folder / f"{in_file.stem}_tracking.csv"
        track_df.to_csv(output_csv)

        # Movie Generation (OpenCV)
        print("Generating video...")

        track_data = track_df[["track_id", "frame", "centroid-0", "centroid-1"]].copy()
        track_data.columns = ["track_id", "time", "y", "x"]

        vmin, vmax = float(limits[0][0]), float(limits[0][1])

        label_ids = np.unique(labels_stack)
        label_ids = label_ids[label_ids != 0]
        label_cmap = colormaps.get_cmap("tab20")
        label_colors = {
            int(label_id): _cmap_rgb8(label_cmap, i, len(label_ids))
            for i, label_id in enumerate(label_ids)
        }

        track_ids = track_data["track_id"].unique()
        track_cmap = colormaps.get_cmap("hsv")
        track_colors = {
            track_id: _cmap_rgb8(track_cmap, i, len(track_ids))
            for i, track_id in enumerate(track_ids)
        }

        # Pre-group each track's coordinates into NumPy arrays once (original
        # row order preserved), so per-frame drawing is a cheap array slice
        # instead of a full-DataFrame scan per track per frame.
        track_arrays = {}
        for track_id, group in track_data.groupby("track_id", sort=False):
            track_arrays[track_id] = (
                group["time"].to_numpy(),
                group["x"].to_numpy(),
                group["y"].to_numpy(),
            )

        output_video = folder / f"{in_file.stem}_tracking.mov"
        writer = iio.get_writer(
            str(output_video),
            fps=4,
            format="FFMPEG",
            codec="libx264",
            macro_block_size=None,
            ffmpeg_params=["-crf", "18", "-preset", "slow", "-loglevel", "error"],
        )

        # H.264 needs even dimensions; pad the bottom/right edge by 1 px if odd.
        out_h = height + (height % 2)
        out_w = width + (width % 2)

        for t in tqdm(range(num_frames), desc="Rendering Movie"):
            frame = _base_frame(image_stack[t], num_channels, limits, vmin, vmax)
            _draw_contours(frame, labels_stack[t], label_colors)
            _draw_tracks(frame, t, track_ids, track_arrays, track_colors)
            _draw_scalebar_and_time(frame, t, width, height, um_per_px, time_gap_min)

            if out_h != height or out_w != width:
                padded = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                padded[:height, :width] = frame
                frame = padded
            writer.append_data(frame)

        writer.close()
        print(f"Video saved to {output_video}")

    if write_provenance is not None:
        write_provenance(
            folder,
            params={
                "diameter_px": diameter,
                "um_per_px": um_per_px,
                "time_gap_min": time_gap_min,
                "frame_linking_distance_pixels": frame_linking_distance_pixels,
                "segment_gap_closing_distance_pixels": segment_gap_closing_distance_pixels,
                "segment_gap_closing_max_frame_gap": segment_gap_closing_max_frame_gap,
                "tail_length": tail_length,
                "cellpose_model": MODEL_TYPE,
                "use_gpu": USE_GPU,
                "flow_threshold": FLOW_THRESHOLD,
                "cellprob_threshold": CELLPROB_THRESHOLD,
                "min_size": MIN_SIZE,
                "cyto_channels": list(CYTO_CHANNELS),
                "nuclei_channels": list(NUCLEI_CHANNELS),
                "renderer": "opencv",
            },
            script_path=__file__,
            filename="segment_track_provenance.json",
        )


def main():
    """ Main function """
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select Folder Containing Stacks")
    root.destroy()
    if not folder:
        print("No folder selected.")
        return

    try:
        diameter = float(input("Enter cell diameter (in pixels): "))
        um_per_px = float(input("Enter micron-per-pixel scale (µm/px): "))
        time_gap_min = float(input("Enter time gap between frames (minutes): "))
    except ValueError:
        print("Invalid numeric input.")
        return

    global frame_linking_distance_pixels, segment_gap_closing_distance_pixels
    global segment_gap_closing_max_frame_gap, tail_length

    frame_linking_distance_pixels = prompt_int_with_default(
        "Set maximum distance covered by a cell between consecutive frames",
        frame_linking_distance_pixels,
    )

    segment_gap_closing_max_frame_gap = prompt_int_with_default(
        "Set maximum number of frames for re-linking the track after missing detections",
        segment_gap_closing_max_frame_gap,
    )

    segment_gap_closing_distance_pixels = prompt_int_with_default(
        "Set maximum distance covered by a cell over gaps",
        segment_gap_closing_distance_pixels,
    )

    tail_length = prompt_int_with_default(
        "Set number of frames for track permanence", tail_length
    )

    try:
        segment_and_track(folder, diameter, um_per_px, time_gap_min)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
