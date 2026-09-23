import os
import pandas as pd
import numpy as np
from aicsimageio import AICSImage
from cellpose import models
from skimage.io import imsave
from skimage.measure import regionprops_table
from csbdeep.utils import Path, normalize
import matplotlib.pyplot as plt
from glob import glob
import torch
import datetime

try:
    from utils.provenance import write_provenance, sha256_file
except Exception:  # provenance is best-effort; never block analysis
    write_provenance = None
    sha256_file = None

# Global Variables - Changing the default values does not significatively increase accuracy
flow_threshold = 0.4
cellprob_threshold = 0.0
tile_norm_blocksize = 0

# Provenance / auditability
__version__ = "1.0.0"
ANALYSIS_NAME = "cell_area_segment"
try:
    ANALYSIS_SHA256 = sha256_file(__file__) if sha256_file else "unknown"
except Exception:
    ANALYSIS_SHA256 = "unknown"
ANALYSIS_TIMESTAMP = datetime.datetime.now(datetime.timezone.utc).isoformat()
ANALYSIS_INFO = f"{ANALYSIS_NAME}|{__version__}|{ANALYSIS_SHA256[:12]}|{ANALYSIS_TIMESTAMP}"

# Check if GPU is available
use_gpu = torch.cuda.is_available()
if use_gpu:
    print("GPU detected and will be used for segmentation")
else:
    print("No GPU detected, using CPU for segmentation (this will be slower)")

filenames = glob("*_filtered/*.nd2")
print(f"\nFound {len(filenames)} files to process\n")
model = models.CellposeModel(gpu=use_gpu)


def plot_masks_next_to_image(image, masks, img_file, plot_dir, save_masks=True):
    """ QC by plotting mask next to the original image for comparison """
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Plot reference image (left)
    image = normalize(np.moveaxis(image, 0, -1), axis=(0, 1), pmin=1)
    image = np.clip(image, 0, 1)
    ax1.imshow(image)
    ax1.set_title("Image", fontsize=14)
    ax1.axis("off")

    # Plot masks (right)
    ax2.imshow(masks, cmap="tab20", interpolation="nearest")
    ax2.set_title("Segmentation", fontsize=14)
    ax2.axis("off")

    # Adjust spacing
    plt.tight_layout()

    # Save the plot
    output_filename = os.path.join(plot_dir, f"{Path(img_file).stem}.png")
    plt.savefig(output_filename, bbox_inches="tight")

    # Close the figure to free memory
    plt.close(fig)
    if save_masks:
        output_filename = os.path.join(plot_dir, f"masks_2ch_{Path(img_file).stem}.tif")
        imsave(output_filename, masks, compression="zlib", check_contrast=False)

# Main
for idx, filename in enumerate(filenames, 1):
    print(f"[{idx}/{len(filenames)}] Processing: {filename}")
    img = AICSImage(filename)

    x = img.get_image_data("CYX")
    masks, _, _ = model.eval(
        x[1:3, ...],
        batch_size=32,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        normalize={"tile_norm_blocksize": tile_norm_blocksize},
    )

    outfilename = Path(filename).stem.replace(" ", "").replace(".nd2", "")
    outdir = os.path.join(Path(filename).parent, "figures")
    plot_masks_next_to_image(x, masks, outfilename, outdir)

    # export regionprops
    props = regionprops_table(
        masks,
        properties=["label", "area"],
        spacing=(img.physical_pixel_sizes.Y, img.physical_pixel_sizes.X),
    )
    data = pd.DataFrame(props)
    data["analysis_info"] = ANALYSIS_INFO
    outdir = os.path.join(Path(filename).parent, "CSVs")
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    data.to_csv(os.path.join(outdir, outfilename + ".csv"), index=False)


if write_provenance is not None:
    write_provenance(
        ".",
        params={
            "flow_threshold": flow_threshold,
            "cellprob_threshold": cellprob_threshold,
            "tile_norm_blocksize": tile_norm_blocksize,
            "channels": "x[1:3] (actin + tubulin)",
            "cellpose_model": "CellposeModel (Cellpose-SAM)",
            "n_files": len(filenames),
        },
        script_path=__file__,
        filename="segmentation_provenance.json",
    )
