import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

try:
    from utils.provenance import write_provenance
except Exception:  # provenance is best-effort; never block analysis
    write_provenance = None

SEED = 123
np.random.seed(SEED)

flat_15 = glob("*/CSVs/*15_flat*.csv")
flat_75 = glob("*/CSVs/*75_flat*.csv")
cntrl = glob("*/CSVs/*CNTRL*.csv")


def plot_histograms(file_list: list):
    """ Plot comprehensive histogram from merged CSVs """
    all_data = []
    for filename in file_list:
        path_list = filename.split(os.sep)
        path = path_list[0]
        name = path_list[-1]
        out_dir = os.path.join(path, "histograms")
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        data = pd.read_csv(filename)
        all_data.append(data)

        plt.hist(data["area"])
        plt.xlabel(r"Area / $\mu m^2$")
        plt.ylabel("Frequency / arb. u.")
        plt.savefig(os.path.join(out_dir, name.replace(".csv", ".pdf")))
        plt.close()

    all_data = pd.concat(all_data)
    plt.hist(all_data["area"], bins=30)
    plt.xlabel(r"Area / $\mu m^2$")
    plt.ylabel("Frequency / arb. u.")
    all_plot_name = "histogram_area_"
    if all_data["area"].max() > 8000:
        print("Found very large cells in :")
        print(filename)

    if "15_flat" in filename:
        all_plot_name += "15_flat"
    elif "75_flat" in filename:
        all_plot_name += "75_flat"
    elif "CNTRL" in filename:
        all_plot_name += "CNTRL"
    all_plot_name += ".pdf"
    plt.tight_layout()
    plt.savefig(all_plot_name)
    plt.close()
    return all_data


all_15 = plot_histograms(flat_15)
all_75 = plot_histograms(flat_75)
all_cntrl = plot_histograms(cntrl)
labels = ["Control", "Stiffer gels", "Softer gels"]
all_data = [
    all_cntrl["area"].to_numpy(),
    all_15["area"].to_numpy(),
    all_75["area"].to_numpy(),
]
for data, label in zip(all_data, labels):
    print(f"Median for {label}: {np.median(data):.2f}")

bp = plt.boxplot(all_data, labels=labels)
for idx, _ in enumerate(all_data):
    x = np.random.normal(idx + 1, 0.08, size=len(all_data[idx]))
    plt.plot(x, all_data[idx], ".", color="grey", alpha=0.2)
plt.ylim(None, 7000)
plt.ylabel(r"Area / $\mu m^2$")
plt.tight_layout()
plt.savefig("overview_experiment.pdf")
plt.close()

if write_provenance is not None:
    write_provenance(
        ".",
        params={"outputs": "per-condition histograms + overview_experiment.pdf"},
        script_path=__file__,
        filename="histograms_provenance.json",
    )
