import os
from glob import glob
import pandas as pd
import datetime

try:
    from utils.provenance import write_provenance, sha256_file
except Exception:  # provenance is best-effort; never block analysis
    write_provenance = None
    sha256_file = None

__version__ = "1.0.0"
MERGE_NAME = "cell_area_merge"
try:
    MERGE_SHA256 = sha256_file(__file__) if sha256_file else "unknown"
except Exception:
    MERGE_SHA256 = "unknown"
MERGE_INFO = f"{MERGE_NAME}|{__version__}|{MERGE_SHA256[:12]}|{datetime.datetime.now(datetime.timezone.utc).isoformat()}"

flat_15 = glob("*/CSVs/*15_flat*.csv")
flat_75 = glob("*/CSVs/*75_flat*.csv")
cntrl = glob("*/CSVs/*CNTRL*.csv")
file_lists = [flat_15, flat_75, cntrl]
conditions = ["Stiffer gels", "Softer gels", "Control"]
condition_indices = [3, 2, 1]
all_data = []
for file_list, condition, condition_idx in zip(
    file_lists, conditions, condition_indices
):
    for filename in file_list:
        path_list = filename.split(os.sep)
        path = path_list[0]
        name = path_list[-1]

        data = pd.read_csv(filename)
        data["condition_index"] = condition_idx
        experiment = None
        if "250330" in path:
            experiment = 1
        elif "250331" in path:
            experiment = 2
        elif "250403" in path:
            experiment = 3
        else:
            raise ValueError(f"Experiment {path} not found!")
        data["experiment_idx"] = experiment
        data["FOV"] = int(name.split("_")[0])
        data["condition"] = condition
        data["file_name"] = name.replace(".csv", "")
        data["experiment"] = path
        all_data.append(data)

# merge dataframes
all_data = pd.concat(all_data)
# sort FOVs
min_fov = 1
for fov in sorted(all_data["FOV"].unique()):
    print(fov)
    all_data.loc[all_data["FOV"] == fov, "FOV"] = min_fov
    min_fov += 1

all_data["merge_info"] = MERGE_INFO
all_data.to_csv("data_ligHTS_cell_area.csv", index=False)

if write_provenance is not None:
    write_provenance(
        ".",
        params={
            "output": "data_ligHTS_cell_area.csv",
            "n_input_csvs": len(flat_15) + len(flat_75) + len(cntrl),
        },
        script_path=__file__,
        filename="merge_provenance.json",
    )
