import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from utils.provenance import write_provenance
except Exception:  # provenance is best-effort; never block analysis
    write_provenance = None

# Global data structures
folder_track_data = {}
tag_colors = {}



def tag_key(tag):
    """ Sort-key for tags: digits first (numeric), then lowercase, then uppercase, then others """
    if tag.isdigit():
        return (0, int(tag))
    elif tag.islower():
        return (1, tag)
    elif tag.isupper():
        return (2, tag)
    return (3, tag)



def sort_tags(tags):
    """ Custom sorting function for tags """
    return sorted(tags, key=tag_key)


# Strong colorblind-friendly palette
strong_colorblind_palette = sns.color_palette("colorblind")



def assign_colors_iteratively(existing_colors, new_tags):
    """ Assign colors iteratively """
    palette = strong_colorblind_palette * (
        (len(existing_colors) + len(new_tags)) // len(strong_colorblind_palette) + 1
    )
    color_index = len(existing_colors)
    for tag in new_tags:
        if tag not in existing_colors:
            existing_colors[tag] = palette[color_index]
            color_index += 1
    return existing_colors



def has_large_gap(track_df):
    """ Quality control on tracks """
    return (track_df["frame"].diff().dropna() > 2).any()



def process_csv(file_path):
    """ Function to read and process each CSV file containing migration info """
    df = pd.read_csv(file_path)
    str_marker = "sorted"

    if str_marker not in file_path:
        df = df[["frame", "track_id", "centroid-0", "centroid-1"]].rename(
            columns={"centroid-0": "Y", "centroid-1": "X"}
        )
    else:
        df = df[["frame", "track_id", "X", "Y"]]

    df.sort_values(by=["track_id", "frame"], inplace=True)
    df["Displacement"] = np.sqrt(
        df.groupby("track_id")["X"].diff() ** 2
        + df.groupby("track_id")["Y"].diff() ** 2
    )
    df["Angle"] = np.degrees(
        np.arctan2(
            df.groupby("track_id")["Y"].diff(), df.groupby("track_id")["X"].diff()
        )
    ).mod(360)
    df["Nematic_Order"] = np.cos(2 * np.radians(df["Angle"] - 90))

    track_sizes = df.groupby("track_id").size()
    valid_tracks = track_sizes[track_sizes >= 8].index
    df = df[df["track_id"].isin(valid_tracks)]

    invalid_tracks = df.groupby("track_id").filter(has_large_gap)["track_id"].unique()
    df = df[~df["track_id"].isin(invalid_tracks)]

    df["X_normalized"] = df.groupby("track_id")["X"].transform(lambda x: x - x.iloc[0])
    df["Y_normalized"] = df.groupby("track_id")["Y"].transform(lambda y: y - y.iloc[0])

    return df



def create_track_movement_tiff(df, output_path, tag):
    """ Create plot of normalized track movements """
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.tick_params(width=2, labelsize=12)
    for spine in ax.spines.values():
        spine.set_linewidth(3)

    for track_id, group in df.groupby("track_id"):
        plt.plot(
            group["X_normalized"],
            group["Y_normalized"],
            marker="o",
            linestyle="-",
            linewidth=1,
            markersize=2,
            alpha=0.5,
            color=tag_colors.get(tag, "gray"),
        )

    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlim([-350, 350])
    plt.ylim([-350, 350])
    plt.xlabel(
        "X Displacement (Normalized)", fontweight="bold", fontsize=12, family="Arial"
    )
    plt.ylabel(
        "Y Displacement (Normalized)", fontweight="bold", fontsize=12, family="Arial"
    )
    plt.title(
        f"Normalized Tracks: {tag}", fontweight="bold", fontsize=12, family="Arial"
    )
    plt.grid(True)

    movement_tiff_path = output_path.replace(".csv", "_track_movement.tiff")
    plt.savefig(movement_tiff_path)
    plt.close()



def create_nematic_order_plot(df, um_per_pix, min_frame, output_path):
    """ Plot histogram and compute metrics for weighted nematic order and mean velocity per track """
    disp_t = df.groupby("track_id")["Displacement"].sum()
    nframe = (df.groupby("track_id")["Displacement"].size() - 1).clip(lower=1)

    # Discard tracks with zero total displacement
    valid_tracks = disp_t > 0
    disp_t = disp_t[valid_tracks]
    nframe = nframe[valid_tracks]

    df = df[df["track_id"].isin(disp_t.index)]

    weighted_nematic_order_per_track = (
        df["Nematic_Order"] * df["Displacement"]
    ).groupby(df["track_id"]).sum() / disp_t

    # Plot histogram of weighted nematic order
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.tick_params(width=2, labelsize=12)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    counts, bins = np.histogram(weighted_nematic_order_per_track)
    plt.hist(bins[:-1], bins, weights=counts)
    plt.xlim([-1, 1])
    plt.xlabel("Weighted Nematic Order", fontweight="bold", fontsize=12, family="Arial")
    plt.ylabel("Frequency", fontweight="bold", fontsize=12, family="Arial")
    plt.title(
        "Frequency of Nematic Order Weighted on Track",
        fontweight="bold",
        fontsize=12,
        family="Arial",
    )
    plt.grid(True)

    nematic_tiff_path = output_path.replace(".csv", "_nematic.tiff")
    plt.savefig(nematic_tiff_path)
    plt.close()

    # Compute summary metrics per FOV
    weighted_nematic_order_fov = (
        weighted_nematic_order_per_track * (disp_t * nframe)
    ).sum() / (disp_t * nframe).sum()
    wno_rms_fov = np.sqrt(
        (
            (weighted_nematic_order_per_track - weighted_nematic_order_fov) ** 2
            * (disp_t * nframe)
        ).sum()
        / (disp_t * nframe).sum()
    )
    disp_t_um = disp_t * um_per_pix
    meanvel_per_track = disp_t_um / (nframe * min_frame)
    meanvel_fov = meanvel_per_track.mean()
    meanvel_sd_fov = meanvel_per_track.std()
    track_id = list(range(1, len(meanvel_per_track) + 1))

    return (
        track_id,
        weighted_nematic_order_per_track,
        meanvel_per_track,
        disp_t_um,
        nframe,
        weighted_nematic_order_fov,
        wno_rms_fov,
        meanvel_fov,
        meanvel_sd_fov,
    )



def create_combined_track_plot_per_tag(folder_name, output_path):
    """ Plot combined normalized tracks by substrate """
    global tag_colors
    plt.rcParams.update({"font.size": 12, "font.family": "Arial"})

    for tag, df_list in folder_track_data[folder_name].items():
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        ax.tick_params(width=2, labelsize=12)
        for spine in ax.spines.values():
            spine.set_linewidth(3)

        for df in df_list:
            for _, group in df.groupby("track_id"):
                plt.plot(
                    group["X_normalized"],
                    group["Y_normalized"],
                    linestyle="-",
                    linewidth=1,
                    alpha=0.5,
                    color=tag_colors.get(tag, "gray"),
                )

        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        plt.axvline(0, color="gray", linestyle="--", linewidth=1)
        plt.xlim([-350, 350])
        plt.ylim([-350, 350])
        plt.xlabel(
            "X Displacement (Normalized)",
            fontweight="bold",
            fontsize=12,
            family="Arial",
        )
        plt.ylabel(
            "Y Displacement (Normalized)",
            fontweight="bold",
            fontsize=12,
            family="Arial",
        )
        plt.title(
            f"Combined Normalized Tracks - Tag: {tag}",
            fontweight="bold",
            fontsize=12,
            family="Arial",
        )
        plt.grid(True)

        plt.savefig(
            os.path.join(output_path, f"{folder_name}_combined_track_plot_{tag}.tiff"),
            bbox_inches="tight",
        )
        plt.close()



def create_summary_plots(plotdf, folder_path, folder_name, flag):
    """ Create summary plots for each folder or recap ones combining all folders """
    plotdf["TAG"] = pd.Categorical(
        plotdf["TAG"], categories=sort_tags(plotdf["TAG"].unique()), ordered=True
    )

    for metric, ylabel in [
        ("WNO", "Cell migration\nWeighted Nematic Order [a.u.]"),
        ("MeanVel", "Mean Cell Displacement [μm/min]"),
    ]:
        plt.figure(figsize=(2.76, 2.76))
        plt.rcParams.update(
            {"font.size": 12, "font.family": "Arial", "font.weight": "bold"}
        )
        ax = plt.gca()
        ax.tick_params(width=2, labelsize=12)
        ax.spines["top"].set_color("none")
        ax.spines["right"].set_color("none")

        if not flag:
            g = sns.catplot(
                data=plotdf,
                kind="point",
                x="TAG",
                y=metric,
                dodge=False,
                color="#707272",
                linestyle="none",
                marker="_",
                markersize=50,
                errorbar="sd",
                capsize=0.15,
                err_kws={"linewidth": 1, "alpha": 0.7},
                legend=False,
            )
            sns.swarmplot(
                data=plotdf,
                x="TAG",
                y=metric,
                hue="TAG",
                dodge=False,
                legend=False,
                size=5,
                ax=g.ax,
            )
        else:
            g = sns.catplot(
                data=plotdf,
                kind="point",
                x="TAG",
                y=metric,
                dodge=False,
                color="#707272",
                linestyle="none",
                marker="_",
                markersize=50,
                errorbar="sd",
                capsize=0.15,
                err_kws={"linewidth": 1, "alpha": 0.7},
                legend=False,
            )
            sns.swarmplot(
                data=plotdf,
                x="TAG",
                y=metric,
                hue="FOLDER",
                dodge=False,
                legend=flag,
                size=5,
                ax=g.ax,
            )

        plt.ylabel(ylabel, weight="bold", fontsize=12, family="Arial")
        plt.xlabel("Pitch design [µm]", weight="bold", fontsize=12, family="Arial")
        plt.ylim([0, 1])
        if metric == "WNO":
            plt.yticks(np.linspace(0, 1, 5))


        plt.savefig(
            os.path.join(folder_path, f"{folder_name}_Summary_{metric}.png"),
            bbox_inches="tight",
        )
        plt.close()



def main():
    """ Main function """
    global tag_colors
    root = tk.Tk()
    root.withdraw()
    folder_paths = []

    # Ask user to select folders
    while True:
        folder_path = filedialog.askdirectory(title="Select a Folder with CSV Files")
        if not folder_path:
            break
        folder_paths.append(folder_path)
        if not messagebox.askyesno("Continue", "Do you want to add another folder?"):
            break

    if not folder_paths:
        print("No folders selected. Exiting.")
        return

    # Input parameters from user
    um_per_pix = float(input("Enter the conversion factor (microns per pixel): "))
    min_frame = float(input("Enter the duration of each timeframe (in minutes): "))

    all_data = []
    tidy = []
    count = 0

    for folder_path in folder_paths:
        folder_name = os.path.basename(folder_path)
        exp_label = folder_name
        (
            track_id,
            wno_track,
            meanvel_track,
            disp_track,
            frame_track,
            wno_fov,
            wno_rms_fov,
            MeanVel_fov,
            MeanVel_sd_fov,
            filelist,
            tag,
            gelma,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        sorted_folder = os.path.join(folder_path, folder_name)
        os.makedirs(sorted_folder, exist_ok=True)
        folder_track_data[folder_name] = {}

        # Process all CSV files in the folder
        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                file_path = os.path.join(folder_path, file)
                print(f"Processing {file}...")
                df = process_csv(file_path)
                if df["track_id"].nunique() < 10:
                    print(f"Skipping {file} – less than 10 valid tracks.")
                    continue

                output_path = os.path.join(sorted_folder, file)
                # Extract tag from filename format "(number)_(string)_XXXX_(other)"
                parts = file.split("_")
                s = "Glass" if parts[1] == "CNTRL" else "Gel"
                if s == "Gel":
                    t = parts[2]
                else:
                    t = "CNTRL"
                if t not in folder_track_data[folder_name]:
                    folder_track_data[folder_name][t] = []
                    tag_colors = assign_colors_iteratively(tag_colors, [t])
                folder_track_data[folder_name][t].append(df)

                # Generate plots and summary metrics
                (
                    track_id,
                    wno_track,
                    meanvel_track,
                    disp_track,
                    frame_track,
                    wno,
                    wno_rms,
                    meanvel,
                    meanvel_sd,
                ) = create_nematic_order_plot(df, um_per_pix, min_frame, output_path)
                create_track_movement_tiff(df, output_path, t)

                exp_label = os.path.basename(folder_path)
                all_data.append(
                    {
                        "WNO": wno,
                        "WNO_rms": wno_rms,
                        "MeanVel": meanvel,
                        "MeanVel_sd": meanvel_sd,
                        "TAG": t,
                        "STIFF": s,
                        "FILE": file,
                        "FOLDER": exp_label,
                    }
                )
                for tid, w, vel, disp, nfr in zip(
                    track_id, wno_track, meanvel_track, disp_track, frame_track
                ):
                    tidy.append(
                        {
                            "TrackID": tid + count,
                            "Track_wno": w,
                            "Track_displacement [um]": disp,
                            "Track_MeanVel [um/min]": vel,
                            "Track length [frame]": nfr,
                            "TAG": t,
                            "STIFF": s,
                            "FILE": file,
                            "FOLDER": exp_label,
                        }
                    )
                count += len(track_id)
                wno_fov.append(wno)
                wno_rms_fov.append(wno_rms)
                MeanVel_fov.append(meanvel)
                MeanVel_sd_fov.append(meanvel_sd)
                filelist.append(file[:-4])
                tag.append(t)
                gelma.append(s)

        # Save summary and plots for each folder
        df1 = pd.DataFrame(
            data={
                "WNO_fov": wno_fov,
                "WNO_rms": wno_rms_fov,
                "MeanVel": MeanVel_fov,
                "MeanVel_sd": MeanVel_sd_fov,
                "FOV": filelist,
            }
        )
        df1.to_csv(
            os.path.join(folder_path, f"{folder_name}_recap.csv"), sep=",", index=False
        )
        flag = False

        folder_plotdf = pd.DataFrame(
            data={
                "WNO": wno_fov,
                "MeanVel": MeanVel_fov,
                "TAG": tag,
                "STIFF": gelma,
                "FOLDER": [exp_label] * len(wno_fov),
            }
        )
        create_summary_plots(folder_plotdf, folder_path, folder_name, flag)
        create_combined_track_plot_per_tag(folder_name, folder_path)

        if write_provenance is not None:
            write_provenance(
                folder_path,
                params={
                    "um_per_pix": um_per_pix,
                    "min_frame_minutes": min_frame,
                    "folder": folder_name,
                },
                script_path=__file__,
                filename="migration_analysis_provenance.json",
            )

        print(f"Recap and summary plots for folder {folder_name} saved.")

    # Save combined summary and tidy data
    combined_plotdf = pd.DataFrame(all_data)
    combined_trackdf = pd.DataFrame(tidy)
    flag = True
    last_common_path = os.path.commonpath(folder_paths)
    create_summary_plots(combined_plotdf, last_common_path, "Combined", flag)
    combined_plotdf.to_csv(
        os.path.join(last_common_path, "Summary_recap.csv"), index=False
    )
    combined_trackdf.to_csv(
        os.path.join(last_common_path, "Tidy_recap.csv"), index=False
    )

    if write_provenance is not None:
        write_provenance(
            last_common_path,
            params={
                "um_per_pix": um_per_pix,
                "min_frame_minutes": min_frame,
                "scope": "combined_across_folders",
                "folders": [os.path.basename(p) for p in folder_paths],
            },
            script_path=__file__,
            filename="migration_analysis_combined_provenance.json",
        )


if __name__ == "__main__":
    main()
