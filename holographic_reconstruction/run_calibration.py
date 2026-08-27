"""Run the multimodal calibration end to end and write every intermediate.

Order of operations is fixed and enforced: the nanoindentation surface is the
physical reference and is reconstructed first, the confocal axial factor is then
derived against it, and only then is the holographic index difference derived.

The confocal depths are read from the output of ``groove_analyzer.py``, which is
used unmodified, so that this script never becomes a second implementation of
that pipeline.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import multimodal_calibration as M


def main() -> int:
    """Reconstruct, calibrate and write the audit tables."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base", type=Path, default=Path(__file__).resolve().parent)
    arguments = parser.parse_args()
    base: Path = arguments.base

    points = M.read_nano_points(base / "NanoindentationChiaro")
    grid = M.nano_square_grid(points)
    nano = M.nano_metrology(grid)
    np.save(base / "nano_grid.npy", grid)
    (base / "nano.pkl").write_bytes(pickle.dumps(nano))
    print(
        f"nano   depth {nano.depth_mean_um:.3f} +/- {nano.depth_sd_um:.3f} um "
        f"(n={nano.n_periods}) pitch {nano.pitch_um:.3f} tilt {nano.misalignment_deg:+.2f} "
        f"lock x{nano.locked_amplitude / nano.naive_amplitude:.3f} "
        f"recovery {nano.injected_recovery:.3f}"
    )

    recap = pd.read_csv(base / "conf_out" / "groove_recap.csv")
    confocal_depths = recap.depth_um.to_numpy(float)
    if "depth_bias_corr_factor" in recap.columns:
        depth_bias = recap.depth_bias_corr_factor.to_numpy(float)
    else:
        depth_bias = np.ones_like(confocal_depths)
    print(
        "confocal depths (bias-corrected by groove_analyzer.py, factors "
        f"{depth_bias}): {confocal_depths}"
    )

    records = []
    for folder in sorted((base / "HPI" / "HPI").iterdir()):
        if not folder.is_dir():
            continue
        result = M.analyse_surface(M.load_hpi_field(folder), M.HPI_XY_UM)
        records.append(
            {
                "field": folder.name,
                "opd_nm": result.depth_mean_um,
                "opd_sd_nm": result.depth_sd_um,
                "n_periods": result.n_periods,
                "pitch_um": result.pitch_um,
                "tilt_deg": result.misalignment_deg,
                "lock_gain": result.locked_amplitude / result.naive_amplitude,
                "injected_recovery": result.injected_recovery,
            }
        )
        print(f"hpi    {folder.name}: OPD {result.depth_mean_um:.2f} nm")
    hpi = pd.DataFrame(records)
    hpi.to_csv(base / "hpi_fields.csv", index=False)

    calibration = M.calibrate(nano, confocal_depths, hpi.opd_nm.to_numpy(float))
    low, high = M.nano_independent_delta_n_bounds(
        hpi.opd_nm.to_numpy(float), confocal_depths, calibration.bounds
    )
    payload = {
        "d_nano_um": calibration.depth_nano_um,
        "d_nano_sd_um": calibration.depth_nano_sd_um,
        "d_confocal_um": calibration.depth_confocal_um,
        "confocal_depth_bias_factors": depth_bias.tolist(),
        "confocal_depths_uncorrected_um": (confocal_depths / depth_bias).tolist(),
        "opd_nm": calibration.optical_path_nm,
        "zeta": calibration.zeta,
        "zeta_ci": list(calibration.zeta_ci),
        "delta_n": calibration.delta_n,
        "delta_n_ci": list(calibration.delta_n_ci),
        "gel_index": calibration.gel_index,
        "bounds": calibration.bounds,
        "delta_n_nano_independent": [low, high],
        "pitch_nano_um": nano.pitch_um,
        "pitch_confocal_um": float(recap.pitch_um.mean()),
        "pitch_hpi_um": float(hpi.pitch_um.mean()),
        "confocal_depths_um": confocal_depths.tolist(),
        "hpi_opd_nm": hpi.opd_nm.tolist(),
    }
    (base / "calibration.json").write_text(json.dumps(payload, indent=2))
    print(
        f"\nzeta {calibration.zeta:.4f} {calibration.zeta_ci} | "
        f"delta_n {calibration.delta_n:.5f} {calibration.delta_n_ci}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
