"""Step 1: build the synthetic dataset and record its ground truth.

The files written here are plausible Chiaro files marked as synthetic in the directory name, the file name
and eleven header fields.

Coverage is deliberately wider than the real curves:

* the ordinary case, at 2, 8 and 20 kPa, at both sampling rates;
* the imperfections QC is meant to tolerate: drift, the ferrule-top
  oscillation, extra noise;
* the mechanics that make a gel not a Hertzian half-space: a bonded finite
  film, a compliant surface layer, a depth-graded modulus, JKR adhesion, and
  viscoelastic and poro-viscoelastic relaxation;
* the pathologies QC is meant to reject: no contact, starting in contact,
  too shallow, a glass-hard contact, a snap-in, a slip plateau, a debris
  spike.

Eight replicates of each with independent noise, plus one noise-free twin are produced per
scenario.

Outputs
-------
data/synthetic/SYNTHETIC_<scenario>/matrix_<scenario>_1/SYNTHETIC_S-1 ...txt
data/synthetic/SYNTHETIC_NOISEFREE_<scenario>/... 
data/synthetic/truth.csv    
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("LIGHTS_DATA", CODE.parent / "data"))
sys.path.insert(0, str(CODE / "_common"))

import lights_synth as ls  # nopep8

OUT = ROOT / "synthetic"

#: Noise settings, switched off for the noise-free twin
NOISE_KEYS = (
    "deflection_noise_nm",
    "oscillation_nN",
    "oscillation_hz",
    "drift_nN_per_s",
)


def truth_record(truth, folder: Path, name: str, noise_free: bool) -> dict:
    """Rows of the ground-truth table."""
    prefix = "SYNTHETIC_NOISEFREE" if noise_free else "SYNTHETIC"
    stem = Path(name).stem
    record = asdict(truth)
    extra = record.pop("extra")
    record.update(
        {
            "campaign": f"{prefix}_{truth.scenario}",
            "sample_id": f"{truth.scenario}_1",
            "x_index": int(stem.split("X-")[1][:2]),
            "y_index": 1,
            "curve_id": f"{prefix}_{truth.scenario}/{truth.scenario}_1/{stem}",
            "relative_path": f"{folder.name}/{name}",
            "path": (folder / name).relative_to(ROOT).as_posix(),
            "noise_free": noise_free,
            **{f"gen_{k}": v for k, v in extra.items()},
        }
    )
    record["relax_times_s"] = str(list(truth.relax_times_s))
    record["relax_weights"] = str(list(truth.relax_weights))
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=8)
    ap.add_argument("--scenario", default=None)
    args = ap.parse_args()
    names = [args.scenario] if args.scenario else list(ls.SCENARIOS)
    print(f"generator {ls.SYNTH_VERSION}, {args.replicates} replicates")

    rows = []
    for scenario in names:
        spec = ls.SCENARIOS[scenario]
        folder = OUT / f"SYNTHETIC_{scenario}" / f"matrix_{scenario}_1"
        for rep in range(1, args.replicates + 1):
            columns, truth, header = ls.generate_curve(
                scenario,
                seed=1000 * names.index(scenario) + rep,
                **spec["kw"],
            )
            name = f"SYNTHETIC_S-1 X-{rep:02d} Y-01 I-01.txt"
            ls.write_curve(folder / name, columns, truth, header)
            rows.append(truth_record(truth, folder, name, False))

        cfg = ls.SynthConfig(
            deflection_noise_nm=0.0,
            piezo_noise_nm=0.0,
            oscillation_nN=0.0,
            drift_nN_per_s=0.0,
        )
        kw = {k: v for k, v in spec["kw"].items() if k not in NOISE_KEYS}
        columns, truth, header = ls.generate_curve(
            scenario, cfg=cfg, seed=999_999, **kw
        )
        folder_nf = (
            OUT / f"SYNTHETIC_NOISEFREE_{scenario}" / f"matrix_{scenario}_1"
        )
        name = "SYNTHETIC_S-1 X-99 Y-01 I-01.txt"
        ls.write_curve(folder_nf / name, columns, truth, header)
        rows.append(truth_record(truth, folder_nf, name, True))
        print(f"{scenario:<22}{spec['note']}")

    df = pd.DataFrame(rows)
    out = OUT / "truth.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {len(df)} files and {out}")
    print(
        "peak force (nN) by scenario:\n"
        + df.groupby("scenario")["gen_peak_force_nN"]
        .median()
        .round(1)
        .to_string()
    )


if __name__ == "__main__":
    main()
