# Contact modulus of GelMA hydrogels by instrumented nanoindentation

Raw curves, analysis code, a validation against known ground truth, and the
results, for 1058 indentations on 32 photopolymerised GelMA gels across three
campaigns.

## Repository layout

This folder ships the software only; it contains no data files. The dataset
(raw curves, synthetic set, results and validation) is archived separately on
Zenodo, [10.5281/zenodo.21840232](https://doi.org/10.5281/zenodo.21840232).
Download and unpack it, then point the code at it: every script resolves its
paths from one `ROOT` constant, which defaults to `../data` and is overridden
by the `LIGHTS_DATA` environment variable.

```
code/     pyproject.toml, setup.cfg, _common/, scripts/, verify.py
```

```bash
# example: run against the unpacked archive wherever it lives
set LIGHTS_DATA=/path/to/data
python code/scripts/03_analyze.py
```

## Install

Using pip:
```bash
# Recommended: install into a clean Python 3.11 virtual environment (venv or conda)
pip install -r requirements.txt
```

Using Conda:
```bash
conda env create -f environment.yml
conda activate lights-indentation
```

## Run

```bash
python code/scripts/01_synthesise.py    # the synthetic set + its truth table
python code/scripts/02_validate.py      # recover that truth
python code/scripts/03_analyze.py       # QC + pipeline over real datasets, defaults to `raw/` folder in `data/`
python code/scripts/04_statistics.py    # gel-level statistics
python code/verify.py                   # six gates for control
```

The chain takes about four minutes end to end, of which `03_analyze.py` is two.
`python code/verify.py --full` re-runs the chain and then the gates.

## What is measured

Contact modulus *E\**, over 0.5-3.5 um of indentation, at 5 um s⁻¹, with a
spherical probe of radius 25-27.5 um, in 1x PBS at pH 7.4 and room temperature.
The window and the rate are part of the quantity: the apparent modulus of these
gels rises with indentation depth.

Three measurements of that depth dependence, none of which takes a contact
point as given, are in `data/results/statistics.csv` and per curve in
`data/results/per_curve.csv`:

| | |
|---|---|
| pointwise modulus (d*F*/d*s*)/(2√(*Rδ*)), per-curve ratio at 4.00 um to 0.75 um | x1.36, x1.66, x1.49 for the three campaigns, on the 155 / 196 / 173 curves reaching both bins |
| window ladder: linearised Hertz refitted over [lo, 1] x peak force, contact point a free intercept | x0.66 at lo = 0.02 rising monotonically to x1.20 at lo = 0.70, relative to lo = 0.30, with no plateau |
| curvature of (d*F*/d*s*)² against raw displacement, a straight line for a homogeneous half-space | deep-half slope x2.83 the shallow-half slope; 90.9 % of curves exceed 1 |




