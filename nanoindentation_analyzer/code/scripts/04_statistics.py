"""Step 4: gel-level statistics and the variance budget.

Every comparison is made at gel level, because the independently prepared
gel is the experimental unit. The 36 indentations inside a gel are
repeated measures of one casting.
At four gels a side the exact two-sided rank-sum cannot return a p below
2/70 = 0.029, so every comparison that reaches the floor is reported as being
at it.

The variance budget is a nested random-effects fit on ln E*

Outputs
-------
data/results/per_gel.csv          one row per gel
data/results/variance_budget.csv  the nested decomposition per campaign
data/results/statistics.csv       every number the text quotes
data/results/figures/modulus_by_concentration.png / .eps
data/results/figures/batch_and_storage.png
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("LIGHTS_DATA", CODE.parent / "data"))
sys.path.insert(0, str(CODE / "_common"))

import lights_stats as lstat  # nopep8
import matplotlib  # nopep8

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # nopep8

OUT = ROOT / "results"
CONDITION_COLOUR = {75: "#56B4E9", 100: "#0072B2", 150: "#D55E00"}
CAMPAIGN_LABEL = {
    "D1_June2025": "batch 1",
    "D2_June2026": "batch 2, fresh",
    "D3_July2026": "batch 2, +1 month",
}

PAIRS = (
    ("batch", "D1_June2025", "D2_June2026", (75, 100, 150)),
    ("storage", "D2_June2026", "D3_July2026", (75, 100, 150)),
)

GROUP_LABELS = {
    "batch": ("Batch:", "Batch 1", "Batch 2"),
    "storage": ("Timepoint:", "Fresh", "1 month old"),
}


def gel_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["campaign", "condition_mg_ml", "sample_id"])
        .agg(
            n=("E_star_kPa", "size"),
            median_kPa=("E_star_kPa", "median"),
            geomean_kPa=(
                "E_star_kPa",
                lambda x: float(np.exp(np.log(x).mean())),
            ),
            sigma_log=("E_star_kPa", lambda x: float(np.log(x).std(ddof=1))),
            median_contact_s0_um=("contact_s0_um", "median"),
        )
        .reset_index()
    )


def _sqrt(x) -> float:
    return float(np.sqrt(x)) if np.isfinite(x) and x >= 0 else np.nan


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write CSV
    """
    out = frame.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(
                lambda v: v.replace(",", ";") if isinstance(v, str) else v
            )
    out.columns = [str(c).replace(",", ";") for c in out.columns]
    out.to_csv(path, index=False)


def nested_variance(values, gel, cond) -> dict:
    """Nested random effects on ln E* by moments.

    Reported as standard deviations on the log scale.
    """
    y = np.log(values)
    within, gel_n = [], []
    cond_means = []
    for c in np.unique(cond):
        gel_means = []
        for g in np.unique(gel[cond == c]):
            v = y[(cond == c) & (gel == g)]
            if v.size < 2:
                continue
            within.append(np.var(v, ddof=1) * (v.size - 1))
            gel_n.append(v.size)
            gel_means.append(v.mean())
        if len(gel_means) >= 2:
            cond_means.append(np.mean(gel_means))
    dof = sum(n - 1 for n in gel_n)
    s2_within = float(np.sum(within) / dof) if dof else np.nan
    # between-gel: variance of the gel means, corrected for sampling error
    per_cond = []
    for c in np.unique(cond):
        means = [
            y[(cond == c) & (gel == g)].mean()
            for g in np.unique(gel[cond == c])
            if y[(cond == c) & (gel == g)].size >= 2
        ]
        if len(means) >= 2:
            per_cond.append(np.var(means, ddof=1))
    n_bar = float(np.mean(gel_n)) if gel_n else np.nan
    s2_gel = (
        max(float(np.mean(per_cond)) - s2_within / n_bar, 0.0)
        if per_cond
        else np.nan
    )
    s2_cond = (
        float(np.var(cond_means, ddof=1)) if len(cond_means) >= 2 else np.nan
    )
    return {
        "sigma_within_gel": _sqrt(s2_within),
        "sigma_between_gel": _sqrt(s2_gel),
        "sigma_between_condition": _sqrt(s2_cond),
        "cv_within_gel_percent": (
            100 * np.sqrt(np.exp(s2_within) - 1)
            if np.isfinite(s2_within)
            else np.nan
        ),
        "mean_curves_per_gel": n_bar,
    }


def variance_budget(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for campaign, sub in [*list(df.groupby("campaign")), ("ALL", df)]:
        comp = nested_variance(
            sub.E_star_kPa.to_numpy(),
            (sub.campaign + "/" + sub.sample_id).to_numpy(),
            (sub.campaign + "/" + sub.condition_mg_ml.astype(str)).to_numpy(),
        )
        s_within = comp["sigma_within_gel"]
        s_analysis = float(np.nanmedian(sub.sigma_analysis))
        s_meas = float(np.nanmedian(sub.sigma_measurement))
        residual = s_within**2 - s_analysis**2 - s_meas**2
        rows.append(
            {
                "campaign": campaign,
                "n_curves": len(sub),
                "n_gels": int((sub.campaign + sub.sample_id).nunique()),
                **comp,
                "sigma_analysis": s_analysis,
                "sigma_measurement": s_meas,
                "sigma_material_and_position": _sqrt(residual),
                "analysis_share_of_variance": s_analysis**2 / s_within**2,
                "measurement_share_of_variance": s_meas**2 / s_within**2,
                "material_share_of_variance": max(residual, 0.0) / s_within**2,
            }
        )
    return pd.DataFrame(rows)


def concentration_figure(gels: pd.DataFrame) -> None:
    """One point per gel, bar at the median of the gel medians."""
    campaigns = [c for c in CAMPAIGN_LABEL if c in set(gels.campaign)]
    fig, axes = plt.subplots(
        1, len(campaigns), figsize=(2.6 * len(campaigns), 3.5), sharey=True
    )
    for ax, campaign in zip(np.atleast_1d(axes), campaigns, strict=False):
        frame = gels[gels.campaign == campaign]
        conds = sorted(frame.condition_mg_ml.unique())
        medians = []
        for i, cond in enumerate(conds):
            v = frame.loc[frame.condition_mg_ml == cond, "median_kPa"]
            ax.plot(
                i + np.linspace(-0.13, 0.13, len(v)),
                v,
                "o",
                ms=5.5,
                color=CONDITION_COLOUR.get(cond, "0.4"),
                mec="k",
                mew=0.5,
                clip_on=False,
                zorder=4,
            )
            med = float(np.median(v))
            medians.append(med)
            ax.hlines(med, i - 0.30, i + 0.30, color="k", lw=1.8, zorder=5)
            ax.annotate(
                f"{med:.2f}",
                xy=(i + 0.33, med),
                fontsize=7,
                va="center",
                ha="left",
            )
        for i in range(len(conds) - 1):
            ax.annotate(
                f"x{medians[i + 1] / medians[i]:.2f}",
                xy=(i + 0.5, float(np.sqrt(medians[i] * medians[i + 1]))),
                fontsize=6.5,
                ha="center",
                va="bottom",
                color="0.35",
            )
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([str(c) for c in conds])
        ax.set_xlabel("GelMA (mg mL$^{-1}$)")
        ax.set_title(CAMPAIGN_LABEL[campaign], fontsize=8.5)
        ax.set_xlim(-0.55, len(conds) - 0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("contact modulus $E^*$ (kPa)")
    axes[0].set_yscale("log")
    axes[0].set_ylim(1.2, 16.0)
    axes[0].set_yticks([1.5, 2, 3, 5, 8, 12])
    axes[0].get_yaxis().set_major_formatter(
        matplotlib.ticker.ScalarFormatter()
    )
    axes[0].get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())

    fig.tight_layout(rect=(0, 0.10, 1, 1))
    for ext in ("png", "eps"):
        fig.savefig(
            OUT / "figures" / f"modulus_by_concentration.{ext}", dpi=400
        )
    plt.close(fig)


def _modulus_group(ax, x: float, values: np.ndarray, marker: str) -> None:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return
    lo, hi, med = float(v.min()), float(v.max()), float(np.median(v))
    if hi > lo:
        ax.add_patch(
            plt.Rectangle(
                (x - 0.15, lo),
                0.30,
                hi - lo,
                fill=False,
                edgecolor="k",
                lw=1.0,
                zorder=2,
            )
        )
    ax.hlines(med, x - 0.15, x + 0.15, color="k", lw=1.7, zorder=2.6)
    jit = np.linspace(-0.06, 0.06, v.size) if v.size > 1 else np.zeros(1)
    ax.plot(
        x + jit, v, marker, color="k", mec="k", mew=0.5, ms=6,
        ls="none", zorder=3,
    )


def batch_and_storage(gels: pd.DataFrame) -> pd.DataFrame:
    """Gel-level batch and storage comparisons
    """
    band = "#e7e7e7"
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.0))
    for ax, (label, a, b, conds) in zip(axes, PAIRS, strict=False):
        title, lab_a, lab_b = GROUP_LABELS[label]
        for i, cond in enumerate(conds):
            ax.axvspan(i - 0.42, i + 0.42, color=band, lw=0, zorder=0)
            va = gels[
                (gels.campaign == a) & (gels.condition_mg_ml == cond)
            ].median_kPa.to_numpy()
            vb = gels[
                (gels.campaign == b) & (gels.condition_mg_ml == cond)
            ].median_kPa.to_numpy()
            _modulus_group(ax, i - 0.18, va, "o")
            _modulus_group(ax, i + 0.18, vb, "^")
            if va.size < 2 or vb.size < 2:
                continue
            rs = lstat.exact_ranksum(np.log(va), np.log(vb))
            ratio = float(
                np.exp(np.median(np.log(vb)) - np.median(np.log(va)))
            )
            rows.append(
                {
                    "comparison": label,
                    "campaign_a": a,
                    "campaign_b": b,
                    "condition_mg_ml": cond,
                    "n_a": int(va.size),
                    "n_b": int(vb.size),
                    "median_a_kPa": float(np.median(va)),
                    "median_b_kPa": float(np.median(vb)),
                    "ratio_b_over_a": ratio,
                    "p_exact_two_sided": float(rs["p_exact_two_sided"]),
                    "p_floor": float(rs["p_floor"]),
                    "at_floor": bool(rs["at_floor"]),
                    "cliffs_delta": float(rs["cliffs_delta"]),
                }
            )
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([str(c) for c in conds], fontweight="bold")
        ax.set_xlim(-0.5, len(conds) - 0.5)
        ax.set_ylim(0.0, 15.0)
        ax.set_yticks(np.arange(0.0, 15.01, 2.5))
        ax.set_xlabel("GelMA Concentration [mg mL$^{-1}$]", fontweight="bold")
        ax.set_ylabel("Apparent Modulus [kPa]", fontweight="bold")
        for lbl in ax.get_yticklabels():
            lbl.set_fontweight("bold")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(1.8)
        ax.tick_params(width=1.8, length=5)
        ax.set_box_aspect(0.81)
        ax.set_axisbelow(True)
        ax.grid(axis="y", ls=":", lw=0.6, color="0.7")
        ax.plot([], [], "o", color="k", mec="k", ms=6, ls="none", label=lab_a)
        ax.plot([], [], "^", color="k", mec="k", ms=6, ls="none", label=lab_b)
        leg = ax.legend(
            title=title,
            fontsize=7,
            title_fontsize=7,
            loc="upper left",
            frameon=True,
            edgecolor="0.5",
        )
        leg.get_title().set_fontweight("bold")
        for text in leg.get_texts():
            text.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "batch_and_storage.png", dpi=300)
    plt.close(fig)
    return pd.DataFrame(rows)


def statistic(quantity, value, **kw) -> dict:
    row = {
        "quantity": quantity,
        "campaign": kw.get("campaign", "ALL"),
        "condition_mg_ml": kw.get("condition", np.nan),
        "value": value,
        "unit": kw.get("unit", ""),
        "n": kw.get("n", np.nan),
        "p_value": kw.get("p_value", np.nan),
        "p_floor": kw.get("p_floor", np.nan),
        "notes": kw.get("notes", ""),
    }
    return row


def depth_dependence(df: pd.DataFrame) -> list[dict]:
    """The three contact-point-free measurements aggregated."""
    rows = []
    ratio = (df["Ept_4.00_kPa"] / df["Ept_0.75_kPa"]).replace(
        [np.inf, -np.inf], np.nan
    )
    for campaign, sub in [
        *list(df.assign(r=ratio).groupby("campaign")),
        ("ALL", df.assign(r=ratio)),
    ]:
        ok = sub.r[np.isfinite(sub.r)]
        rows.append(
            statistic(
                "pointwise modulus, per-curve E*(4.00 um) / E*(0.75 um)",
                float(ok.median()),
                campaign=campaign,
                n=int(ok.size),
                notes="median over curves reaching both 0.25 um bins",
            )
        )
    for lo in (0.02, 0.05, 0.10, 0.20, 0.50, 0.70):
        v = (df[f"ladder_E_{lo:.2f}_kPa"] / df["ladder_E_0.30_kPa"]).replace(
            [np.inf, -np.inf], np.nan
        )
        rows.append(
            statistic(
                f"window ladder, per-curve E*[{lo:.2f},1] / E*[0.30,1]",
                float(v.median()),
                n=int(v.notna().sum()),
                notes="linearised plot, contact point a free intercept",
            )
        )
    k = df.k2_slope_ratio.replace([np.inf, -np.inf], np.nan)
    rows.append(
        statistic(
            "stiffness plot, deep-half slope / shallow-half slope of "
            "(dF/ds)^2 vs s",
            float(k.median()),
            n=int(k.notna().sum()),
            notes=f"1 for a half-space; {100 * float((k > 1).mean()):.1f} % "
            "of curves exceed 1",
        )
    )
    return rows


def adjustments(df: pd.DataFrame) -> list[dict]:
    """ Multiplicative terms."""
    film = (df.E_star_film_h50_kPa / df.E_star_kPa).replace(
        [np.inf, -np.inf], np.nan
    )
    envelope = (df.E_star_film_h30_kPa / df.E_star_film_h1000_kPa).replace(
        [np.inf, -np.inf], np.nan
    )
    rows = [
        statistic(
            "bonded-film adjustment at h = 50 um, E*(film) / E*(pipeline)",
            float(film.median()),
            n=int(film.notna().sum()),
            notes="stated, not applied: the thickness was not measured",
        ),
        statistic(
            "bonded-film envelope, E*(h = 30 um) / E*(h = 1000 um)",
            float(envelope.median()),
            n=int(envelope.notna().sum()),
            notes="span of the adjustment over an unmeasured thickness",
        ),
    ]
    for cap in ("2.5", "4.0"):
        v = (df[f"E_star_cap{cap}um_kPa"] / df.E_star_kPa).replace(
            [np.inf, -np.inf], np.nan
        )
        rows.append(
            statistic(
                f"window choice, E*(cap {cap} um) / E*(cap 3.5 um)",
                float(v.median()),
                n=int(v.notna().sum()),
                notes="the depth dependence, seen as a window sensitivity",
            )
        )
    return rows


def main() -> None:
    pd.set_option("display.width", 210)
    df = pd.read_csv(OUT / "per_curve.csv")
    cond = pd.read_csv(OUT / "condition_summary.csv")
    gels = gel_table(df)
    write_csv(gels, OUT / "per_gel.csv")

    budget = variance_budget(df)
    write_csv(budget, OUT / "variance_budget.csv")
    print("variance budget (sigma on ln E*):")
    print(budget.round(4).to_string(index=False))

    rows = [
        statistic(
            "curves retained by quality control",
            float(cond.n_retained.sum()) / float(cond.n_measured.sum()),
            n=int(cond.n_measured.sum()),
            notes=f"{int(cond.n_retained.sum())} of "
            f"{int(cond.n_measured.sum())}; worst condition "
            f"{100 * cond.retention.min():.1f} %",
        )
    ]
    for _, g in (
        gels.groupby(["campaign", "condition_mg_ml"])
        .median_kPa.agg(n="size", median="median")
        .reset_index()
        .iterrows()
    ):
        rows.append(
            statistic(
                "median of gel medians",
                float(g["median"]),
                campaign=g.campaign,
                condition=g.condition_mg_ml,
                unit="kPa",
                n=int(g.n),
                notes="E* over 0.5-3.5 um of indentation at 5 um/s",
            )
        )
    # Trend across concentrations, at gel level, on the campaign that has
    # all three: exact over every distinct assignment of the twelve gels.
    for campaign in ("D2_June2026", "D3_July2026"):
        sub = gels[gels.campaign == campaign]
        jt = lstat.jonckheere_terpstra(
            [
                sub.loc[sub.condition_mg_ml == c, "median_kPa"].to_numpy()
                for c in sorted(sub.condition_mg_ml.unique())
            ],
            alternative="increasing",
        )
        rows.append(
            statistic(
                "Jonckheere-Terpstra trend across 75/100/150, gel level",
                jt["statistic"],
                campaign=campaign,
                n=jt["n"],
                p_value=jt["p_value"],
                notes=jt["notes"],
            )
        )
    bs = batch_and_storage(gels)
    for _, r in bs.iterrows():
        rows.append(
            statistic(
                f"{r.comparison}: later / earlier, gel medians",
                r.ratio_b_over_a,
                campaign=f"{r.campaign_b} vs {r.campaign_a}",
                condition=r.condition_mg_ml,
                n=int(r.n_a + r.n_b),
                p_value=r.p_exact_two_sided,
                p_floor=r.p_floor,
                notes="exact two-sided rank-sum on gel medians"
                + (
                    ", at the floor: complete separation" if r.at_floor else ""
                ),
            )
        )
    rows += depth_dependence(df)
    rows += adjustments(df)
    for _, b in budget.iterrows():
        for key, unit in (
            ("sigma_within_gel", "ln E*"),
            ("sigma_between_gel", "ln E*"),
            ("sigma_between_condition", "ln E*"),
            ("material_share_of_variance", ""),
            ("analysis_share_of_variance", ""),
            ("measurement_share_of_variance", ""),
        ):
            rows.append(
                statistic(
                    key.replace("_", " "),
                    float(b[key]),
                    campaign=b.campaign,
                    unit=unit,
                    n=int(b.n_curves),
                )
            )
    stats = pd.DataFrame(rows)
    write_csv(stats, OUT / "statistics.csv")

    concentration_figure(gels)
    print("\nbatch and storage:")
    print(bs.round(4).to_string(index=False))
    print("\nstatistics.csv:")
    print(
        stats[["quantity", "campaign", "condition_mg_ml", "value", "p_value"]]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
