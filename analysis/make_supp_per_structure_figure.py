#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


STRUCTURE_CANDIDATES = [
    "canonical_smiles",
    "canonical_SMILES",
    "smiles",
    "SMILES",
    "drug_id",
    "drug_name",
]

TRUE_CANDIDATES = [
    "y_true",
    "target",
    "LN_IC50",
]

PRED_CANDIDATES = [
    "y_pred",
    "pred",
    "prediction",
]


def first_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def safe_pcc(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)

    good = np.isfinite(y) & np.isfinite(p)
    y = y[good]
    p = p[good]

    if len(y) < 2:
        return np.nan

    if np.std(y) == 0 or np.std(p) == 0:
        return np.nan

    return float(np.corrcoef(y, p)[0, 1])


def load_per_structure(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, low_memory=False)

    structure_col = first_col(df, STRUCTURE_CANDIDATES)

    if structure_col is None:
        raise ValueError(
            "Cannot identify structure column in %s; columns=%r"
            % (path, list(df.columns))
        )

    # Already per-structure.
    if "pcc" in df.columns:
        out = df[[structure_col, "pcc"]].copy()
        out.columns = ["structure", "pcc"]
        out["structure"] = out["structure"].astype(str)
        out["pcc"] = pd.to_numeric(
            out["pcc"],
            errors="coerce",
        )
        return out

    # Otherwise derive from row-level predictions.
    ycol = first_col(df, TRUE_CANDIDATES)
    pcol = first_col(df, PRED_CANDIDATES)

    if ycol is None or pcol is None:
        raise ValueError(
            "Cannot identify PCC or y_true/y_pred in %s; columns=%r"
            % (path, list(df.columns))
        )

    rows = []

    for key, g in df.groupby(structure_col, sort=True):
        rows.append({
            "structure": str(key),
            "pcc": safe_pcc(
                pd.to_numeric(
                    g[ycol],
                    errors="coerce",
                ),
                pd.to_numeric(
                    g[pcol],
                    errors="coerce",
                ),
            ),
        })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument(
        "--expected_structures",
        type=int,
        default=41,
    )
    args = ap.parse_args()

    spec = pd.read_csv(args.spec)

    required = {"model", "seed", "path"}
    missing = required - set(spec.columns)
    if missing:
        raise ValueError(
            "Spec missing columns: %s"
            % sorted(missing)
        )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    model_order = list(
        dict.fromkeys(spec["model"].astype(str))
    )

    seed_rows = []

    for _, row in spec.iterrows():
        model = str(row["model"])
        seed = str(row["seed"])
        path = str(row["path"])

        ps = load_per_structure(path)

        if ps["structure"].nunique() != args.expected_structures:
            raise RuntimeError(
                "%s seed=%s has %d structures; expected %d | %s"
                % (
                    model,
                    seed,
                    ps["structure"].nunique(),
                    args.expected_structures,
                    path,
                )
            )

        ps["model"] = model
        ps["seed"] = seed
        ps["source_file"] = path

        seed_rows.append(ps)

        defined = ps["pcc"].notna().sum()

        print(
            "[LOAD] %-16s seed=%-5s "
            "structures=%d defined=%d macro=%.6f"
            % (
                model,
                seed,
                len(ps),
                defined,
                ps["pcc"].mean(),
            )
        )

    seed_df = pd.concat(
        seed_rows,
        ignore_index=True,
    )

    seed_df.to_csv(
        output / "per_structure_seed_values.csv",
        index=False,
    )

    # For models with multiple optimization seeds,
    # average PCC structure-by-structure.
    # This preserves one value per held-out structure
    # for visualization and avoids giving DRT 3x
    # more plotting weight than single-run comparators.
    agg_rows = []

    for model in model_order:
        m = seed_df[
            seed_df["model"] == model
        ].copy()

        pivot = m.pivot_table(
            index="structure",
            columns="seed",
            values="pcc",
            aggfunc="first",
        )

        if len(pivot) != args.expected_structures:
            raise RuntimeError(
                "%s has inconsistent structure set"
                % model
            )

        values = pivot.mean(
            axis=1,
            skipna=True,
        )

        for structure, value in values.items():
            agg_rows.append({
                "model": model,
                "structure": structure,
                "pcc": float(value),
                "n_seeds": int(
                    pivot.loc[structure]
                    .notna()
                    .sum()
                ),
            })

    agg = pd.DataFrame(agg_rows)

    agg.to_csv(
        output / "per_structure_model_values.csv",
        index=False,
    )

    summary_rows = []

    for model in model_order:
        vals = (
            agg.loc[
                agg["model"] == model,
                "pcc",
            ]
            .dropna()
            .to_numpy(np.float64)
        )

        summary_rows.append({
            "model": model,
            "n_structures": int(len(vals)),
            "mean": float(np.mean(vals)),
            "sd": float(
                np.std(vals, ddof=1)
            ),
            "median": float(np.median(vals)),
            "q25": float(
                np.quantile(vals, 0.25)
            ),
            "q75": float(
                np.quantile(vals, 0.75)
            ),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        })

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        output / "per_structure_summary.csv",
        index=False,
    )

    print()
    print(summary.to_string(index=False))

    # ---------------------------------------------------------
    # Publication figure
    # ---------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10.6, 5.8)
    )

    data = [
        agg.loc[
            agg["model"] == model,
            "pcc",
        ]
        .dropna()
        .to_numpy(np.float64)
        for model in model_order
    ]

    positions = np.arange(
        1,
        len(model_order) + 1,
    )

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        showfliers=False,
        patch_artist=True,
        medianprops={
            "linewidth": 1.5,
        },
        boxprops={
            "facecolor": "white",
            "linewidth": 1.2,
        },
        whiskerprops={
            "linewidth": 1.1,
        },
        capprops={
            "linewidth": 1.1,
        },
    )

    rng = np.random.default_rng(42)

    for x, vals in zip(positions, data):
        jitter = rng.normal(
            loc=0.0,
            scale=0.055,
            size=len(vals),
        )

        ax.scatter(
            np.full(len(vals), x) + jitter,
            vals,
            s=18,
            alpha=0.55,
            edgecolors="none",
        )

        # Mean marker
        ax.scatter(
            [x],
            [np.mean(vals)],
            marker="D",
            s=48,
            facecolors="white",
            edgecolors="black",
            linewidths=1.2,
            zorder=5,
        )

    ax.axhline(
        0.0,
        linewidth=0.8,
        linestyle="--",
    )

    ax.set_ylabel(
        "Within-structure Pearson correlation"
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(
        model_order,
        rotation=28,
        ha="right",
    )

    ax.set_ylim(-1.05, 1.05)

    ax.set_title(
        "Drug-blind performance across 41 held-out molecular structures"
    )

    ax.text(
        0.01,
        0.02,
        "Each point represents one held-out canonical structure; "
        "diamonds indicate means.",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
    )

    fig.tight_layout()

    pdf = output / "Figure_S_per_structure_PCC_distribution.pdf"
    png = output / "Figure_S_per_structure_PCC_distribution.png"

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )
    fig.savefig(
        png,
        dpi=600,
        bbox_inches="tight",
    )

    plt.close(fig)

    caption = (
        "Supplementary Figure Sx. Distribution of within-structure "
        "Pearson correlation coefficients across the 41 canonical-SMILES "
        "structures held out from downstream GDSC IC50 supervision. "
        "Each point represents the correlation between observed and "
        "predicted LN_IC50 values across cell lines for one held-out "
        "structure; boxes summarize the interquartile range and median, "
        "and diamonds indicate the mean. For DRT scratch and pretrained "
        "models, structure-specific PCC values were averaged across "
        "optimization seeds 42, 123, and 2026 before visualization so "
        "that each molecular structure contributed one value per model. "
        "Classical and CSG2A comparator distributions correspond to their "
        "validation-selected final runs. The plot complements pooled "
        "metrics by showing heterogeneity in within-compound cell-line "
        "ranking performance."
    )

    (
        output / "Figure_S_caption.txt"
    ).write_text(
        caption + "\n",
        encoding="utf-8",
    )

    print()
    print("[SAVE]", pdf)
    print("[SAVE]", png)
    print(
        "[SAVE]",
        output / "per_structure_model_values.csv",
    )
    print(
        "[SAVE]",
        output / "per_structure_summary.csv",
    )


if __name__ == "__main__":
    main()
