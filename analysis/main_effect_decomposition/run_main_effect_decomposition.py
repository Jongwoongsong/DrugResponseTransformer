#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    if len(x) < 2:
        return np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def r2_from_fit(y, fit):
    y = np.asarray(y, float)
    fit = np.asarray(fit, float)

    sst = np.sum(
        (y - np.mean(y)) ** 2
    )

    if sst <= 0:
        return np.nan

    sse = np.sum(
        (y - fit) ** 2
    )

    return float(1.0 - sse / sst)


# =====================================================================
# recover exact 413 eligible structures from final drug cache
# =====================================================================


def group_mean_fit(values, labels):
    x = pd.DataFrame(
        {
            "v": np.asarray(
                values,
                float,
            ),
            "g": labels,
        }
    )

    return (
        x.groupby("g")["v"]
        .transform("mean")
        .to_numpy(float)
    )


def additive_ols_fit(
    df,
    value_col,
):
    """
    Exact sparse OLS:
      value ~ intercept + cell + drug

    Reference coding prevents rank deficiency.
    """

    y = (
        pd.to_numeric(
            df[value_col],
            errors="coerce",
        )
        .to_numpy(float)
    )

    cell = pd.Categorical(
        df["cell_id"]
    )

    drug = pd.Categorical(
        df["canonical_smiles"]
    )

    c = cell.codes.astype(int)
    d = drug.codes.astype(int)

    nc = len(cell.categories)
    nd = len(drug.categories)
    n = len(df)

    # columns:
    # 0 = intercept
    # 1...(nc-1) = non-reference cell levels
    # nc...(nc+nd-2) = non-reference drug levels

    rows = [
        np.arange(n)
    ]

    cols = [
        np.zeros(
            n,
            dtype=int,
        )
    ]

    vals = [
        np.ones(
            n,
            dtype=float,
        )
    ]

    m_cell = c > 0

    rows.append(
        np.where(m_cell)[0]
    )

    cols.append(
        c[m_cell]
    )

    vals.append(
        np.ones(
            np.sum(m_cell),
            dtype=float,
        )
    )

    m_drug = d > 0

    rows.append(
        np.where(m_drug)[0]
    )

    cols.append(
        nc + d[m_drug] - 1
    )

    vals.append(
        np.ones(
            np.sum(m_drug),
            dtype=float,
        )
    )

    row = np.concatenate(rows)
    col = np.concatenate(cols)
    val = np.concatenate(vals)

    n_cols = (
        1
        + (nc - 1)
        + (nd - 1)
    )

    X = sparse.coo_matrix(
        (
            val,
            (row, col),
        ),
        shape=(n, n_cols),
    ).tocsr()

    sol = lsqr(
        X,
        y,
        atol=1e-10,
        btol=1e-10,
        iter_lim=5000,
    )

    beta = sol[0]

    fit = np.asarray(
        X @ beta
    ).ravel()

    residual = y - fit

    return {
        "fit": fit,
        "residual": residual,
        "lsqr_istop": int(sol[1]),
        "lsqr_iterations": int(sol[2]),
    }


def decomposition(df, value_col):
    y = pd.to_numeric(
        df[value_col],
        errors="coerce",
    ).to_numpy(float)

    cell_fit = group_mean_fit(
        y,
        df["cell_id"],
    )

    drug_fit = group_mean_fit(
        y,
        df["canonical_smiles"],
    )

    add = additive_ols_fit(
        df,
        value_col,
    )

    additive_fit = add["fit"]
    residual = add["residual"]

    sst = float(
        np.sum(
            (y - np.mean(y)) ** 2
        )
    )

    residual_fraction = float(
        np.sum(
            residual ** 2
        )
        / sst
    )

    return {
        "variance": float(
            np.var(
                y,
                ddof=0,
            )
        ),

        "sd": float(
            np.std(
                y,
                ddof=0,
            )
        ),

        "cell_only_r2":
            r2_from_fit(
                y,
                cell_fit,
            ),

        "drug_only_r2":
            r2_from_fit(
                y,
                drug_fit,
            ),

        "cell_plus_drug_r2":
            r2_from_fit(
                y,
                additive_fit,
            ),

        "residual_fraction_after_additive":
            residual_fraction,

        "additive_residual_sd":
            float(
                np.std(
                    residual,
                    ddof=0,
                )
            ),

        "additive_fit":
            additive_fit,

        "additive_residual":
            residual,

        "lsqr_istop":
            add["lsqr_istop"],

        "lsqr_iterations":
            add["lsqr_iterations"],
    }


def main_effect_alignment(d):
    true_cell = (
        d.groupby("cell_id")["y_true"]
        .mean()
    )

    pred_cell = (
        d.groupby("cell_id")["y_pred"]
        .mean()
    )

    cell_idx = (
        true_cell.index
        .intersection(
            pred_cell.index
        )
    )

    cell_mean_pcc = pearson(
        true_cell.loc[cell_idx],
        pred_cell.loc[cell_idx],
    )

    true_drug = (
        d.groupby("canonical_smiles")["y_true"]
        .mean()
    )

    pred_drug = (
        d.groupby("canonical_smiles")["y_pred"]
        .mean()
    )

    drug_idx = (
        true_drug.index
        .intersection(
            pred_drug.index
        )
    )

    drug_mean_pcc = pearson(
        true_drug.loc[drug_idx],
        pred_drug.loc[drug_idx],
    )

    true_decomp = decomposition(
        d,
        "y_true",
    )

    pred_decomp = decomposition(
        d,
        "y_pred",
    )

    interaction_pcc = pearson(
        true_decomp[
            "additive_residual"
        ],
        pred_decomp[
            "additive_residual"
        ],
    )

    return {
        "cell_mean_alignment_pcc":
            cell_mean_pcc,

        "drug_mean_alignment_pcc":
            drug_mean_pcc,

        "two_way_residual_interaction_pcc":
            interaction_pcc,

        "true_decomp":
            true_decomp,

        "pred_decomp":
            pred_decomp,
    }


# =====================================================================
# fixed-cell PCC within a selected drug subset
# =====================================================================


def aggregate_cell_drug(d):
    """
    Collapse duplicate release-specific observations for the same
    cell line x canonical molecular structure.
    """
    return (
        d.groupby(
            ["cell_id", "canonical_smiles"],
            as_index=False,
        )
        .agg(
            y_true=("y_true", "mean"),
            y_pred=("y_pred", "mean"),
            n_records=("y_true", "size"),
        )
    )


def _to_builtin(x):
    if isinstance(x, dict):
        return {k: _to_builtin(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_builtin(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def main():
    ap = argparse.ArgumentParser(
        description="Two-way cell-line/drug main-effect decomposition."
    )
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--cell_col", default="cell_id")
    ap.add_argument("--drug_col", default="canonical_smiles")
    ap.add_argument("--true_col", default="y_true")
    ap.add_argument("--pred_col", default="y_pred")
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    raw = pd.read_csv(args.input_csv)

    required = [
        args.cell_col,
        args.drug_col,
        args.true_col,
        args.pred_col,
    ]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError("Missing columns: %s" % missing)

    df = pd.DataFrame({
        "cell_id": raw[args.cell_col],
        "canonical_smiles": raw[args.drug_col],
        "y_true": pd.to_numeric(raw[args.true_col], errors="coerce"),
        "y_pred": pd.to_numeric(raw[args.pred_col], errors="coerce"),
    }).dropna()

    raw_rows = len(df)
    df = aggregate_cell_drug(df)

    result = _to_builtin(main_effect_alignment(df))
    result["audit"] = {
        "raw_response_rows": int(raw_rows),
        "aggregated_cell_drug_pairs": int(len(df)),
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
