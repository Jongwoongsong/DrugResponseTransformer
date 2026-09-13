#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch


ROOT = Path.cwd().resolve()
BASE = ROOT / "revision_work/20260811/da_bidir_pge_v2_union"

# Force imports from the exact final implementation.
sys.path.insert(0, str(BASE))

import final_union_da_ic50_finetune as fin
from Model.DrugResponseTransformer import TokenBatch


DEFAULT_CKPT = (
    BASE
    / "runs/final_union_da_mixed_pretrained_full_s42_bs32"
    / "checkpoints"
    / "final_union_da_mixed_pretrained_full_s42_bs32_20260814-011708.pth"
)


def mean_sd(values):
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return {"n": 0, "mean": None, "sd": None, "median": None}
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "median": float(np.median(x)),
    }


def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    ok = np.isfinite(a) & np.isfinite(b)
    a = a[ok]
    b = b[ok]

    if len(a) < 2:
        return float("nan")

    ra = pd.Series(a).rank(method="average").to_numpy(dtype=float)
    rb = pd.Series(b).rank(method="average").to_numpy(dtype=float)

    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")

    return float(np.corrcoef(ra, rb)[0, 1])


def topk_agreement(a, b, k=20):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    k = min(int(k), len(a), len(b))
    ia = set(np.argsort(-a)[:k].tolist())
    ib = set(np.argsort(-b)[:k].tolist())

    inter = len(ia & ib)
    union = len(ia | ib)

    return {
        "overlap_n": int(inter),
        "overlap_fraction": float(inter / max(1, k)),
        "jaccard": float(inter / max(1, union)),
    }


def normalize_positive(x):
    x = np.asarray(x, dtype=float)
    s = float(np.sum(x))
    if not np.isfinite(s) or s <= 0:
        return np.zeros_like(x)
    return x / s


def load_final_model(checkpoint, device):
    checkpoint = Path(checkpoint).resolve()

    print("=" * 90)
    print("FINAL DRT IG — LOAD")
    print("=" * 90)
    print("[CHECKPOINT]", checkpoint)
    print("[DEVICE]", device)

    pkg = fin.safe_torch_load(str(checkpoint))

    if not isinstance(pkg, dict) or "model_state_dict" not in pkg:
        raise RuntimeError("Expected final fine-tune checkpoint with model_state_dict")

    ckpt_args = dict(pkg["args"])
    args = fin.Args(**ckpt_args)

    # Device is an analysis choice only; all scientific/model args remain checkpoint-derived.
    args.device = str(device)

    df, c = fin.load_compact_gdsc(args.csv_path)
    drug_cache, drug_fdim = fin.load_drug_cache(args.drug_graph_cache)
    cell_cache = fin.load_cell_cache(args.cell_graph_cache, args.num_pathways)

    before = len(df)
    df = df[
        df[c["smiles"]].isin(drug_cache)
        & df[c["cell"]].isin(cell_cache)
    ].reset_index(drop=True)

    print("[CACHE COVERAGE] {}/{}".format(len(df), before))
    if len(df) != 415671:
        raise RuntimeError(
            "Expected final covered rows=415671, observed={}".format(len(df))
        )

    train_df, val_df, test_df = fin.split_dataframe(
        df,
        args.split_mode,
        args.split_seed,
        args.valid_ratio,
        args.test_ratio,
        c,
    )

    print(
        "[SPLIT] train={} val={} test={}".format(
            len(train_df), len(val_df), len(test_df)
        )
    )

    if args.split_mode != "mixed":
        raise RuntimeError(
            "This analysis was designed for final drug-blind checkpoint; "
            "checkpoint split_mode={}".format(args.split_mode)
        )

    if len(test_df) != 41567:
        raise RuntimeError(
            "Expected final mixed test n=41567, got {}".format(len(test_df))
        )

    n_struct = int(test_df[c["smiles"]].astype(str).nunique())
    print("[TEST STRUCTURES]", n_struct)

    if n_struct != 413:
        raise RuntimeError("Expected 413 final mixed test structures")

    landmarks = fin.load_landmarks(args.landmark_csv)
    if len(landmarks) != 268:
        raise RuntimeError("Expected 268 landmarks, got {}".format(len(landmarks)))

    model, cfg = fin.build_model(
        landmarks,
        drug_fdim,
        args,
        torch.device(device),
    )

    model.load_state_dict(pkg["model_state_dict"], strict=True)
    model.eval()

    # We only need gradients with respect to token representations.
    for p in model.parameters():
        p.requires_grad_(False)

    expected_cfg = pkg.get("model_config", None)
    if expected_cfg is not None:
        observed_cfg = asdict(cfg)
        diff = {
            k: (expected_cfg.get(k), observed_cfg.get(k))
            for k in sorted(set(expected_cfg) | set(observed_cfg))
            if expected_cfg.get(k) != observed_cfg.get(k)
        }
        print("[MODEL CONFIG DIFF]", json.dumps(diff, sort_keys=True))
        if diff:
            raise RuntimeError("Rebuilt model config differs from checkpoint")

    print("[EDGE DIRECTION]", args.edge_direction)
    print("[EDGE ENCODING]", args.edge_encoding)
    print("[FIXED CONTEXT] dose={} time={}".format(
        args.fixed_dose, args.fixed_time
    ))

    dataset = fin.IC50Dataset(
        test_df,
        c,
        drug_cache,
        cell_cache,
        args.fixed_time,
        args.fixed_dose,
    )

    return model, pkg, args, c, test_df, dataset


def select_test_indices(
    test_df,
    c,
    per_structure,
    max_structures,
    seed,
):
    smiles_col = c["smiles"]
    cell_col = c["cell"]

    structures = sorted(test_df[smiles_col].astype(str).unique().tolist())

    if max_structures > 0:
        structures = structures[:max_structures]

    selected = []

    for struct_i, smi in enumerate(structures):
        mask = test_df[smiles_col].astype(str) == str(smi)
        sub = test_df.loc[mask].copy()

        # Prevent repeated GDSC records for one cell line from dominating.
        sub = sub.drop_duplicates(subset=[cell_col])

        positions = sub.index.to_numpy(dtype=int)

        if len(positions) == 0:
            continue

        n = min(int(per_structure), len(positions))

        rng = np.random.RandomState(seed + struct_i)
        chosen = rng.choice(positions, size=n, replace=False)

        for idx in sorted(chosen.tolist()):
            selected.append(int(idx))

    return selected


def build_exact_token_batch(model, batch, device):
    time = batch["time"].to(device).reshape(-1)
    dose = batch["dose"].to(device).reshape(-1)

    # Reproduce FaithfulTokenDedupDrugResponseTransformer.forward exactly
    # through the pre-Transformer stage.
    drug_tokens, drug_masks, drug_gene_ids = model._encode_drug_tokens(
        batch["drug_graph"]
    )

    cell_tokens, cell_masks, cell_gene_ids = (
        model._encode_cell_tokens_deduplicated(
            batch["cell_seq"],
            batch["cell_ids"],
        )
    )

    tokens, masks, gene_ids = model._combine_tokens(
        drug_tokens,
        drug_masks,
        drug_gene_ids,
        cell_tokens,
        cell_masks,
        cell_gene_ids,
    )

    token_batch = TokenBatch(tokens, masks, gene_ids)
    token_batch = model._add_condition_tokens(token_batch, time, dose)
    token_batch = model._align_mask_with_tokens(token_batch)

    return token_batch, time, dose


def predict_from_tokens(model, token_batch, tokens, time, dose):
    tb = TokenBatch(
        tokens,
        token_batch.mask,
        token_batch.gene_ids,
    )

    transformer_output = model._process_transformer(
        tb,
        return_attn=False,
    )

    pred = model._process_ic50_output(
        transformer_output,
        tb,
        time,
        dose,
        bge=None,
    )

    if isinstance(pred, tuple):
        pred = pred[0]

    return pred.reshape(-1)


def get_gene_positions(token_batch, model):
    if token_batch.batch_size != 1:
        raise RuntimeError("IG implementation expects batch size 1")

    gids = token_batch.gene_ids[0]

    positions = [
        i
        for i, gene in enumerate(gids)
        if gene is not None
    ]

    names = [str(gids[i]) for i in positions]

    if len(positions) != 268:
        raise RuntimeError(
            "Expected exactly 268 named gene tokens, got {}".format(
                len(positions)
            )
        )

    expected = [str(x) for x in model.landmark_gene_order]

    if names != expected:
        raise RuntimeError(
            "Gene-token order differs from model.landmark_gene_order"
        )

    return positions, names


@torch.no_grad()
def get_attention_scores(
    model,
    token_batch,
    gene_positions,
    time,
    dose,
):
    transformer_output, attn_maps = model._process_transformer(
        token_batch,
        return_attn=True,
    )

    pred_out = model._process_ic50_output(
        (transformer_output, attn_maps),
        token_batch,
        time,
        dose,
        bge=None,
    )

    if not isinstance(pred_out, tuple):
        raise RuntimeError(
            "Expected IC50 + attention tuple when return_attn=True"
        )

    pred_attn = pred_out[0].reshape(-1)

    if isinstance(attn_maps, (list, tuple)):
        attn_last = attn_maps[-1]
    else:
        attn_last = attn_maps

    if attn_last.dim() != 4:
        raise RuntimeError(
            "Expected attention [B,H,T,T], got {}".format(
                tuple(attn_last.shape)
            )
        )

    # Existing case-study definition:
    # last-layer mean across heads, dose-query + time-query.
    attn_mean = attn_last.mean(dim=1)
    cond_to_all = attn_mean[:, 0, :] + attn_mean[:, 1, :]

    gene_cond = cond_to_all[
        0,
        torch.tensor(
            gene_positions,
            dtype=torch.long,
            device=cond_to_all.device,
        ),
    ]

    # Also retain downstream learned pooling alpha as a secondary
    # attention/readout signal.
    body_out = transformer_output[:, 2:, :]
    body_mask = token_batch.mask[:, 2:]

    scores = torch.matmul(body_out, model.pool_query)
    scores = scores.masked_fill(~body_mask, float("-inf"))
    alpha = torch.softmax(scores, dim=1)

    body_gene_positions = [
        int(pos - 2)
        for pos in gene_positions
    ]

    gene_pool = alpha[
        0,
        torch.tensor(
            body_gene_positions,
            dtype=torch.long,
            device=alpha.device,
        ),
    ]

    return (
        pred_attn,
        gene_cond.detach().cpu().numpy(),
        gene_pool.detach().cpu().numpy(),
    )


def integrated_gradients_gene_tokens(
    model,
    token_batch,
    gene_positions,
    time,
    dose,
    steps,
):
    actual = token_batch.tokens.detach()

    # Representation-level baseline:
    # remove only the 268 gene-token vectors while keeping
    # atom tokens, dose/time context, token positions and masks fixed.
    baseline = actual.clone()
    baseline[:, gene_positions, :] = 0.0

    delta = actual - baseline

    with torch.no_grad():
        pred_actual = predict_from_tokens(
            model,
            token_batch,
            actual,
            time,
            dose,
        )[0].item()

        pred_baseline = predict_from_tokens(
            model,
            token_batch,
            baseline,
            time,
            dose,
        )[0].item()

    gradients = []

    # Trapezoidal-rule IG with `steps` intervals and steps+1 points.
    for step_i in range(steps + 1):
        alpha = float(step_i) / float(steps)

        x = (
            baseline
            + alpha * delta
        ).detach().requires_grad_(True)

        pred = predict_from_tokens(
            model,
            token_batch,
            x,
            time,
            dose,
        )

        grad = torch.autograd.grad(
            outputs=pred.sum(),
            inputs=x,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]

        gradients.append(
            grad[:, gene_positions, :].detach()
        )

    g = torch.stack(gradients, dim=0)

    integral = (
        0.5 * g[0]
        + g[1:-1].sum(dim=0)
        + 0.5 * g[-1]
    ) / float(steps)

    delta_gene = delta[:, gene_positions, :]
    ig = delta_gene * integral

    # Signed sum is used for the IG completeness identity.
    ig_signed = ig.sum(dim=-1)[0]

    # L1 magnitude across embedding dimensions is the ranking score
    # compared with non-negative attention scores.
    ig_abs = ig.abs().sum(dim=-1)[0]

    prediction_delta = pred_actual - pred_baseline
    ig_sum = float(ig_signed.sum().item())
    residual = ig_sum - prediction_delta

    return {
        "ig_signed": ig_signed.detach().cpu().numpy(),
        "ig_abs": ig_abs.detach().cpu().numpy(),
        "pred_actual": float(pred_actual),
        "pred_baseline": float(pred_baseline),
        "prediction_delta": float(prediction_delta),
        "ig_signed_sum": float(ig_sum),
        "completeness_residual": float(residual),
        "completeness_relative_abs": float(
            abs(residual) / (abs(prediction_delta) + 1e-8)
        ),
    }


def analyze_one(
    model,
    dataset,
    dataset_index,
    device,
    steps,
    equivalence_tol,
):
    item = dataset[dataset_index]
    batch = fin.collate_ic50([item])

    time = batch["time"].to(device).reshape(-1)
    dose = batch["dose"].to(device).reshape(-1)

    with torch.no_grad():
        full_pred = model(
            drug_graph=batch["drug_graph"],
            cell_graph_seq=batch["cell_seq"],
            cell_ids=batch["cell_ids"],
            time=time,
            dose=dose,
            return_attn=False,
            bge=None,
        )

        if isinstance(full_pred, tuple):
            full_pred = full_pred[0]

        full_pred = float(full_pred.reshape(-1)[0].item())

        token_batch, time, dose = build_exact_token_batch(
            model,
            batch,
            device,
        )

        token_pred = float(
            predict_from_tokens(
                model,
                token_batch,
                token_batch.tokens,
                time,
                dose,
            )[0].item()
        )

    eq_abs = abs(full_pred - token_pred)

    print(
        "[EQUIVALENCE] idx={} full={:.8f} token={:.8f} abs_delta={:.3e}".format(
            dataset_index,
            full_pred,
            token_pred,
            eq_abs,
        ),
        flush=True,
    )

    if eq_abs > equivalence_tol:
        raise RuntimeError(
            "Token-path prediction equivalence failed: {:.3e} > {:.3e}".format(
                eq_abs,
                equivalence_tol,
            )
        )

    gene_positions, gene_names = get_gene_positions(
        token_batch,
        model,
    )

    with torch.no_grad():
        pred_attn, cond_attn, pool_alpha = get_attention_scores(
            model,
            token_batch,
            gene_positions,
            time,
            dose,
        )

    attn_eq = abs(float(pred_attn[0].item()) - full_pred)

    print(
        "[ATTN EQUIVALENCE] abs_delta={:.3e}".format(attn_eq),
        flush=True,
    )

    if attn_eq > equivalence_tol:
        raise RuntimeError(
            "return_attn prediction changed output: {:.3e}".format(attn_eq)
        )

    ig = integrated_gradients_gene_tokens(
        model,
        token_batch,
        gene_positions,
        time,
        dose,
        steps=steps,
    )

    ig_abs = ig["ig_abs"]

    ig_norm = normalize_positive(ig_abs)
    cond_norm = normalize_positive(cond_attn)
    pool_norm = normalize_positive(pool_alpha)

    rho_cond = spearman(ig_abs, cond_attn)
    rho_pool = spearman(ig_abs, pool_alpha)

    top_cond = topk_agreement(
        ig_abs,
        cond_attn,
        k=20,
    )
    top_pool = topk_agreement(
        ig_abs,
        pool_alpha,
        k=20,
    )

    meta = batch["meta"]

    sample_id = "{}|{}|{}".format(
        meta["canonical_smiles"][0],
        meta["cell_id"][0],
        meta["source_row"][0],
    )

    sample_row = {
        "sample_id": sample_id,
        "dataset_index": int(dataset_index),
        "source_row": int(meta["source_row"][0]),
        "canonical_smiles": str(meta["canonical_smiles"][0]),
        "drug_id": str(meta["drug_id"][0]),
        "cell_id": str(meta["cell_id"][0]),
        "true_ln_ic50": float(batch["y"][0].item()),
        "pred_ln_ic50": float(full_pred),
        "equivalence_abs_delta": float(eq_abs),
        "attention_equivalence_abs_delta": float(attn_eq),
        "ig_steps": int(steps),
        "ig_spearman_vs_cond_attn": float(rho_cond),
        "ig_spearman_vs_pool_alpha": float(rho_pool),
        "ig_cond_top20_overlap_n": int(top_cond["overlap_n"]),
        "ig_cond_top20_overlap_fraction": float(
            top_cond["overlap_fraction"]
        ),
        "ig_cond_top20_jaccard": float(top_cond["jaccard"]),
        "ig_pool_top20_overlap_n": int(top_pool["overlap_n"]),
        "ig_pool_top20_overlap_fraction": float(
            top_pool["overlap_fraction"]
        ),
        "ig_pool_top20_jaccard": float(top_pool["jaccard"]),
        "ig_pred_actual": float(ig["pred_actual"]),
        "ig_pred_baseline": float(ig["pred_baseline"]),
        "ig_prediction_delta": float(ig["prediction_delta"]),
        "ig_signed_sum": float(ig["ig_signed_sum"]),
        "ig_completeness_residual": float(
            ig["completeness_residual"]
        ),
        "ig_completeness_relative_abs": float(
            ig["completeness_relative_abs"]
        ),
    }

    gene_rows = []

    for i, gene in enumerate(gene_names):
        gene_rows.append({
            "sample_id": sample_id,
            "dataset_index": int(dataset_index),
            "source_row": int(meta["source_row"][0]),
            "canonical_smiles": str(meta["canonical_smiles"][0]),
            "drug_id": str(meta["drug_id"][0]),
            "cell_id": str(meta["cell_id"][0]),
            "gene": str(gene),
            "ig_abs": float(ig_abs[i]),
            "ig_signed": float(ig["ig_signed"][i]),
            "ig_norm": float(ig_norm[i]),
            "cond_attn": float(cond_attn[i]),
            "cond_attn_norm": float(cond_norm[i]),
            "pool_alpha": float(pool_alpha[i]),
            "pool_alpha_norm": float(pool_norm[i]),
        })

    print(
        "[IG] idx={} rho_cond={:.4f} top20_cond={}/20 "
        "rho_pool={:.4f} completeness_rel={:.4f}".format(
            dataset_index,
            rho_cond,
            top_cond["overlap_n"],
            rho_pool,
            ig["completeness_relative_abs"],
        ),
        flush=True,
    )

    return sample_row, gene_rows


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CKPT),
    )
    p.add_argument(
        "--device",
        default="cpu",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=32,
    )
    p.add_argument(
        "--per_structure",
        type=int,
        default=5,
    )
    p.add_argument(
        "--max_structures",
        type=int,
        default=0,
        help="0 = all available test structures",
    )
    p.add_argument(
        "--selection_seed",
        type=int,
        default=42,
    )
    p.add_argument(
        "--equivalence_tol",
        type=float,
        default=1e-5,
    )
    p.add_argument(
        "--out",
        required=True,
    )

    args_cli = p.parse_args()

    if args_cli.steps < 2:
        raise ValueError("--steps must be >=2")

    out = Path(args_cli.out).resolve()
    out.mkdir(parents=True, exist_ok=False)

    device = torch.device(args_cli.device)

    model, pkg, run_args, c, test_df, dataset = load_final_model(
        args_cli.checkpoint,
        device,
    )

    selected = select_test_indices(
        test_df,
        c,
        per_structure=args_cli.per_structure,
        max_structures=args_cli.max_structures,
        seed=args_cli.selection_seed,
    )

    selection_rows = []

    for idx in selected:
        row = test_df.iloc[idx]
        selection_rows.append({
            "dataset_index": int(idx),
            "source_row": int(row["_source_row"]),
            "canonical_smiles": str(row[c["smiles"]]),
            "drug_id": str(row[c["drug"]]),
            "cell_id": str(row[c["cell"]]),
            "ln_ic50": float(row[c["label"]]),
        })

    pd.DataFrame(selection_rows).to_csv(
        out / "selection.csv",
        index=False,
    )

    print("=" * 90)
    print("IG SAMPLE SET")
    print("=" * 90)
    print("[N SAMPLES]", len(selected))
    print(
        "[N STRUCTURES]",
        len(set(x["canonical_smiles"] for x in selection_rows)),
    )
    print("[STEPS]", args_cli.steps)
    print("[BASELINE] configured by IG v2 wrapper; see [IG V2 CONFIG]")
    print("[HELD FIXED] drug atoms + dose/time + mask")
    print("=" * 90)

    sample_rows = []
    gene_rows = []

    for n, dataset_index in enumerate(selected, 1):
        print(
            "\n[{}/{}] dataset_index={}".format(
                n,
                len(selected),
                dataset_index,
            ),
            flush=True,
        )

        sr, gr = analyze_one(
            model,
            dataset,
            dataset_index,
            device,
            steps=args_cli.steps,
            equivalence_tol=args_cli.equivalence_tol,
        )

        sample_rows.append(sr)
        gene_rows.extend(gr)

        # Incremental checkpointing of analysis outputs.
        pd.DataFrame(sample_rows).to_csv(
            out / "sample_metrics.partial.csv",
            index=False,
        )
        pd.DataFrame(gene_rows).to_csv(
            out / "gene_attributions.partial.csv",
            index=False,
        )

    sample_df = pd.DataFrame(sample_rows)
    gene_df = pd.DataFrame(gene_rows)

    sample_df.to_csv(
        out / "sample_metrics.csv",
        index=False,
    )
    gene_df.to_csv(
        out / "gene_attributions.csv",
        index=False,
    )

    # Equal-weight structure-level aggregation.
    structure_gene = (
        gene_df
        .groupby(
            ["canonical_smiles", "gene"],
            as_index=False,
        )[
            [
                "ig_norm",
                "cond_attn_norm",
                "pool_alpha_norm",
            ]
        ]
        .mean()
    )

    structure_gene.to_csv(
        out / "structure_gene_mean_normalized.csv",
        index=False,
    )

    structure_rows = []

    for smi, g in structure_gene.groupby("canonical_smiles"):
        if len(g) != 268:
            raise RuntimeError(
                "Structure {} has {} genes, expected 268".format(
                    smi,
                    len(g),
                )
            )

        igv = g["ig_norm"].to_numpy()
        cv = g["cond_attn_norm"].to_numpy()
        pv = g["pool_alpha_norm"].to_numpy()

        tc = topk_agreement(igv, cv, 20)
        tp = topk_agreement(igv, pv, 20)

        structure_rows.append({
            "canonical_smiles": str(smi),
            "n_genes": int(len(g)),
            "ig_spearman_vs_cond_attn": spearman(igv, cv),
            "ig_spearman_vs_pool_alpha": spearman(igv, pv),
            "ig_cond_top20_overlap_n": tc["overlap_n"],
            "ig_cond_top20_overlap_fraction": tc["overlap_fraction"],
            "ig_cond_top20_jaccard": tc["jaccard"],
            "ig_pool_top20_overlap_n": tp["overlap_n"],
            "ig_pool_top20_overlap_fraction": tp["overlap_fraction"],
            "ig_pool_top20_jaccard": tp["jaccard"],
        })

    structure_df = pd.DataFrame(structure_rows)
    structure_df.to_csv(
        out / "structure_metrics.csv",
        index=False,
    )

    # Overall gene ranking after equal-weight structure normalization.
    aggregate_gene = (
        structure_gene
        .groupby("gene", as_index=False)[
            [
                "ig_norm",
                "cond_attn_norm",
                "pool_alpha_norm",
            ]
        ]
        .mean()
    )

    aggregate_gene.to_csv(
        out / "aggregate_gene_scores.csv",
        index=False,
    )

    av_ig = aggregate_gene["ig_norm"].to_numpy()
    av_cond = aggregate_gene["cond_attn_norm"].to_numpy()
    av_pool = aggregate_gene["pool_alpha_norm"].to_numpy()

    agg_cond_top = topk_agreement(av_ig, av_cond, 20)
    agg_pool_top = topk_agreement(av_ig, av_pool, 20)

    summary = {
        "analysis": "representation_level_integrated_gradients",
        "checkpoint": str(Path(args_cli.checkpoint).resolve()),
        "checkpoint_sha256": fin.sha256_file(args_cli.checkpoint),
        "split_mode": run_args.split_mode,
        "split_seed": int(run_args.split_seed),
        "optimization_seed": int(run_args.seed),
        "ig_steps": int(args_cli.steps),
        "baseline": (
            "zero 268 pathway-aggregated gene-token representations; "
            "drug atom tokens, dose/time context tokens, token masks, and "
            "all fitted model parameters held fixed"
        ),
        "attention_reference": (
            "final Transformer layer; mean across heads; "
            "dose-query plus time-query attention to each gene token"
        ),
        "secondary_reference": (
            "final learned attention-pooling alpha for each gene token"
        ),
        "selection": {
            "all_test_structures": bool(args_cli.max_structures == 0),
            "per_structure_unique_cell_lines": int(
                args_cli.per_structure
            ),
            "selection_seed": int(args_cli.selection_seed),
            "n_samples": int(len(sample_df)),
            "n_structures": int(
                sample_df["canonical_smiles"].nunique()
            ),
        },
        "prediction_equivalence_abs_delta": mean_sd(
            sample_df["equivalence_abs_delta"].tolist()
        ),
        "attention_prediction_equivalence_abs_delta": mean_sd(
            sample_df["attention_equivalence_abs_delta"].tolist()
        ),
        "ig_completeness_relative_abs": mean_sd(
            sample_df["ig_completeness_relative_abs"].tolist()
        ),
        "sample_level": {
            "spearman_ig_vs_condition_attention": mean_sd(
                sample_df[
                    "ig_spearman_vs_cond_attn"
                ].tolist()
            ),
            "top20_overlap_fraction_ig_vs_condition_attention": mean_sd(
                sample_df[
                    "ig_cond_top20_overlap_fraction"
                ].tolist()
            ),
            "spearman_ig_vs_pool_alpha": mean_sd(
                sample_df[
                    "ig_spearman_vs_pool_alpha"
                ].tolist()
            ),
            "top20_overlap_fraction_ig_vs_pool_alpha": mean_sd(
                sample_df[
                    "ig_pool_top20_overlap_fraction"
                ].tolist()
            ),
        },
        "structure_level": {
            "spearman_ig_vs_condition_attention": mean_sd(
                structure_df[
                    "ig_spearman_vs_cond_attn"
                ].tolist()
            ),
            "top20_overlap_fraction_ig_vs_condition_attention": mean_sd(
                structure_df[
                    "ig_cond_top20_overlap_fraction"
                ].tolist()
            ),
            "spearman_ig_vs_pool_alpha": mean_sd(
                structure_df[
                    "ig_spearman_vs_pool_alpha"
                ].tolist()
            ),
            "top20_overlap_fraction_ig_vs_pool_alpha": mean_sd(
                structure_df[
                    "ig_pool_top20_overlap_fraction"
                ].tolist()
            ),
        },
        "aggregate_gene_ranking": {
            "spearman_ig_vs_condition_attention": spearman(
                av_ig,
                av_cond,
            ),
            "top20_overlap_ig_vs_condition_attention": agg_cond_top,
            "spearman_ig_vs_pool_alpha": spearman(
                av_ig,
                av_pool,
            ),
            "top20_overlap_ig_vs_pool_alpha": agg_pool_top,
        },
    }

    with open(out / "summary.json", "w") as f:
        json.dump(
            summary,
            f,
            indent=2,
            sort_keys=True,
        )

    # Remove partial names only after successful completion.
    for partial in [
        out / "sample_metrics.partial.csv",
        out / "gene_attributions.partial.csv",
    ]:
        if partial.exists():
            partial.unlink()

    print("\n" + "=" * 90)
    print("FINAL IG SUMMARY")
    print("=" * 90)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("=" * 90)
    print("[PASS] final-aligned IG analysis completed")
    print("[OUT]", out)


if __name__ == "__main__":
    main()
