#!/usr/bin/env python3
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
V1 = HERE / "run_final_mixed_ig.py"

spec = importlib.util.spec_from_file_location("ig_v1", str(V1))
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def integrated_gradients_gene_tokens_v2(
    model,
    token_batch,
    gene_positions,
    time,
    dose,
    steps,
):
    baseline_mode = os.environ.get(
        "IG_BASELINE_MODE",
        "gene_embedding",
    )
    quadrature = os.environ.get(
        "IG_QUADRATURE",
        "gauss_legendre",
    )

    actual = token_batch.tokens.detach()
    baseline = actual.clone()

    if baseline_mode == "zero":
        baseline[:, gene_positions, :] = 0.0

    elif baseline_mode == "gene_embedding":
        if not hasattr(model, "gene_embedding"):
            raise RuntimeError(
                "model.gene_embedding is missing"
            )

        gene_emb = model.gene_embedding.detach().to(
            device=actual.device,
            dtype=actual.dtype,
        )

        if gene_emb.dim() != 2:
            raise RuntimeError(
                "Expected gene_embedding [268,D], got {}".format(
                    tuple(gene_emb.shape)
                )
            )

        if gene_emb.shape[0] != len(gene_positions):
            raise RuntimeError(
                "gene_embedding rows={} but gene positions={}".format(
                    gene_emb.shape[0],
                    len(gene_positions),
                )
            )

        if gene_emb.shape[1] != actual.shape[-1]:
            raise RuntimeError(
                "gene_embedding dim={} but token dim={}".format(
                    gene_emb.shape[1],
                    actual.shape[-1],
                )
            )

        baseline[:, gene_positions, :] = gene_emb.unsqueeze(0)

    else:
        raise ValueError(
            "Unknown IG_BASELINE_MODE={}".format(baseline_mode)
        )

    delta = actual - baseline

    with torch.no_grad():
        pred_actual = base.predict_from_tokens(
            model,
            token_batch,
            actual,
            time,
            dose,
        )[0].item()

        pred_baseline = base.predict_from_tokens(
            model,
            token_batch,
            baseline,
            time,
            dose,
        )[0].item()

    if quadrature == "gauss_legendre":
        # Gauss-Legendre nodes/weights over [-1,1],
        # transformed to [0,1].
        nodes, weights = np.polynomial.legendre.leggauss(
            int(steps)
        )
        alphas = (nodes + 1.0) / 2.0
        weights = weights / 2.0

        integral = torch.zeros(
            (1, len(gene_positions), actual.shape[-1]),
            device=actual.device,
            dtype=actual.dtype,
        )

        for alpha, weight in zip(alphas, weights):
            x = (
                baseline
                + float(alpha) * delta
            ).detach().requires_grad_(True)

            pred = base.predict_from_tokens(
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

            integral = integral + (
                float(weight)
                * grad[:, gene_positions, :]
            )

    elif quadrature == "trapezoid":
        gradients = []

        for step_i in range(steps + 1):
            alpha = float(step_i) / float(steps)

            x = (
                baseline
                + alpha * delta
            ).detach().requires_grad_(True)

            pred = base.predict_from_tokens(
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

    else:
        raise ValueError(
            "Unknown IG_QUADRATURE={}".format(quadrature)
        )

    delta_gene = delta[:, gene_positions, :]
    ig = delta_gene * integral

    ig_signed = ig.sum(dim=-1)[0]
    ig_abs = ig.abs().sum(dim=-1)[0]

    prediction_delta = pred_actual - pred_baseline
    ig_sum = float(ig_signed.sum().item())
    residual = ig_sum - prediction_delta

    print(
        "[IG V2 CONFIG] baseline={} quadrature={} steps={}".format(
            baseline_mode,
            quadrature,
            steps,
        ),
        flush=True,
    )

    print(
        "[IG COMPLETENESS] "
        "actual={:.8f} baseline={:.8f} delta={:.8f} "
        "ig_sum={:.8f} residual={:.8f}".format(
            pred_actual,
            pred_baseline,
            prediction_delta,
            ig_sum,
            residual,
        ),
        flush=True,
    )

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


# Replace only the numerical IG integrator.
# Everything else remains the already-audited v1 implementation.
base.integrated_gradients_gene_tokens = (
    integrated_gradients_gene_tokens_v2
)


def find_cli_arg(name):
    if name not in sys.argv:
        return None
    i = sys.argv.index(name)
    if i + 1 >= len(sys.argv):
        return None
    return sys.argv[i + 1]


if __name__ == "__main__":
    baseline_mode = os.environ.get(
        "IG_BASELINE_MODE",
        "gene_embedding",
    )
    quadrature = os.environ.get(
        "IG_QUADRATURE",
        "gauss_legendre",
    )

    print(
        "[IG V2] baseline={} quadrature={}".format(
            baseline_mode,
            quadrature,
        ),
        flush=True,
    )

    base.main()

    # Correct/extend provenance in summary.json because
    # v1's text describes the original zero/trapezoid implementation.
    out_arg = find_cli_arg("--out")

    if out_arg:
        summary_path = Path(out_arg).resolve() / "summary.json"

        if summary_path.exists():
            summary = json.loads(summary_path.read_text())

            summary["analysis"] = (
                "representation_level_integrated_gradients_v2"
            )
            summary["baseline_mode"] = baseline_mode
            summary["quadrature"] = quadrature

            if baseline_mode == "gene_embedding":
                summary["baseline"] = (
                    "learned gene-identity embeddings for the 268 "
                    "pathway-aggregated gene-token positions; "
                    "drug atom tokens, dose/time context, token masks, "
                    "and fitted parameters held fixed"
                )
            else:
                summary["baseline"] = (
                    "zero vectors for the 268 pathway-aggregated "
                    "gene-token positions; drug atom tokens, dose/time "
                    "context, token masks, and fitted parameters held fixed"
                )

            summary_path.write_text(
                json.dumps(
                    summary,
                    indent=2,
                    sort_keys=True,
                )
            )
