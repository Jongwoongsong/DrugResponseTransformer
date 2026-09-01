#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gat_report", required=True)
    parser.add_argument("--csg2a_report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    gat = json.loads(Path(args.gat_report).read_text())
    csg = json.loads(Path(args.csg2a_report).read_text())

    rows = []
    gat_test = gat["final"]["test"]
    rows.append({
        "model": "GAT-Cross exact",
        "selection": "validation PCC",
        "pcc": gat_test["metrics"]["pcc"],
        "rmse": gat_test["metrics"]["rmse_ln_ic50"],
        "mae": gat_test["metrics"]["mae_ln_ic50"],
        "macro_drug_pcc": gat_test["per_drug_macro_pcc"],
    })
    for key in [
        "original_val_mse",
        "sensitivity_val_pcc",
    ]:
        result = csg["checkpoints"][key]["final"]["test"]
        rows.append({
            "model": "CSG2A exact",
            "selection": key,
            "pcc": result["metrics"]["pcc"],
            "rmse": result["metrics"]["rmse_ln_ic50"],
            "mae": result["metrics"]["mae_ln_ic50"],
            "macro_drug_pcc": result["per_drug_macro_pcc"],
        })

    report = {
        "important_note": (
            "The uploaded GAT-Cross reference has old test PCC "
            "0.902453, so it is not yet proven to be the manuscript "
            "Table-1 GAT result 0.9193."
        ),
        "rows": rows,
    }
    Path(args.output).write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    print("=" * 78)
    print("EXACT-SPLIT GAT / CSG2A SUMMARY")
    print("=" * 78)
    for row in rows:
        print(
            "%-18s %-22s PCC=%.6f RMSE=%.6f MAE=%.6f MACRO=%.6f"
            % (
                row["model"],
                row["selection"],
                row["pcc"],
                row["rmse"],
                row["mae"],
                row["macro_drug_pcc"],
            )
        )
    print("[REPORT]", args.output)


if __name__ == "__main__":
    main()
