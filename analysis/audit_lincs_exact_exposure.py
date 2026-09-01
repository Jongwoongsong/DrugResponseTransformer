#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem


def canonicalize_smiles(value):
    if pd.isna(value):
        return None

    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None

    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None

    return Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True,
    )


def find_smiles_column(columns):
    lowered = {c.lower(): c for c in columns}

    for candidate in (
        "canonical_smiles",
        "smiles",
    ):
        if candidate in lowered:
            return lowered[candidate]

    for c in columns:
        if "smiles" in c.lower():
            return c

    raise ValueError(
        "No SMILES column found. "
        f"Available columns: {list(columns)[:30]}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Audit exact canonical-structure overlap between "
            "a LINCS pretraining table and a drug-blind test manifest."
        )
    )

    parser.add_argument(
        "--lincs-csv",
        required=True,
        help="LINCS pretraining CSV containing a SMILES column.",
    )

    parser.add_argument(
        "--test-manifest",
        default="splits/drug_blind/test.csv",
        help="Drug-blind test manifest.",
    )

    parser.add_argument(
        "--output-dir",
        default="results/lincs_exposure_recomputed",
        help="Output directory.",
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=100000,
    )

    args = parser.parse_args()

    lincs_path = Path(args.lincs_csv)
    test_path = Path(args.test_manifest)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Test structures
    # --------------------------------------------------------
    test = pd.read_csv(test_path, low_memory=False)
    test_smi_col = find_smiles_column(test.columns)

    test_can = test[test_smi_col].map(canonicalize_smiles)

    if test_can.isna().any():
        raise ValueError(
            "One or more test-manifest SMILES failed RDKit parsing."
        )

    test_structures = set(test_can)

    # --------------------------------------------------------
    # LINCS structures
    # --------------------------------------------------------
    header = pd.read_csv(lincs_path, nrows=0)
    lincs_smi_col = find_smiles_column(header.columns)

    raw_counts = Counter()
    total_rows = 0

    for chunk in pd.read_csv(
        lincs_path,
        usecols=[lincs_smi_col],
        chunksize=args.chunksize,
        low_memory=False,
    ):
        total_rows += len(chunk)

        counts = (
            chunk[lincs_smi_col]
            .astype(str)
            .str.strip()
            .value_counts()
        )

        raw_counts.update(counts.to_dict())

    canonical_counts = Counter()
    invalid = []

    for raw, n in raw_counts.items():
        c = canonicalize_smiles(raw)

        if c is None:
            invalid.append(raw)
        else:
            canonical_counts[c] += int(n)

    # --------------------------------------------------------
    # Exposure table
    # --------------------------------------------------------
    rows = []

    for smi in sorted(test_structures):
        n_profiles = int(canonical_counts.get(smi, 0))

        rows.append({
            "canonical_smiles": smi,
            "lincs_exposed": n_profiles > 0,
            "n_lincs_profiles": n_profiles,
        })

    membership = pd.DataFrame(rows)

    n_exposed = int(membership["lincs_exposed"].sum())
    n_unexposed = len(membership) - n_exposed

    membership.to_csv(
        outdir / "exact_exposure_membership.csv",
        index=False,
    )

    summary = {
        "pretraining_rows": int(total_rows),
        "pretraining_unique_raw_smiles": len(raw_counts),
        "pretraining_unique_canonical_structures":
            len(canonical_counts),
        "invalid_unique_smiles": len(invalid),
        "test_structures": len(membership),
        "lincs_exposed": n_exposed,
        "lincs_unexposed": n_unexposed,
        "exposure_fraction":
            n_exposed / len(membership),
        "canonicalization": (
            "RDKit MolFromSmiles -> "
            "MolToSmiles(canonical=True, "
            "isomericSmiles=True)"
        ),
    }

    with open(
        outdir / "summary.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
