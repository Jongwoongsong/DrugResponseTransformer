#!/usr/bin/env python3

import argparse
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from Model.CellLine_graph import (
    NUM_EDGE_TYPES,
    SUBTYPE_LIST,
    SUBTYPE_TO_IDX,
    _list_kgml_files,
    nx_to_pyg,
    parse_kegg_xml,
)


EXPECTED_CELLS = 218
EXPECTED_PATHWAYS = 31


def directed_to_directionaware(graph):
    """
    Input
      edge_attr[:, 0:11] = KEGG relation multi-hot

    Output
      edge_attr[:, 0:11] = same relation identity
      edge_attr[:, 11]   = direction flag
          0 = original KEGG
          1 = computational reverse

    Reverse edge is added only if that reverse pair is not already
    present as an original KEGG edge.
    """

    edge_index = (
        graph.edge_index.detach()
        .long()
        .cpu()
        .contiguous()
    )

    edge_attr = (
        graph.edge_attr.detach()
        .float()
        .cpu()
        .contiguous()
    )

    if edge_attr.dim() == 1:
        edge_attr = edge_attr[:, None]

    if (
        edge_attr.dim() != 2
        or edge_attr.size(0) != edge_index.size(1)
        or edge_attr.size(1) != NUM_EDGE_TYPES
    ):
        raise RuntimeError(
            "Expected directed edge_attr [E,%d], got %r"
            % (
                NUM_EDGE_TYPES,
                tuple(edge_attr.shape),
            )
        )

    original_count = int(edge_index.size(1))

    original_flag = torch.zeros(
        original_count,
        1,
        dtype=edge_attr.dtype,
    )

    original_attr = torch.cat(
        [
            edge_attr,
            original_flag,
        ],
        dim=1,
    )

    pairs = {
        (
            int(edge_index[0, i]),
            int(edge_index[1, i]),
        )
        for i in range(original_count)
    }

    add = [
        i
        for i in range(original_count)
        if (
            int(edge_index[0, i])
            != int(edge_index[1, i])
            and (
                int(edge_index[1, i]),
                int(edge_index[0, i]),
            ) not in pairs
        )
    ]

    graph.edge_index = edge_index
    graph.edge_attr = original_attr

    if not add:
        return graph, original_count, 0

    idx = torch.tensor(
        add,
        dtype=torch.long,
    )

    reverse_index = (
        edge_index[:, idx]
        .flip(0)
        .contiguous()
    )

    reverse_relation = (
        edge_attr.index_select(
            0,
            idx,
        )
        .clone()
    )

    reverse_flag = torch.ones(
        len(add),
        1,
        dtype=edge_attr.dtype,
    )

    reverse_attr = torch.cat(
        [
            reverse_relation,
            reverse_flag,
        ],
        dim=1,
    )

    graph.edge_index = torch.cat(
        [
            edge_index,
            reverse_index,
        ],
        dim=1,
    )

    graph.edge_attr = torch.cat(
        [
            original_attr,
            reverse_attr,
        ],
        dim=0,
    )

    return graph, original_count, len(add)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--perturbed_csv",
        required=True,
    )
    ap.add_argument(
        "--basal_csv",
        required=True,
    )
    ap.add_argument(
        "--kegg_pathway_dir",
        required=True,
    )
    ap.add_argument(
        "--output",
        required=True,
    )
    ap.add_argument(
        "--report",
        required=True,
    )
    ap.add_argument(
        "--num_pathways",
        type=int,
        default=31,
    )

    args = ap.parse_args()

    if NUM_EDGE_TYPES != 11:
        raise RuntimeError(
            "Expected 11 KEGG relation channels, got %d"
            % NUM_EDGE_TYPES
        )

    # Preserve the already-established alias correction regardless
    # of whether it lives in the parser or cache-builder version.
    SUBTYPE_TO_IDX[
        "binding/association"
    ] = SUBTYPE_TO_IDX[
        "binding"
    ]

    # ------------------------------------------------------------
    # Fail-fast: make sure we really imported the UNION parser.
    # ------------------------------------------------------------

    parser_file = inspect.getfile(
        parse_kegg_xml
    )

    parser_source = inspect.getsource(
        parse_kegg_xml
    )

    print(
        "[PARSER IMPORT]",
        parser_file,
        flush=True,
    )

    if (
        "has_edge" not in parser_source
        or "subtypes" not in parser_source
    ):
        raise RuntimeError(
            "Imported parser does not appear to contain "
            "same-pair subtype-union handling"
        )

    print(
        "[PASS] union parser import verified",
        flush=True,
    )

    # ------------------------------------------------------------
    # LINCS cell list
    # ------------------------------------------------------------

    header = pd.read_csv(
        args.perturbed_csv,
        nrows=0,
    )

    if "cell_iname" not in header.columns:
        raise RuntimeError(
            "cell_iname not found in PGE CSV"
        )

    cells = (
        pd.read_csv(
            args.perturbed_csv,
            usecols=["cell_iname"],
            low_memory=False,
        )["cell_iname"]
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    print(
        "[LINCS CELLS]",
        len(cells),
        flush=True,
    )

    if len(cells) != EXPECTED_CELLS:
        raise RuntimeError(
            "Expected %d LINCS cells, got %d"
            % (
                EXPECTED_CELLS,
                len(cells),
            )
        )

    # ------------------------------------------------------------
    # Basal expression
    # ------------------------------------------------------------

    basal_raw = pd.read_csv(
        args.basal_csv,
        low_memory=False,
    )

    if "cell_iname" not in basal_raw.columns:
        raise RuntimeError(
            "basal_final.csv lacks cell_iname"
        )

    gene_cols = [
        c
        for c in basal_raw.columns
        if str(c).isdigit()
    ]

    basal = (
        basal_raw[
            ["cell_iname"] + gene_cols
        ]
        .assign(
            cell_iname=lambda x:
                x["cell_iname"]
                .astype(str)
                .str.strip()
        )
        .drop_duplicates(
            subset=["cell_iname"]
        )
        .set_index("cell_iname")
    )

    basal.columns = [
        str(c)
        for c in basal.columns
    ]

    basal = basal.apply(
        pd.to_numeric,
        errors="coerce",
    )

    print(
        "[BASAL]",
        {
            "cells": int(basal.shape[0]),
            "genes": int(basal.shape[1]),
        },
        flush=True,
    )

    missing = sorted(
        set(cells)
        - set(basal.index.astype(str))
    )

    if missing:
        raise RuntimeError(
            "Basal table misses %d LINCS cells: %r"
            % (
                len(missing),
                missing[:10],
            )
        )

    # ------------------------------------------------------------
    # Parse KEGG ONCE with the union-fixed parser
    # ------------------------------------------------------------

    files = _list_kgml_files(
        args.kegg_pathway_dir
    )[:args.num_pathways]

    if len(files) != EXPECTED_PATHWAYS:
        raise RuntimeError(
            "Expected 31 pathways, got %d"
            % len(files)
        )

    parsed = []

    for pathway_idx, filename in enumerate(
        files
    ):
        nxg = parse_kegg_xml(
            os.path.join(
                args.kegg_pathway_dir,
                filename,
            )
        )

        parsed.append(
            (
                pathway_idx,
                filename,
                nxg,
            )
        )

    print(
        "[KEGG] parsed once:",
        len(parsed),
        flush=True,
    )

    # ------------------------------------------------------------
    # Build all 218 cell graph sequences
    # ------------------------------------------------------------

    cache = {}

    total_original = 0
    total_reverse = 0

    relation_original = torch.zeros(
        NUM_EDGE_TYPES,
        dtype=torch.long,
    )

    relation_reverse = torch.zeros(
        NUM_EDGE_TYPES,
        dtype=torch.long,
    )

    node_occurrences = []

    for ci, cell_id in enumerate(cells):

        if ci % 25 == 0:
            print(
                "[CELL] %d/%d"
                % (
                    ci,
                    len(cells),
                ),
                flush=True,
            )

        row = basal.loc[
            cell_id
        ]

        basal_vector = {
            str(gene_id): float(value)
            for gene_id, value
            in row.to_dict().items()
            if pd.notna(value)
        }

        sequence = []

        for (
            pathway_idx,
            filename,
            nxg,
        ) in parsed:

            graph = nx_to_pyg(
                nxg,
                basal_vector,
                use_onehot=True,
            )

            if graph.x.numel() == 0:
                raise RuntimeError(
                    "Empty graph: cell=%s pathway=%s"
                    % (
                        cell_id,
                        filename,
                    )
                )

            graph.pathway_idx = (
                pathway_idx
            )

            graph.node_ids = [
                str(x)
                for x in graph.node_ids
            ]

            graph, n_original, n_reverse = (
                directed_to_directionaware(
                    graph
                )
            )

            direction = (
                graph.edge_attr[
                    :, NUM_EDGE_TYPES
                ]
            )

            if not bool(
                (
                    (direction == 0)
                    | (direction == 1)
                ).all()
            ):
                raise RuntimeError(
                    "Bad direction flag"
                )

            original_mask = (
                direction == 0
            )

            reverse_mask = (
                direction == 1
            )

            relation_original += (
                graph.edge_attr[
                    original_mask,
                    :NUM_EDGE_TYPES,
                ]
                .gt(0)
                .long()
                .sum(dim=0)
            )

            relation_reverse += (
                graph.edge_attr[
                    reverse_mask,
                    :NUM_EDGE_TYPES,
                ]
                .gt(0)
                .long()
                .sum(dim=0)
            )

            total_original += (
                n_original
            )

            total_reverse += (
                n_reverse
            )

            sequence.append(
                graph
            )

        if len(sequence) != EXPECTED_PATHWAYS:
            raise RuntimeError(
                "%s produced %d pathways"
                % (
                    cell_id,
                    len(sequence),
                )
            )

        cache[cell_id] = sequence

        node_occurrences.append(
            sum(
                len(g.node_ids)
                for g in sequence
            )
        )

    # ------------------------------------------------------------
    # Save RAW dict: compatible with faithful PGE loader
    # ------------------------------------------------------------

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output.exists():
        raise RuntimeError(
            "Output already exists: %s"
            % output
        )

    tmp = output.with_suffix(
        output.suffix + ".tmp"
    )

    torch.save(
        cache,
        tmp,
    )

    os.replace(
        tmp,
        output,
    )

    relation_original_dict = {
        SUBTYPE_LIST[i]:
            int(relation_original[i])
        for i in range(
            NUM_EDGE_TYPES
        )
    }

    relation_reverse_dict = {
        SUBTYPE_LIST[i]:
            int(relation_reverse[i])
        for i in range(
            NUM_EDGE_TYPES
        )
    }

    report = {
        "cells":
            len(cache),

        "num_pathways":
            EXPECTED_PATHWAYS,

        "graphs":
            len(cache)
            * EXPECTED_PATHWAYS,

        "basal_genes":
            int(basal.shape[1]),

        "parser":
            parser_file,

        "parser_relation_merge":
            "union_same_directed_gene_pair",

        "binding_alias":
            "binding/association -> binding",

        "edge_direction":
            "direction_aware_bidirectional",

        "relation_channels":
            NUM_EDGE_TYPES,

        "direction_flag_column":
            NUM_EDGE_TYPES,

        "direction_semantics": {
            "0": "original_KEGG",
            "1": "computational_reverse",
        },

        "original_edges_total":
            total_original,

        "reverse_edges_added_total":
            total_reverse,

        "node_occurrence_min":
            int(min(node_occurrences)),

        "node_occurrence_max":
            int(max(node_occurrences)),

        "relation_type_counts_original":
            relation_original_dict,

        "relation_type_counts_reverse":
            relation_reverse_dict,

        "output":
            str(output),
    }

    Path(
        args.report
    ).write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
    )

    print(
        "[LINCS UNION DA CACHE AUDIT] "
        + json.dumps(
            report,
            sort_keys=True,
        ),
        flush=True,
    )

    # reload bytes once
    check = torch.load(
        output,
        map_location="cpu",
    )

    if (
        not isinstance(check, dict)
        or len(check) != EXPECTED_CELLS
    ):
        raise RuntimeError(
            "Saved cache reload failed"
        )

    if not all(
        isinstance(v, (list, tuple))
        and len(v) == EXPECTED_PATHWAYS
        for v in check.values()
    ):
        raise RuntimeError(
            "Saved cache pathway-count validation failed"
        )

    print(
        "[PASS] union-fixed LINCS218 "
        "direction-aware cache saved and reloaded"
    )


if __name__ == "__main__":
    main()
