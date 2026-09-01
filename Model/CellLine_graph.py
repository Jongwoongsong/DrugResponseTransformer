# CellLine_graph.py (learnable edge type version)
import os
import math
import gzip
import numpy as np
import xml.etree.ElementTree as ET
import networkx as nx
import torch
from torch_geometric.data import Data

# =========================

# =========================

SUBTYPE_LIST = [
    "activation",        # 0
    "inhibition",        # 1
    "phosphorylation",   # 2
    "dephosphorylation", # 3
    "binding",           # 4
    "dissociation",      # 5
    "expression",        # 6
    "repression",
    "ubiquitination",    # 8
    "methylation",       # 9
    "unknown",           # 10 (fallback)
]

SUBTYPE_TO_IDX = {name: idx for idx, name in enumerate(SUBTYPE_LIST)}
NUM_EDGE_TYPES = len(SUBTYPE_LIST)


SUBTYPE_INIT_VALUES = [
    1.0,   # activation
    -1.0,  # inhibition
    0.5,   # phosphorylation
    -0.5,  # dephosphorylation
    0.3,   # binding
    -0.3,  # dissociation
    0.7,   # expression
    -0.7,  # repression
    0.4,   # ubiquitination
    0.2,   # methylation
    0.0,   # unknown
]

# =========================

# =========================
def _find_all(elem, tag):
    return list(elem.findall(f".//{tag}")) + list(elem.findall(f".//{{*}}{tag}"))

def _open_xml_any(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            data = f.read()
        return ET.ElementTree(ET.fromstring(data))
    else:
        return ET.parse(path)

# =========================

# =========================
def parse_kegg_xml(file_path):
    tree = _open_xml_any(file_path)
    root = tree.getroot()

    entry_nodes = {}
    for entry in _find_all(root, 'entry'):
        eid = entry.get('id')
        if not eid:
            continue
        name = entry.get('name') or ""
        genes = [g.replace("hsa:", "") for g in name.split() if g.startswith("hsa:")]
        entry_nodes[eid] = set(genes)

    changed = True
    while changed:
        changed = False
        for entry in _find_all(root, 'entry'):
            eid = entry.get('id')
            if not eid:
                continue
            acc = set(entry_nodes.get(eid, set()))
            for comp in _find_all(entry, 'component'):
                cid = comp.get('id')
                if cid in entry_nodes:
                    before = len(acc)
                    acc |= entry_nodes[cid]
                    if len(acc) > before:
                        changed = True
            entry_nodes[eid] = acc

    graph = nx.DiGraph()
    for eid, genes in entry_nodes.items():
        for gid in genes:
            graph.add_node(gid, type='gene')

    for relation in _find_all(root, 'relation'):
        e1, e2 = relation.get('entry1'), relation.get('entry2')
        if not e1 or not e2:
            continue
        subtypes = [st.get('name', '') for st in _find_all(relation, 'subtype')]
        if e1 in entry_nodes and e2 in entry_nodes:
            for g1 in entry_nodes[e1]:
                for g2 in entry_nodes[e2]:
                    # Gene-level projection can map multiple KGML relation
                    # records onto the same directed gene pair. Preserve all
                    # relation subtypes instead of allowing nx.DiGraph's
                    # last-write semantics to overwrite earlier annotations.
                    if graph.has_edge(g1, g2):
                        previous = graph[g1][g2].get("subtypes", [])
                        merged = list(dict.fromkeys(
                            list(previous) + list(subtypes)
                        ))
                        graph[g1][g2]["subtypes"] = merged
                    else:
                        graph.add_edge(g1, g2, subtypes=list(subtypes))

    return graph

# =========================
# 3) Edge feature - INDEX VERSION
# =========================
def get_edge_type_indices(subtypes):
    """
    Return indices corresponding to the provided subtype names.
    Return all matching indices when multiple subtypes are present.
    """
    indices = []
    for st in subtypes:
        st_lower = st.lower().strip()
        if st_lower in SUBTYPE_TO_IDX:
            indices.append(SUBTYPE_TO_IDX[st_lower])
        else:
            indices.append(SUBTYPE_TO_IDX["unknown"])

    if len(indices) == 0:
        indices.append(SUBTYPE_TO_IDX["unknown"])

    return indices

def get_edge_type_onehot(subtypes):
    """
    Multi-hot encoding allows multiple subtypes per edge.
    Returns a binary vector of length NUM_EDGE_TYPES.
    """
    onehot = [0.0] * NUM_EDGE_TYPES
    indices = get_edge_type_indices(subtypes)
    for idx in indices:
        onehot[idx] = 1.0
    return onehot

# =========================

# =========================
def estimate_expression_with_neighbors(
    graph,
    basal_vector,
    max_iter=2,
    fallback_default=0.0,
    use_median=False
):
    expr = {node: float(basal_vector.get(node, float('nan'))) for node in graph.nodes}

    for _ in range(max_iter):
        updated = False
        new_expr = expr.copy()
        for node in graph.nodes:
            if not math.isnan(expr[node]):
                continue

            n1 = list(graph.predecessors(node)) + list(graph.successors(node))
            n1_vals = [expr[n] for n in n1 if not math.isnan(expr[n])]
            if n1_vals:
                new_expr[node] = float(sum(n1_vals) / max(1, len(n1_vals)))
                updated = True
                continue

            n2 = set()
            for n in n1:
                n2.update(graph.predecessors(n))
                n2.update(graph.successors(n))
            n2.discard(node)
            n2_vals = [expr[n] for n in n2 if not math.isnan(expr[n])]
            if n2_vals:
                new_expr[node] = float(sum(n2_vals) / max(1, len(n2_vals)))
                updated = True

        expr = new_expr
        if not updated:
            break

    path_vals = [v for v in expr.values() if not math.isnan(v)]
    path_agg = float(np.median(path_vals) if (use_median and path_vals) else (np.mean(path_vals) if path_vals else fallback_default))

    for node in graph.nodes:
        if math.isnan(expr[node]):
            expr[node] = path_agg

    return expr

# =========================

# =========================
def nx_to_pyg(graph, basal_vector, use_onehot=True):
    """
    - x: [N, 1] (interpolated basal expression)
    - edge_index: [2, E]
    - edge_attr: [E, NUM_EDGE_TYPES] (multi-hot) or [E, 1] (primary index)
    - node_ids: list of str(entrez)
    """
    nodes = list(graph.nodes)
    if len(nodes) == 0:
        return Data(
            x=torch.empty((0, 1), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, NUM_EDGE_TYPES), dtype=torch.float32),
            node_ids=[]
        )

    node_idx = {n: i for i, n in enumerate(nodes)}
    estimated = estimate_expression_with_neighbors(graph, basal_vector)

    x = torch.tensor([[estimated[n]] for n in nodes], dtype=torch.float32)

    ei, ea = [], []
    for u, v, attr in graph.edges(data=True):
        ei.append([node_idx[u], node_idx[v]])
        subtypes = attr.get("subtypes", [])

        if use_onehot:
            ea.append(get_edge_type_onehot(subtypes))
        else:

            indices = get_edge_type_indices(subtypes)
            ea.append([float(indices[0])])

    edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous() if ei else torch.empty((2, 0), dtype=torch.long)

    if use_onehot:
        edge_attr = torch.tensor(ea, dtype=torch.float32) if ea else torch.empty((0, NUM_EDGE_TYPES), dtype=torch.float32)
    else:
        edge_attr = torch.tensor(ea, dtype=torch.float32) if ea else torch.empty((0, 1), dtype=torch.float32)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_ids=[str(n) for n in nodes]
    )

# =========================

# =========================
def _list_kgml_files(kegg_pathway_dir):
    exts = (".xml", ".kgml", ".xml.gz", ".kgml.gz")
    files = [f for f in os.listdir(kegg_pathway_dir) if f.lower().endswith(exts)]
    files.sort()
    return files

def create_cell_line_graph(basal_df, kegg_pathway_dir, cell_iname, max_pathways=None, use_onehot=True):
    """
    basal_df: index = cell_iname, columns = Entrez(str)
    Returns: (graphs (list[Data]), count)
    """
    cell_iname = str(cell_iname)
    if cell_iname not in basal_df.index:
        return [], 0

    basal_vector = {str(k): float(v) for k, v in basal_df.loc[cell_iname].to_dict().items()}

    graphs = []
    files = _list_kgml_files(kegg_pathway_dir)
    if max_pathways:
        files = files[:max_pathways]

    for idx, fname in enumerate(files):
        fpath = os.path.join(kegg_pathway_dir, fname)
        try:
            nxg = parse_kegg_xml(fpath)
            pyg = nx_to_pyg(nxg, basal_vector, use_onehot=use_onehot)
            if pyg.x.numel() > 0:
                pyg.pathway_idx = idx
                graphs.append(pyg)
        except Exception as e:
            print(f"[WARNING] Failed to process {fname}: {e}")
            continue

    return graphs, len(graphs)