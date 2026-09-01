# -*- coding: utf-8 -*-
import os
import logging
from typing import List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType as Hyb, ChiralType as Chi
import networkx as nx
import scipy

import torch.nn.functional as F

logger = logging.getLogger(__name__)

def one_hot(x, choices: List):
    v = [0.0] * len(choices)
    try:
        v[choices.index(x)] = 1.0
    except ValueError:
        pass
    return v


ATOM_TYPES     = [1,6,7,8,9,15,16,17,19,23,25,26,29,30,33,34,35,44,45,50,51,52,53,73,74,77,78,79,80,81]
DEGREES        = list(range(0, 6))
FORMAL_CHARGES = [-2,-1,0,1,2]
NUM_HS         = list(range(0, 5))
HYBS           = [Hyb.SP, Hyb.SP2, Hyb.SP3, Hyb.SP3D, Hyb.SP3D2]
CHIRALS        = [Chi.CHI_UNSPECIFIED, Chi.CHI_TETRAHEDRAL_CW, Chi.CHI_TETRAHEDRAL_CCW, Chi.CHI_OTHER]

def atom_features(atom: Chem.Atom):
    feats = []
    feats += one_hot(atom.GetAtomicNum(), ATOM_TYPES)
    feats += one_hot(atom.GetTotalDegree(), DEGREES)
    feats += one_hot(atom.GetFormalCharge(), FORMAL_CHARGES)
    feats += one_hot(atom.GetTotalNumHs(), NUM_HS)
    feats += one_hot(atom.GetHybridization(), HYBS)
    feats.append(1.0 if atom.GetIsAromatic() else 0.0)
    feats.append(1.0 if atom.IsInRing() else 0.0)
    feats += one_hot(atom.GetChiralTag(), CHIRALS)
    return feats

def calculate_position_encoding(graph_data: Data, k: int = 1, skip_first: bool = False, standardize: bool = True) -> torch.Tensor:
    """
    Use eigenvectors of the normalized Laplacian as positional encodings.
    - If skip_first=True, skip the constant eigenvector associated with lambda=0.
    - If standardize=True, standardize each positional-encoding dimension.
    Returns: [N, k] (float32, graph_data.x.device)
    """
    N = int(graph_data.x.size(0))
    device = graph_data.x.device

    if N == 0 or graph_data.edge_index is None or graph_data.edge_index.numel() == 0:
        return torch.zeros((N, k), dtype=torch.float, device=device)

    edge_index = graph_data.edge_index

    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(N))
    nx_graph.add_edges_from(edge_index.t().tolist())


    lap = nx.normalized_laplacian_matrix(nx_graph).toarray()
    eigvals, eigvecs = scipy.linalg.eigh(lap)

    start_idx = 1 if skip_first else 0
    vecs = eigvecs[:, start_idx:start_idx + k]

    if vecs.shape[1] < k:
        pad = np.zeros((N, k - vecs.shape[1]), dtype=vecs.dtype)
        vecs = np.concatenate([vecs, pad], axis=1)


    for j in range(vecs.shape[1]):
        if np.sum(vecs[:, j]) < 0:
            vecs[:, j] = -vecs[:, j]


    if standardize:
        for j in range(vecs.shape[1]):
            col = vecs[:, j]
            std = col.std()
            if std > 1e-12:
                vecs[:, j] = (col - col.mean()) / std
            else:
                vecs[:, j] = col - col.mean()

    pe = torch.tensor(vecs, dtype=torch.float, device=device)
    return pe

def smiles_to_graph(smiles: str) -> Data:
    """
    Construct a molecular graph using RDKit.
    - Node features: 57-dimensional atom one-hot/property features.
    - Edges are added bidirectionally.
    - edge_attr: four-dimensional bond-type one-hot encoding.
    - When possible, precompute positional encodings during graph construction and store them in data.pe for cache reuse.
    """
    if not isinstance(smiles, str) or smiles.strip() == "":
        raise ValueError(f"empty or invalid SMILES: {smiles!r}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    x = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float)

    edge_indices = []
    edge_features = []

    bond_type_dict = {
        Chem.BondType.SINGLE:  0,
        Chem.BondType.DOUBLE:  1,
        Chem.BondType.TRIPLE:  2,
        Chem.BondType.AROMATIC:3,
    }

    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        idx = bond_type_dict.get(bond.GetBondType(), 0)
        bond_feature = [0, 0, 0, 0]
        bond_feature[idx] = 1


        edge_indices.append([u, v]); edge_features.append(bond_feature)
        edge_indices.append([v, u]); edge_features.append(bond_feature)

    if len(edge_indices) == 0:
        edge_index = torch.empty((2,0), dtype=torch.long)
        edge_attr  = torch.empty((0,4), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr  = torch.tensor(edge_features, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


    try:
        pe = calculate_position_encoding(data, k=1, skip_first=False, standardize=True)
        data.pe = pe.cpu()
    except Exception as e:

        logger.debug(f"[PE-precompute] failed for SMILES={smiles[:24]}...: {e}")

    return data