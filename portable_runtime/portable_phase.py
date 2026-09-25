"""Feature construction for the frozen B3/B4 phase-equilibrium models.

This module removes the paper-release path dependency from interactive
inference.  Its equations and feature ordering mirror the frozen production
runner; the model pack remains the authority for the final column order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem

from .model_f_robustness_utils import (
    component_text,
    ensure_family_columns,
    normalize_library_columns,
    rdkit_or_zero,
    static_feature_frames,
)

R = 8.31446261815324


def canonical_smiles(value: str) -> str:
    text = str(value).strip()
    mol = Chem.MolFromSmiles(text) if text else None
    if mol is None:
        raise ValueError("The SMILES string could not be parsed.")
    return Chem.MolToSmiles(mol, canonical=True)


def ideal_branch(x, tm, h):
    x = np.clip(pd.to_numeric(x).to_numpy(float), 1e-12, 1.0)
    tm = pd.to_numeric(tm).to_numpy(float)
    h = pd.to_numeric(h).to_numpy(float) * 1000
    out = 1 / (1 / tm - R / h * np.log(x))
    out[(~np.isfinite(out)) | (out >= tm) | (tm <= 0) | (h <= 0)] = np.nan
    return out


def build_b4_block(points: pd.DataFrame):
    d, cols = normalize_library_columns(points, "ratio")
    d = ensure_family_columns(d, cols)
    mol1 = rdkit_or_zero(d, "SMILES1", "mol1_")
    mol2 = rdkit_or_zero(d, "SMILES2", "mol2_")
    _, router, _ = static_feature_frames(d, cols)
    x1, x2 = pd.to_numeric(d.X1), pd.to_numeric(d.X2)
    t1, t2, h1, h2 = map(lambda c: pd.to_numeric(d[c]), ["T1", "T2", "H1", "H2"])
    hard = pd.to_numeric(d.T_BA)
    ib1, ib2 = ideal_branch(x1, t1, h1), ideal_branch(x2, t2, h2)
    ideal = np.fmax(ib1, ib2)
    comp = pd.DataFrame({"X1": x1, "X2": x2, "min_X": np.minimum(x1, x2)})
    structural_cols = [c for c in router.columns if not any(k in c for k in
        ["baseline", "tpred", "tm_gap", "min_x", "extreme_composition", "large_component_tm_gap"])]
    structural = pd.concat([comp, mol1, mol2, router[structural_cols]], axis=1)
    thermal = pd.DataFrame({
        "T1": t1, "T2": t2, "H1": h1, "H2": h2,
        "weighted_T": x1 * t1 + x2 * t2,
        "weighted_H": x1 * h1 + x2 * h2,
        "abs_T_gap": (t1 - t2).abs(),
        "abs_H_gap": (h1 - h2).abs(),
    })
    idealf = pd.DataFrame({
        "ideal_branch1": ib1, "ideal_branch2": ib2, "ideal_envelope": ideal,
        "ideal_minus_weighted_T": ideal - (x1 * t1 + x2 * t2),
    })
    nonideal = pd.DataFrame({
        "hardmax_branch1": pd.to_numeric(d.hardmax_branch1_K),
        "hardmax_branch2": pd.to_numeric(d.hardmax_branch2_K),
        "T_hardmax": hard,
        "ideal_minus_hardmax": ideal - hard,
        "hardmax_minus_min_T": hard - np.minimum(t1, t2),
        "hardmax_minus_max_T": hard - np.maximum(t1, t2),
    })
    physics_cols = [c for c in router.columns if c not in structural_cols]
    return d, pd.concat([structural, thermal, idealf, nonideal, router[physics_cols]], axis=1)


__all__ = ["build_b4_block", "canonical_smiles", "component_text", "normalize_library_columns"]
