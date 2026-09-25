# --- DES_Paper_Release ---
# 模块: 06_residual_correction
# 作用: 纯组分训练流水线文件 feature_fingerprint.py
# 原始路径: C:\code2026\experiments\experiments\f4_structure_aware_residual\src\feature_fingerprint.py
# 说明: 这是论文开源整理副本。脚本内硬编码路径多为 D:\DES_Project，
#       本机镜像一般在 C:\code2026。本整理包不重跑 ORCA/CREST/70 万 SLE。
# -------------------------
"""Morgan fingerprint features for F4 Model B."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError as e:
    raise ImportError(
        "Model B 需要 RDKit。请安装: pip install rdkit-pypi 或 conda install -c conda-forge rdkit"
    ) from e


def fingerprint_column_names(prefix: str, n_bits: int) -> list[str]:
    return [f"{prefix}{i}" for i in range(n_bits)]


def smiles_to_morgan_fp(smiles: str, radius: int, n_bits: int) -> np.ndarray:
    out = np.zeros(n_bits, dtype=np.float64)
    if not isinstance(smiles, str) or not smiles.strip():
        warnings.warn(f"Empty SMILES for fingerprint prefix; using zero vector", stacklevel=2)
        return out

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        warnings.warn(f"Invalid SMILES for fingerprint; using zero vector", stacklevel=2)
        return out

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float64)
    for i in range(n_bits):
        arr[i] = float(fp[i])
    return arr


def build_fingerprint_matrix(
    smiles_series: pd.Series,
    prefix: str,
    *,
    radius: int = 2,
    n_bits: int = 1024,
) -> pd.DataFrame:
    cols = fingerprint_column_names(prefix, n_bits)
    rows = [
        smiles_to_morgan_fp(str(s) if pd.notna(s) else "", radius, n_bits) for s in smiles_series
    ]
    return pd.DataFrame(rows, columns=cols, index=smiles_series.index)
