# --- DES_Paper_Release ---
# 模块: 06_residual_correction
# 作用: 纯组分训练流水线文件 feature_chemberta.py
# 原始路径: C:\code2026\experiments\experiments\f4_structure_aware_residual\src\feature_chemberta.py
# 说明: 这是论文开源整理副本。脚本内硬编码路径多为 D:\DES_Project，
#       本机镜像一般在 C:\code2026。本整理包不重跑 ORCA/CREST/70 万 SLE。
# -------------------------
"""Optional ChemBERTa SMILES encoder for F4 Model C."""

from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np

_CHEMBERTA_AVAILABLE = False
_CHEMBERTA_ERROR = ""
_torch = None
_transformers = None

try:
    import torch as _torch
    import transformers as _transformers

    _CHEMBERTA_AVAILABLE = True
except Exception as e:
    _CHEMBERTA_ERROR = str(e)

_EMBEDDING_CACHE: dict[tuple[str, str, str], np.ndarray] = {}


def is_chemberta_available() -> bool:
    return _CHEMBERTA_AVAILABLE


def chemberta_unavailable_message() -> str:
    base = (
        "Model C 需要 torch 与 transformers。\n"
        "安装: pip install torch transformers\n"
        "或: pip install -r ../../requirements-dl.txt && pip install transformers"
    )
    if _CHEMBERTA_ERROR:
        base += f"\n当前环境错误: {_CHEMBERTA_ERROR}"
    return base


@lru_cache(maxsize=2)
def _load_model_and_tokenizer(model_name: str):
    if not _CHEMBERTA_AVAILABLE:
        raise ImportError(chemberta_unavailable_message())
    tokenizer = _transformers.AutoTokenizer.from_pretrained(model_name)
    model = _transformers.AutoModel.from_pretrained(model_name)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return tokenizer, model


def encode_smiles_batch(
    smiles_list: list[str],
    model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
    pooling: str = "cls",
    device: str = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    """Encode SMILES to embeddings; invalid SMILES -> zero vector."""
    if not _CHEMBERTA_AVAILABLE:
        raise ImportError(chemberta_unavailable_message())

    pooling = pooling.lower()
    if pooling not in ("cls", "mean"):
        raise ValueError("pooling 须为 cls 或 mean")

    tokenizer, model = _load_model_and_tokenizer(model_name)
    dev = _torch.device(device)
    model = model.to(dev)

    n = len(smiles_list)
    hidden = model.config.hidden_size
    out = np.zeros((n, hidden), dtype=np.float64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_smiles = smiles_list[start:end]
        texts = []
        valid_idx = []
        for j, smi in enumerate(batch_smiles):
            if isinstance(smi, str) and smi.strip():
                texts.append(smi.strip())
                valid_idx.append(j)
            else:
                warnings.warn("Empty SMILES in ChemBERTa batch; using zero embedding", stacklevel=2)

        if not texts:
            continue

        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(dev) for k, v in enc.items()}

        with _torch.no_grad():
            outputs = model(**enc)
            hidden_states = outputs.last_hidden_state

        if pooling == "cls":
            emb = hidden_states[:, 0, :].cpu().numpy()
        else:
            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (hidden_states * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            emb = (summed / counts).cpu().numpy()

        for local_j, vec in zip(valid_idx, emb):
            out[start + local_j] = vec.astype(np.float64)

    return out


def encode_smiles_cached(
    smiles: str,
    model_name: str,
    pooling: str,
    device: str = "cpu",
) -> np.ndarray:
    key = (smiles, model_name, pooling)
    if key not in _EMBEDDING_CACHE:
        _EMBEDDING_CACHE[key] = encode_smiles_batch([smiles], model_name, pooling, device)[0]
    return _EMBEDDING_CACHE[key]


def build_chemberta_matrix(
    smiles_series,
    prefix: str,
    *,
    model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
    pooling: str = "cls",
    device: str = "cpu",
) -> "pd.DataFrame":
    import pandas as pd

    smiles_list = [str(s) if pd.notna(s) else "" for s in smiles_series]
    emb = encode_smiles_batch(smiles_list, model_name, pooling, device)
    cols = [f"{prefix}{i}" for i in range(emb.shape[1])]
    return pd.DataFrame(emb, columns=cols, index=smiles_series.index)
