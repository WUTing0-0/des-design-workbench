# --- DES_Paper_Release ---
# 模块: 06_residual_correction
# 作用: 纯组分训练流水线文件 feature_llm_hybrid.py
# 原始路径: C:\code2026\experiments\experiments\f4_structure_aware_residual\src\feature_llm_hybrid.py
# 说明: 这是论文开源整理副本。脚本内硬编码路径多为 D:\DES_Project，
#       本机镜像一般在 C:\code2026。本整理包不重跑 ORCA/CREST/70 万 SLE。
# -------------------------
"""Hybrid LLM feature builders for residual stacking."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_chemberta import build_chemberta_matrix, is_chemberta_available


def _openai_api_key() -> str | None:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return key or None


def is_api_embedding_available() -> bool:
    return _openai_api_key() is not None


def _post_openai_embedding(text: str, model: str) -> list[float]:
    api_key = _openai_api_key()
    if not api_key:
        raise RuntimeError("未检测到 OPENAI_API_KEY，无法使用 API embedding")

    payload = json.dumps({"input": text, "model": model}).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/embeddings",
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"OpenAI embedding API 失败: {e.code} {detail}") from e
    except Exception as e:
        raise RuntimeError(f"OpenAI embedding 请求失败: {e}") from e

    return body["data"][0]["embedding"]


def build_api_embedding_matrix(
    smiles_series: pd.Series,
    prefix: str,
    *,
    model_name: str,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    cache: dict[str, list[float]] = {}
    if cache_path and cache_path.is_file():
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)

    unique_smiles = []
    for v in smiles_series.astype(str).fillna("").tolist():
        s = v.strip()
        if s and s not in cache:
            unique_smiles.append(s)

    for s in unique_smiles:
        cache[s] = _post_openai_embedding(s, model_name)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)

    dim = 0
    for vec in cache.values():
        dim = len(vec)
        break
    if dim == 0:
        raise RuntimeError("API embedding 结果为空")

    cols = [f"{prefix}{i}" for i in range(dim)]
    rows = []
    for s in smiles_series.astype(str).fillna("").tolist():
        ss = s.strip()
        if ss and ss in cache:
            rows.append(np.asarray(cache[ss], dtype=np.float64))
        else:
            rows.append(np.zeros(dim, dtype=np.float64))
    return pd.DataFrame(rows, columns=cols, index=smiles_series.index)


def build_llm_feature_pair(
    smiles1: pd.Series,
    smiles2: pd.Series,
    *,
    source: str = "hybrid",
    local_model: str = "seyonec/ChemBERTa-zinc-base-v1",
    pooling: str = "cls",
    device: str = "cpu",
    api_model: str = "text-embedding-3-small",
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, str]:
    """Return concatenated LLM features and actual source used."""
    src = source.lower()
    if src not in ("local", "api", "hybrid"):
        raise ValueError("source must be local|api|hybrid")

    def use_local() -> pd.DataFrame:
        if not is_chemberta_available():
            raise RuntimeError("本地编码器不可用：请安装 torch + transformers")
        e1 = build_chemberta_matrix(smiles1, "llm1_", model_name=local_model, pooling=pooling, device=device)
        e2 = build_chemberta_matrix(smiles2, "llm2_", model_name=local_model, pooling=pooling, device=device)
        return pd.concat([e1, e2], axis=1)

    def use_api() -> pd.DataFrame:
        if not is_api_embedding_available():
            raise RuntimeError("API embedding 不可用：请设置 OPENAI_API_KEY")
        c1 = cache_dir / "api_embed_smiles1.json" if cache_dir else None
        c2 = cache_dir / "api_embed_smiles2.json" if cache_dir else None
        e1 = build_api_embedding_matrix(smiles1, "llm1_", model_name=api_model, cache_path=c1)
        e2 = build_api_embedding_matrix(smiles2, "llm2_", model_name=api_model, cache_path=c2)
        return pd.concat([e1, e2], axis=1)

    if src == "local":
        return use_local(), "local"
    if src == "api":
        return use_api(), "api"

    if is_api_embedding_available():
        return use_api(), "api"
    return use_local(), "local"
