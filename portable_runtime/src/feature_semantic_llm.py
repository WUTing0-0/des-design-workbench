# --- DES_Paper_Release ---
# 模块: 06_residual_correction
# 作用: 纯组分训练流水线文件 feature_semantic_llm.py
# 原始路径: C:\code2026\experiments\experiments\f4_structure_aware_residual\src\feature_semantic_llm.py
# 说明: 这是论文开源整理副本。脚本内硬编码路径多为 D:\DES_Project，
#       本机镜像一般在 C:\code2026。本整理包不重跑 ORCA/CREST/70 万 SLE。
# -------------------------
"""Optional API-based semantic embeddings for the LLM branch."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import requests
except Exception:  # pragma: no cover - urllib fallback
    requests = None


def openai_api_key() -> str | None:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return key or None


def is_openai_embedding_available() -> bool:
    return openai_api_key() is not None


def _post_openai_embedding(text: str, model: str) -> list[float]:
    api_key = openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; cannot build API embeddings.")

    if requests is not None:
        try:
            resp = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                json={"input": text, "model": model},
                timeout=60,
            )
            if resp.status_code >= 400:
                raise RuntimeError(f"OpenAI embedding API failed: {resp.status_code} {resp.text}")
            return resp.json()["data"][0]["embedding"]
        except Exception as e:
            raise RuntimeError(f"OpenAI embedding request failed: {e}") from e

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
        raise RuntimeError(f"OpenAI embedding API failed: {e.code} {detail}") from e
    except Exception as e:
        raise RuntimeError(f"OpenAI embedding request failed: {e}") from e
    return body["data"][0]["embedding"]


def build_openai_semantic_matrix(
    texts: pd.Series,
    *,
    prefix: str = "llm_sem_",
    model_name: str = "text-embedding-3-small",
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Return one embedding row per text with JSON cache."""
    cache: dict[str, list[float]] = {}
    if cache_path and cache_path.is_file():
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)

    normalized = texts.astype(str).fillna("").map(lambda s: s.strip())
    for text in normalized.unique().tolist():
        if text and text not in cache:
            cache[text] = _post_openai_embedding(text, model_name)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)

    dim = len(next(iter(cache.values()))) if cache else 0
    if dim == 0:
        raise RuntimeError("OpenAI semantic embedding result is empty.")

    cols = [f"{prefix}{i}" for i in range(dim)]
    rows = []
    for text in normalized.tolist():
        if text and text in cache:
            rows.append(np.asarray(cache[text], dtype=np.float64))
        else:
            rows.append(np.zeros(dim, dtype=np.float64))
    return pd.DataFrame(rows, columns=cols, index=texts.index)


def build_local_semantic_matrix(
    texts: pd.Series,
    *,
    prefix: str = "local_sem_",
    model_name: str = "BAAI/bge-small-en-v1.5",
    cache_path: Path | None = None,
    device: str = "cpu",
    batch_size: int = 32,
) -> pd.DataFrame:
    """Return local HuggingFace/SentenceTransformer embeddings with JSON cache."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise ImportError(
            "Local semantic embeddings require sentence-transformers. "
            "Install with: pip install sentence-transformers transformers"
        ) from e

    cache: dict[str, list[float]] = {}
    if cache_path and cache_path.is_file():
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)

    normalized = texts.astype(str).fillna("").map(lambda s: s.strip())
    missing = [text for text in normalized.unique().tolist() if text and text not in cache]

    if missing:
        model = SentenceTransformer(model_name, device=device)
        emb = model.encode(
            missing,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        emb = np.asarray(emb, dtype=np.float64)
        for text, vec in zip(missing, emb):
            cache[text] = vec.tolist()

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)

    dim = len(next(iter(cache.values()))) if cache else 0
    if dim == 0:
        raise RuntimeError("Local semantic embedding result is empty.")

    cols = [f"{prefix}{i}" for i in range(dim)]
    rows = []
    for text in normalized.tolist():
        if text and text in cache:
            rows.append(np.asarray(cache[text], dtype=np.float64))
        else:
            rows.append(np.zeros(dim, dtype=np.float64))
    return pd.DataFrame(rows, columns=cols, index=texts.index)


def build_chemberta_pair_matrix(
    smiles1: pd.Series,
    smiles2: pd.Series,
    x1: pd.Series,
    x2: pd.Series,
    *,
    prefix: str = "chemberta_",
    model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
    pooling: str = "cls",
    device: str = "cpu",
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Encode two SMILES columns with ChemBERTa and build pair features."""
    try:
        from src.feature_chemberta import encode_smiles_batch
    except Exception as e:
        raise ImportError("ChemBERTa pair features require torch and transformers.") from e

    cache: dict[str, list[float]] = {}
    if cache_path and cache_path.is_file():
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)

    s1 = smiles1.astype(str).fillna("").map(lambda s: s.strip())
    s2 = smiles2.astype(str).fillna("").map(lambda s: s.strip())
    unique = []
    for smi in pd.concat([s1, s2]).unique().tolist():
        if smi and smi not in cache:
            unique.append(smi)

    if unique:
        emb = encode_smiles_batch(unique, model_name=model_name, pooling=pooling, device=device, batch_size=32)
        for smi, vec in zip(unique, emb):
            cache[smi] = np.asarray(vec, dtype=np.float64).tolist()

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)

    dim = len(next(iter(cache.values()))) if cache else 0
    if dim == 0:
        raise RuntimeError("ChemBERTa embedding result is empty.")

    x1v = pd.to_numeric(x1, errors="coerce").fillna(0.5).values.astype(np.float64).reshape(-1, 1)
    x2v = pd.to_numeric(x2, errors="coerce").fillna(0.5).values.astype(np.float64).reshape(-1, 1)
    z1, z2 = [], []
    zero = np.zeros(dim, dtype=np.float64)
    for a, b in zip(s1.tolist(), s2.tolist()):
        z1.append(np.asarray(cache.get(a, zero), dtype=np.float64))
        z2.append(np.asarray(cache.get(b, zero), dtype=np.float64))
    e1 = np.vstack(z1)
    e2 = np.vstack(z2)
    pair = np.hstack([e1, e2, np.abs(e1 - e2), e1 * e2, x1v * e1 + x2v * e2])
    cols = [f"{prefix}{i}" for i in range(pair.shape[1])]
    return pd.DataFrame(pair, columns=cols, index=smiles1.index)
