# --- DES_Paper_Release ---
# 模块: 06_residual_correction
# 作用: 纯组分训练流水线文件 feature_deepseek_labels.py
# 原始路径: C:\code2026\experiments\experiments\f4_structure_aware_residual\src\feature_deepseek_labels.py
# 说明: 这是论文开源整理副本。脚本内硬编码路径多为 D:\DES_Project，
#       本机镜像一般在 C:\code2026。本整理包不重跑 ORCA/CREST/70 万 SLE。
# -------------------------
"""Convert DeepSeek expert JSONL labels to fixed model features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _read_taxonomy(taxonomy_path: Path) -> dict:
    with open(taxonomy_path, encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path or not path.is_file():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            des_id = str(obj.get("DES_ID", "")).strip()
            if des_id:
                out[des_id] = obj
    return out


def build_deepseek_label_features(
    des_ids: pd.Series,
    *,
    labels_path: Path,
    taxonomy_path: Path,
) -> pd.DataFrame:
    taxonomy = _read_taxonomy(taxonomy_path)
    labels = _read_jsonl(labels_path)

    columns: list[tuple[str, str]] = []
    for section in ("des_type", "functional_groups", "interaction_mechanisms", "risk_flags"):
        for tag in taxonomy.get(section, []):
            columns.append((section, tag))
    extra_cols = [
        "deepseek_confidence",
        "deepseek_bias_likely_too_low",
        "deepseek_bias_likely_too_high",
        "deepseek_bias_uncertain",
        "deepseek_action_accept",
        "deepseek_action_caution",
        "deepseek_action_review",
        "deepseek_has_novel_suggestion",
        "deepseek_label_available",
    ]

    rows = []
    for raw_id in des_ids.astype(str).tolist():
        obj = labels.get(raw_id, {})
        row = {f"deepseek_{section}__{tag}": 0.0 for section, tag in columns}
        if obj:
            for section, tag in columns:
                vals = obj.get(section, [])
                if isinstance(vals, str):
                    vals = [vals]
                row[f"deepseek_{section}__{tag}"] = float(tag in vals)
            try:
                row["deepseek_confidence"] = float(obj.get("confidence", 0.0))
            except Exception:
                row["deepseek_confidence"] = 0.0
            bias = str(obj.get("likely_bias_direction", "uncertain"))
            action = str(obj.get("recommended_action", "review"))
            row["deepseek_bias_likely_too_low"] = float(bias == "likely_too_low")
            row["deepseek_bias_likely_too_high"] = float(bias == "likely_too_high")
            row["deepseek_bias_uncertain"] = float(bias == "uncertain")
            row["deepseek_action_accept"] = float(action == "accept")
            row["deepseek_action_caution"] = float(action == "caution")
            row["deepseek_action_review"] = float(action == "review")
            row["deepseek_has_novel_suggestion"] = float(bool(obj.get("novel_tag_suggestions", [])))
            row["deepseek_label_available"] = 1.0
        else:
            for col in extra_cols:
                row[col] = 0.0
        rows.append(row)

    df = pd.DataFrame(rows, index=des_ids.index)
    for col in extra_cols:
        if col not in df.columns:
            df[col] = 0.0
    return df.fillna(0.0)
