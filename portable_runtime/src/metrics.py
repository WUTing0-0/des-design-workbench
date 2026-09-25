# --- DES_Paper_Release ---
# 模块: 06_residual_correction
# 作用: 纯组分训练流水线文件 metrics.py
# 原始路径: C:\code2026\experiments\experiments\f4_structure_aware_residual\src\metrics.py
# 说明: 这是论文开源整理副本。脚本内硬编码路径多为 D:\DES_Project，
#       本机镜像一般在 C:\code2026。本整理包不重跑 ORCA/CREST/70 万 SLE。
# -------------------------
"""Regression metrics for F4 residual models."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


def compute_metrics(
    T_true: np.ndarray,
    Tpred: np.ndarray,
    T_final: np.ndarray,
    gap_true: np.ndarray,
    gap_hat: np.ndarray,
) -> dict:
    """Compute baseline, final, and gap metrics on finite samples."""
    m = _finite_mask(T_true, Tpred, T_final, gap_true, gap_hat)
    if not m.any():
        nan = float("nan")
        return {
            "n": 0,
            "baseline_mae": nan,
            "baseline_rmse": nan,
            "baseline_me": nan,
            "final_mae": nan,
            "final_rmse": nan,
            "final_me": nan,
            "final_r2": nan,
            "gap_mae": nan,
            "gap_rmse": nan,
        }

    tt = T_true[m]
    tp = Tpred[m]
    tf = T_final[m]
    gt = gap_true[m]
    gh = gap_hat[m]

    baseline_error = tp - tt
    final_error = tf - tt
    gap_error = gh - gt

    return {
        "n": int(m.sum()),
        "baseline_mae": float(np.mean(np.abs(baseline_error))),
        "baseline_rmse": float(np.sqrt(np.mean(baseline_error**2))),
        "baseline_me": float(np.mean(baseline_error)),
        "final_mae": float(np.mean(np.abs(final_error))),
        "final_rmse": float(np.sqrt(np.mean(final_error**2))),
        "final_me": float(np.mean(final_error)),
        "final_r2": float(r2_score(tt, tf)),
        "gap_mae": float(np.mean(np.abs(gap_error))),
        "gap_rmse": float(np.sqrt(np.mean(gap_error**2))),
    }


def metrics_row(fold: int | str, model: str, metrics: dict) -> dict:
    row = {"fold": fold, "model": model}
    row.update(metrics)
    return row


def print_summary_table(rows: list[dict], title: str = "Summary") -> None:
    """Print formatted summary table to stdout."""
    if not rows:
        print(f"{title}: (no rows)")
        return

    headers = [
        "Model",
        "Baseline_MAE",
        "Final_MAE",
        "Baseline_RMSE",
        "Final_RMSE",
        "Final_ME",
        "Final_R2",
    ]
    print(f"\n{title}")
    print("-" * 90)
    print(
        f"{'Model':<22} {'Baseline_MAE':>14} {'Final_MAE':>12} "
        f"{'Baseline_RMSE':>15} {'Final_RMSE':>12} {'Final_ME':>10} {'Final_R2':>10}"
    )
    print("-" * 90)
    for row in rows:
        model = str(row.get("model", row.get("Model", "")))
        print(
            f"{model:<22} "
            f"{row.get('baseline_mae', row.get('Baseline_MAE', float('nan'))):>14.4f} "
            f"{row.get('final_mae', row.get('Final_MAE', float('nan'))):>12.4f} "
            f"{row.get('baseline_rmse', row.get('Baseline_RMSE', float('nan'))):>15.4f} "
            f"{row.get('final_rmse', row.get('Final_RMSE', float('nan'))):>12.4f} "
            f"{row.get('final_me', row.get('Final_ME', float('nan'))):>10.4f} "
            f"{row.get('final_r2', row.get('Final_R2', float('nan'))):>10.4f}"
        )
    print("-" * 90)
