# --- DES_Paper_Release ---
# 模块: 06_residual_correction
# 作用: 纯组分训练流水线文件 split_utils.py
# 原始路径: C:\code2026\experiments\experiments\f4_structure_aware_residual\src\split_utils.py
# 说明: 这是论文开源整理副本。脚本内硬编码路径多为 D:\DES_Project，
#       本机镜像一般在 C:\code2026。本整理包不重跑 ORCA/CREST/70 万 SLE。
# -------------------------
"""GroupKFold cross-validation utilities for F4."""

from __future__ import annotations

from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.metrics import compute_metrics, metrics_row


def make_group_kfold(groups: np.ndarray, n_splits: int = 5) -> GroupKFold:
    n_unique = len(np.unique(groups))
    if n_unique < n_splits:
        raise ValueError(f"唯一组数 {n_unique} < n_splits {n_splits}")
    return GroupKFold(n_splits=n_splits)


def run_sklearn_cv(
    X: np.ndarray,
    gap_true: np.ndarray,
    T_true: np.ndarray,
    Tpred: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int,
    seed: int,
    model_name: str,
    make_estimator: Callable,
    save_model_fn: Callable | None = None,
    models_dir=None,
    model_prefix: str = "model",
) -> tuple[list[dict], pd.DataFrame]:
    """
    Generic GroupKFold OOF for sklearn-style regressors predicting gap.

    save_model_fn(fold, estimator, preprocessor, path) optional.
    """
    gkf = make_group_kfold(groups, n_splits)
    splits = list(gkf.split(np.arange(len(X)), gap_true, groups))

    n = len(X)
    oof_gap_hat = np.full(n, np.nan)
    oof_T_final = np.full(n, np.nan)
    fold_ids = np.full(n, -1, dtype=int)
    fold_metrics: list[dict] = []

    for fold_id, (tr_idx, va_idx) in enumerate(splits):
        est = make_estimator(seed)

        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr = gap_true[tr_idx]

        mtr = np.isfinite(y_tr) & np.all(np.isfinite(X_tr), axis=1)
        est.fit(X_tr[mtr], y_tr[mtr])
        gap_hat_va = est.predict(X_va)

        oof_gap_hat[va_idx] = gap_hat_va
        oof_T_final[va_idx] = Tpred[va_idx] + gap_hat_va
        fold_ids[va_idx] = fold_id

        m_va = np.isfinite(T_true[va_idx]) & np.isfinite(Tpred[va_idx]) & np.isfinite(oof_T_final[va_idx])
        fm = compute_metrics(
            T_true[va_idx][m_va],
            Tpred[va_idx][m_va],
            oof_T_final[va_idx][m_va],
            gap_true[va_idx][m_va],
            gap_hat_va[m_va],
        )
        fold_metrics.append(metrics_row(fold_id, model_name, fm))

        if save_model_fn is not None and models_dir is not None:
            model_path = models_dir / f"{model_prefix}_fold{fold_id}.pkl"
            save_model_fn(fold_id, est, model_path)

    m_all = (
        np.isfinite(T_true)
        & np.isfinite(Tpred)
        & np.isfinite(oof_T_final)
        & np.isfinite(gap_true)
        & np.isfinite(oof_gap_hat)
    )
    oof_metrics = compute_metrics(
        T_true[m_all],
        Tpred[m_all],
        oof_T_final[m_all],
        gap_true[m_all],
        oof_gap_hat[m_all],
    )
    fold_metrics.append(metrics_row("OOF", model_name, oof_metrics))

    pred_df = pd.DataFrame(
        {
            "fold": fold_ids,
            "gap_true": gap_true,
            "gap_hat": oof_gap_hat,
            "T_final": oof_T_final,
        }
    )
    return fold_metrics, pred_df


def default_save_sklearn_model(estimator, path) -> None:
    joblib.dump(estimator, path)
