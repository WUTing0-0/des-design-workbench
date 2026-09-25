# --- DES_Paper_Release ---
# 模块: 06_residual_correction
# 作用: 纯组分训练流水线文件 cv_runner.py
# 原始路径: C:\code2026\experiments\experiments\f4_structure_aware_residual\src\cv_runner.py
# 说明: 这是论文开源整理副本。脚本内硬编码路径多为 D:\DES_Project，
#       本机镜像一般在 C:\code2026。本整理包不重跑 ORCA/CREST/70 万 SLE。
# -------------------------
"""Shared GroupKFold CV runner with per-fold imputation."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd

from src.data_utils import FoldPreprocessor
from src.metrics import compute_metrics, metrics_row, print_summary_table
from src.plotting import save_all_figures
from src.split_utils import make_group_kfold


def run_residual_cv(
    df: pd.DataFrame,
    numeric_df: pd.DataFrame,
    extra_feature_df: pd.DataFrame | None,
    T_true: np.ndarray,
    Tpred: np.ndarray,
    gap_true: np.ndarray,
    groups: np.ndarray,
    *,
    model_name: str,
    model_prefix: str,
    out_dirs: dict[str, Path],
    n_splits: int,
    seed: int,
    make_estimator: Callable[[int], object],
    meta_cols: dict[str, np.ndarray | pd.Series],
) -> tuple[list[dict], pd.DataFrame]:
    """
    Per-fold: fit imputer on train numeric, concat extra features, fit regressor on gap.
    """
    gkf = make_group_kfold(groups, n_splits)
    splits = list(gkf.split(np.arange(len(df)), gap_true, groups))

    n = len(df)
    numeric_arr = numeric_df.values.astype(np.float64)
    extra_arr = None if extra_feature_df is None else extra_feature_df.values.astype(np.float64)

    oof_gap_hat = np.full(n, np.nan)
    oof_T_final = np.full(n, np.nan)
    fold_ids = np.full(n, -1, dtype=int)
    fold_metrics: list[dict] = []

    for fold_id, (tr_idx, va_idx) in enumerate(splits):
        pre = FoldPreprocessor()
        X_num_tr = pre.fit(numeric_arr[tr_idx]).transform(numeric_arr[tr_idx])
        X_num_va = pre.transform(numeric_arr[va_idx])

        if extra_arr is not None:
            X_tr = np.hstack([X_num_tr, extra_arr[tr_idx]])
            X_va = np.hstack([X_num_va, extra_arr[va_idx]])
        else:
            X_tr, X_va = X_num_tr, X_num_va

        y_tr = gap_true[tr_idx]
        est = make_estimator(seed)
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

        pack = {"estimator": est, "preprocessor": pre, "numeric_columns": list(numeric_df.columns)}
        if extra_feature_df is not None:
            pack["extra_columns"] = list(extra_feature_df.columns)
        joblib.dump(pack, out_dirs["models"] / f"{model_prefix}_fold{fold_id}.pkl")

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

    pred_df = pd.DataFrame({"fold": fold_ids})
    for k, v in meta_cols.items():
        pred_df[k] = v
    pred_df["T_true"] = T_true
    pred_df["Tpred"] = Tpred
    pred_df["gap_true"] = gap_true
    pred_df["gap_hat"] = oof_gap_hat
    pred_df["T_final"] = oof_T_final
    pred_df["baseline_error"] = Tpred - T_true
    pred_df["final_error"] = oof_T_final - T_true

    metrics_path = out_dirs["metrics"] / f"{model_prefix}_cv_metrics.csv"
    pd.DataFrame(fold_metrics).to_csv(metrics_path, index=False, encoding="utf-8-sig")

    pred_path = out_dirs["predictions"] / f"{model_prefix}_predictions.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    save_all_figures(
        model_prefix,
        out_dirs["figures"],
        T_true,
        Tpred,
        oof_T_final,
        gap_true,
        oof_gap_hat,
    )

    oof_row = next(r for r in fold_metrics if r["fold"] == "OOF")
    print_summary_table([oof_row], title=f"{model_name} OOF Summary")

    return fold_metrics, pred_df


def baseline_metrics_row(T_true: np.ndarray, Tpred: np.ndarray, model_name: str = "SLE_Tpred") -> dict:
    gap_zero = np.zeros_like(T_true)
    m = compute_metrics(T_true, Tpred, Tpred, T_true - Tpred, gap_zero)
    row = {
        "Model": model_name,
        "Baseline_MAE": m["baseline_mae"],
        "Final_MAE": m["baseline_mae"],
        "Baseline_RMSE": m["baseline_rmse"],
        "Final_RMSE": m["baseline_rmse"],
        "Final_ME": m["baseline_me"],
        "Final_R2": float("nan"),
    }
    row.update(m)
    return row
