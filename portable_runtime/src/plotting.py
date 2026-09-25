# --- DES_Paper_Release ---
# 模块: 06_residual_correction
# 作用: 纯组分训练流水线文件 plotting.py
# 原始路径: C:\code2026\experiments\experiments\f4_structure_aware_residual\src\plotting.py
# 说明: 这是论文开源整理副本。脚本内硬编码路径多为 D:\DES_Project，
#       本机镜像一般在 C:\code2026。本整理包不重跑 ORCA/CREST/70 万 SLE。
# -------------------------
"""Plotting utilities for F4 model evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _finite_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def plot_parity(
    T_true: np.ndarray,
    T_pred: np.ndarray,
    out_path: Path | str,
    *,
    title: str,
    ylabel: str,
) -> None:
    x, y = _finite_xy(T_true, T_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, alpha=0.35, s=12, edgecolors="none")
    if len(x) > 0:
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("T_true (K)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_gap_scatter(
    gap_true: np.ndarray,
    gap_hat: np.ndarray,
    out_path: Path | str,
    *,
    title: str = "Gap prediction",
) -> None:
    x, y = _finite_xy(gap_true, gap_hat)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, alpha=0.35, s=12, edgecolors="none")
    if len(x) > 0:
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
    ax.set_xlabel("gap_true (K)")
    ax.set_ylabel("gap_hat (K)")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_error_distribution(
    baseline_error: np.ndarray,
    final_error: np.ndarray,
    out_path: Path | str,
    *,
    title: str = "Error distribution",
) -> None:
    be = baseline_error[np.isfinite(baseline_error)]
    fe = final_error[np.isfinite(final_error)]
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = 50
    ax.hist(be, bins=bins, alpha=0.55, label="baseline_error (Tpred - T_true)", density=True)
    ax.hist(fe, bins=bins, alpha=0.55, label="final_error (T_final - T_true)", density=True)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Error (K)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_all_figures(
    prefix: str,
    figures_dir: Path,
    T_true: np.ndarray,
    Tpred: np.ndarray,
    T_final: np.ndarray,
    gap_true: np.ndarray,
    gap_hat: np.ndarray,
) -> None:
    figures_dir = Path(figures_dir)
    baseline_error = Tpred - T_true
    final_error = T_final - T_true

    plot_parity(
        T_true,
        Tpred,
        figures_dir / f"{prefix}_parity_baseline.png",
        title=f"{prefix}: baseline parity",
        ylabel="Tpred (K)",
    )
    plot_parity(
        T_true,
        T_final,
        figures_dir / f"{prefix}_parity_final.png",
        title=f"{prefix}: final parity",
        ylabel="T_final (K)",
    )
    plot_gap_scatter(
        gap_true,
        gap_hat,
        figures_dir / f"{prefix}_gap_true_vs_pred.png",
        title=f"{prefix}: gap prediction",
    )
    plot_error_distribution(
        baseline_error,
        final_error,
        figures_dir / f"{prefix}_error_distribution.png",
        title=f"{prefix}: error distribution",
    )
