# churn_telecom/plots.py
"""Funções de visualização reutilizáveis para o projeto churn-telecom."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

logger = logging.getLogger(__name__)


def save_confusion_matrix(
    y_true,
    y_pred,
    path: Path,
    title: str = "Confusion Matrix",
) -> Path:
    """Salva matriz de confusão em PNG e retorna o Path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

    logger.info("Confusion matrix salva em: %s", path)
    return path


def save_roc_curve(
    y_true,
    y_proba,
    path: Path,
    model_name: str = "Model",
    title: str = "ROC Curve",
) -> Path:
    """Salva curva ROC em PNG e retorna o Path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, name=model_name)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

    logger.info("ROC curve salva em: %s", path)
    return path


def save_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    path: Path,
    title: str = "Feature Importance",
    color: str = "steelblue",
    top_n: int = 20,
) -> Path:
    """Salva gráfico de feature importance (barras horizontais) em PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    series = pd.Series(importances, index=feature_names).sort_values(ascending=True)
    top = series.tail(top_n)

    fig, ax = plt.subplots(figsize=(7, 6))
    top.plot(kind="barh", ax=ax, color=color)
    ax.set_title(title)
    ax.set_xlabel("Importância")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

    logger.info("Feature importance salva em: %s", path)
    return path


def save_precision_recall_curve(
    precision_arr: np.ndarray,
    recall_arr: np.ndarray,
    thresholds: np.ndarray,
    best_threshold: float,
    path: Path,
    model_name: str = "Model",
) -> Path:
    """Salva curva Precision-Recall com ponto ótimo destacado."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # índice do threshold ótimo
    idx = int(np.searchsorted(thresholds, best_threshold))
    idx = min(idx, len(recall_arr) - 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    PrecisionRecallDisplay(precision=precision_arr, recall=recall_arr).plot(
        ax=ax, name=model_name
    )
    ax.scatter(
        [recall_arr[idx]],
        [precision_arr[idx]],
        color="coral",
        zorder=5,
        s=80,
        label=f"thr={best_threshold:.2f}",
    )
    ax.axvline(x=recall_arr[idx], color="coral", linestyle="--", alpha=0.6)
    ax.set_title("Precision-Recall Curve — Threshold Tuning")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

    logger.info("PR curve salva em: %s", path)
    return path


def save_threshold_f1_recall(
    thresholds: np.ndarray,
    f1_arr: np.ndarray,
    recall_arr: np.ndarray,
    best_threshold: float,
    path: Path,
) -> Path:
    """Salva gráfico de F1 e Recall por threshold."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, f1_arr, label="F1", color="steelblue")
    ax.plot(thresholds, recall_arr[:-1], label="Recall", color="coral")
    ax.axvline(
        x=best_threshold,
        linestyle="--",
        color="dimgray",
        label=f"thr={best_threshold:.2f}",
    )
    ax.set_xlabel("Threshold")
    ax.set_title("F1 e Recall por Threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

    logger.info("Threshold plot salvo em: %s", path)
    return path


"""
plots_patch.py — colar ao FINAL de churn_telecom/plots.py

Adiciona a função save_training_curves ao módulo de plots existente.
NÃO substitua o arquivo inteiro; apenas cole este bloco ao final.
"""


def save_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    output_path: Path | str,
    train_aucs: list[float] | None = None,
    val_aucs: list[float] | None = None,
    best_epoch: int | None = None,
) -> None:
    """
    Gera e salva o gráfico de curvas de treinamento (loss e opcionalmente AUC).

    Parameters
    ----------
    train_losses : lista de perdas por época (treino)
    val_losses   : lista de perdas por época (validação)
    output_path  : caminho de saída da imagem (ex.: 'models/training_curves.png')
    train_aucs   : AUC-ROC por época no treino (opcional)
    val_aucs     : AUC-ROC por época na validação (opcional)
    best_epoch   : época do melhor checkpoint (marcada com linha vertical)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    has_auc = train_aucs is not None and val_aucs is not None
    n_plots = 2 if has_auc else 1
    epochs = np.arange(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    # ── Loss ────────────────────────────────────────────────────────────────
    ax_loss = axes[0]
    ax_loss.plot(epochs, train_losses, label="train loss", linewidth=1.8)
    ax_loss.plot(epochs, val_losses, label="val loss", linewidth=1.8, linestyle="--")
    if best_epoch is not None:
        ax_loss.axvline(
            best_epoch,
            color="red",
            linestyle=":",
            linewidth=1.2,
            label=f"best epoch ({best_epoch})",
        )
    ax_loss.set_xlabel("Época")
    ax_loss.set_ylabel("BCE Loss")
    ax_loss.set_title("Curva de Loss")
    ax_loss.legend(fontsize=9)
    ax_loss.grid(alpha=0.3)

    # ── AUC ─────────────────────────────────────────────────────────────────
    if has_auc:
        ax_auc = axes[1]
        ax_auc.plot(epochs, train_aucs, label="train AUC", linewidth=1.8)
        ax_auc.plot(epochs, val_aucs, label="val AUC", linewidth=1.8, linestyle="--")
        if best_epoch is not None:
            ax_auc.axvline(
                best_epoch,
                color="red",
                linestyle=":",
                linewidth=1.2,
                label=f"best epoch ({best_epoch})",
            )
        ax_auc.set_xlabel("Época")
        ax_auc.set_ylabel("ROC-AUC")
        ax_auc.set_title("Curva de AUC-ROC")
        ax_auc.set_ylim(0.4, 1.02)
        ax_auc.legend(fontsize=9)
        ax_auc.grid(alpha=0.3)

    fig.suptitle("Histórico de Treinamento — ChurnMLP", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("save_training_curves | salvo em %s", output_path)
