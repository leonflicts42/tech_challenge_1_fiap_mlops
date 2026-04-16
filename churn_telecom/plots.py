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


# ─────────────────────────────────────────────────────────────────────────────
# PATCH para churn_telecom/plots.py
#
# Adicione esta função ao final do arquivo. Mantém o padrão das demais:
# Path absoluto, dpi=120, plt.close(fig), logger.info no fim.
# ─────────────────────────────────────────────────────────────────────────────


def save_training_curves(
    history: dict[str, list[float]],
    path: Path,
    monitor_metric: str = "val_pr_auc",
    title: str = "Training Curves",
) -> Path:
    """Salva curvas de treino da MLP em PNG (loss + métrica monitorada).

    Parâmetros
    ----------
    history         : dict com chaves "train_loss", "val_loss" e a métrica monitorada.
                      Cada valor é a lista de medições por época.
    path            : caminho de saída do PNG.
    monitor_metric  : nome da métrica usada em early stopping (eixo direito).
    title           : título global da figura.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax_loss = plt.subplots(figsize=(8, 4))

    # Eixo esquerdo: losses
    ax_loss.plot(epochs, history["train_loss"], label="train_loss", color="steelblue")
    ax_loss.plot(epochs, history["val_loss"], label="val_loss", color="coral")
    ax_loss.set_xlabel("Época")
    ax_loss.set_ylabel("Loss (BCE)")
    ax_loss.grid(alpha=0.3)

    # Eixo direito: métrica monitorada
    if monitor_metric in history and len(history[monitor_metric]) > 0:
        ax_metric = ax_loss.twinx()
        ax_metric.plot(
            epochs,
            history[monitor_metric],
            label=monitor_metric,
            color="dimgray",
            linestyle="--",
        )
        ax_metric.set_ylabel(monitor_metric)

        lines_l, labels_l = ax_loss.get_legend_handles_labels()
        lines_r, labels_r = ax_metric.get_legend_handles_labels()
        ax_loss.legend(lines_l + lines_r, labels_l + labels_r, loc="best")
    else:
        ax_loss.legend(loc="best")

    ax_loss.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

    logger.info("Training curves salvas em: %s", path)
    return path