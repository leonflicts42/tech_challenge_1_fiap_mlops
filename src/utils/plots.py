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


# ── Visualizações comparativas de múltiplos modelos ───────────────────────────


def plot_all_roc_curves(
    models_probas: dict[str, np.ndarray],
    y_true: np.ndarray,
    save_path: Path,
    title: str = "Curvas ROC — Comparativo de Modelos",
) -> Path:
    """Plota curvas ROC de todos os modelos sobrepostas em um único gráfico.

    Args:
        models_probas: dict {nome_modelo: y_proba}
        y_true:        labels verdadeiros
        save_path:     caminho de saída da imagem
        title:         título do gráfico

    Returns:
        Path do arquivo salvo.
    """
    from sklearn.metrics import RocCurveDisplay

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name, y_proba in models_probas.items():
        RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, name=model_name)

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Aleatório")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("plot_all_roc_curves | salvo em %s", save_path)
    return save_path


def plot_all_pr_curves(
    models_probas: dict[str, np.ndarray],
    y_true: np.ndarray,
    save_path: Path,
    title: str = "Curvas Precision-Recall — Comparativo de Modelos",
) -> Path:
    """Plota curvas Precision-Recall de todos os modelos sobrepostas.

    Args:
        models_probas: dict {nome_modelo: y_proba}
        y_true:        labels verdadeiros
        save_path:     caminho de saída da imagem
        title:         título do gráfico

    Returns:
        Path do arquivo salvo.
    """
    from sklearn.metrics import PrecisionRecallDisplay

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name, y_proba in models_probas.items():
        PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax, name=model_name)

    baseline = y_true.mean()
    ax.axhline(y=baseline, color="k", linestyle="--", lw=1, alpha=0.5, label=f"Baseline ({baseline:.2f})")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("plot_all_pr_curves | salvo em %s", save_path)
    return save_path


def plot_confusion_matrix_grid(
    models_preds: dict[str, np.ndarray],
    y_true: np.ndarray,
    save_path: Path,
    title: str = "Matrizes de Confusão",
) -> Path:
    """Plota grade de matrizes de confusão, uma por modelo.

    Args:
        models_preds: dict {nome_modelo: y_pred (0/1)}
        y_true:       labels verdadeiros
        save_path:    caminho de saída da imagem
        title:        título do gráfico

    Returns:
        Path do arquivo salvo.
    """
    from sklearn.metrics import ConfusionMatrixDisplay

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(models_preds)
    ncols = min(n, 2)
    nrows = (n + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax, (model_name, y_pred) in zip(axes, models_preds.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, ax=ax, colorbar=False,
            display_labels=["Não Churn", "Churn"],
        )
        ax.set_title(model_name, fontweight="bold")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("plot_confusion_matrix_grid | salvo em %s", save_path)
    return save_path


def plot_classification_report_grid(
    models_preds: dict[str, np.ndarray],
    y_true: np.ndarray,
    save_path: Path,
    title: str = "Classification Report — Comparativo",
) -> Path:
    """Plota heatmap do classification report para cada modelo.

    Args:
        models_preds: dict {nome_modelo: y_pred (0/1)}
        y_true:       labels verdadeiros
        save_path:    caminho de saída da imagem
        title:        título do gráfico

    Returns:
        Path do arquivo salvo.
    """
    import pandas as _pd
    from sklearn.metrics import classification_report

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(models_preds)
    ncols = min(n, 2)
    nrows = (n + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3 * nrows))
    axes = np.array(axes).flatten()

    metrics_order = ["precision", "recall", "f1-score"]
    labels_order = ["0", "1", "macro avg", "weighted avg"]

    for ax, (model_name, y_pred) in zip(axes, models_preds.items()):
        report = classification_report(
            y_true, y_pred,
            target_names=["Não Churn (0)", "Churn (1)"],
            output_dict=True,
        )
        rows = {}
        for k, v in report.items():
            if isinstance(v, dict):
                rows[k] = {m: round(v.get(m, 0), 3) for m in metrics_order}

        df_report = _pd.DataFrame(rows).T

        im = ax.imshow(df_report.values.astype(float), vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(metrics_order)))
        ax.set_xticklabels(metrics_order, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(df_report)))
        ax.set_yticklabels(df_report.index, fontsize=9)

        for i in range(len(df_report)):
            for j in range(len(metrics_order)):
                val = df_report.values[i, j]
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9, color="black")

        ax.set_title(model_name, fontweight="bold", fontsize=11)
        fig.colorbar(im, ax=ax, shrink=0.8)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("plot_classification_report_grid | salvo em %s", save_path)
    return save_path


def plot_f1_threshold_curves(
    models_probas: dict[str, np.ndarray],
    y_true: np.ndarray,
    save_path: Path,
    threshold_range: tuple[float, float, float] = (0.05, 0.95, 0.01),
    title: str = "F1-Score por Threshold — Comparativo de Modelos",
) -> Path:
    """Plota F1-score em função do threshold para cada modelo.

    Args:
        models_probas:   dict {nome_modelo: y_proba}
        y_true:          labels verdadeiros
        save_path:       caminho de saída da imagem
        threshold_range: (min, max, step) para varredura de threshold
        title:           título do gráfico

    Returns:
        Path do arquivo salvo.
    """
    from sklearn.metrics import f1_score

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    thresholds = np.arange(*threshold_range)

    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, y_proba in models_probas.items():
        f1_scores = [
            f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
            for t in thresholds
        ]
        ax.plot(thresholds, f1_scores, label=model_name, lw=2)

    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1-Score")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("plot_f1_threshold_curves | salvo em %s", save_path)
    return save_path
