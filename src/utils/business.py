"""
utils/business.py — Funções de valor de negócio para predição de churn.

Centraliza o cálculo de business_value, threshold ótimo e métricas completas.
Importar aqui garante que notebooks e API usem a mesma lógica, sem duplicidade.

Valores padrão vêm de config.py (fonte única de verdade).
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

from config import (
    COST_CLV,
    COST_RETENTION,
    SLO_RECALL_MIN,
    THRESHOLD_MAX,
    THRESHOLD_MIN,
    THRESHOLD_STEP,
)

logger = logging.getLogger(__name__)

# Valores de negócio por célula da matriz de confusão (derivados de config)
VALUE_TP: float = COST_CLV  # churner detectado → retenção bem-sucedida
VALUE_TN: float = COST_RETENTION  # não-churner ignorado → economia operacional
VALUE_FN: float = -COST_CLV  # churner perdido → cancelamento
VALUE_FP: float = -COST_RETENTION  # ação desnecessária → custo desperdiçado


def business_value(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    value_tp: float = VALUE_TP,
    value_tn: float = VALUE_TN,
    value_fn: float = VALUE_FN,
    value_fp: float = VALUE_FP,
) -> float:
    """Calcula o valor de negócio total para um dado threshold.

    Args:
        y_true:    labels verdadeiros (0/1)
        y_proba:   probabilidades previstas pelo modelo
        threshold: limiar de classificação
        value_tp:  valor por verdadeiro positivo (default: COST_CLV)
        value_tn:  valor por verdadeiro negativo (default: COST_RETENTION)
        value_fn:  valor por falso negativo, deve ser negativo (default: -COST_CLV)
        value_fp:  valor por falso positivo, deve ser negativo (default: -COST_RETENTION)

    Returns:
        Valor total: positivo = lucro, negativo = prejuízo.
    """
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp * value_tp + tn * value_tn + fn * value_fn + fp * value_fp


def find_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    slo_recall_min: float = SLO_RECALL_MIN,
    value_tp: float = VALUE_TP,
    value_tn: float = VALUE_TN,
    value_fn: float = VALUE_FN,
    value_fp: float = VALUE_FP,
) -> tuple[float, float]:
    """Varre thresholds e retorna (threshold ótimo, maior valor de negócio).

    Prioriza thresholds que atendem recall >= slo_recall_min.
    Se nenhum threshold atende o SLO, retorna o de maior valor geral (fallback).

    Args:
        y_true:         labels verdadeiros
        y_proba:        probabilidades previstas
        slo_recall_min: recall mínimo aceitável (default: SLO_RECALL_MIN de config)
        value_tp/tn/fn/fp: valores de negócio por célula da matriz de confusão

    Returns:
        Tupla (best_threshold, best_business_value)
    """
    best_t: float = 0.5
    best_val: float = -float("inf")

    thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX, THRESHOLD_STEP)

    for t in thresholds:
        val = business_value(y_true, y_proba, t, value_tp, value_tn, value_fn, value_fp)
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp_count = confusion_matrix(y_true, y_pred).ravel()
        recall = tp_count / max(tp_count + fn, 1)

        if recall >= slo_recall_min and val > best_val:
            best_val = val
            best_t = float(t)

    # Fallback se nenhum threshold atende o SLO
    if best_val == -float("inf"):
        logger.warning(
            "find_best_threshold | nenhum threshold atende SLO recall>=%.2f — "
            "usando fallback (maior valor geral)",
            slo_recall_min,
        )
        for t in thresholds:
            val = business_value(
                y_true, y_proba, t, value_tp, value_tn, value_fn, value_fp
            )
            if val > best_val:
                best_val = val
                best_t = float(t)

    logger.debug("find_best_threshold | best_t=%.2f | best_val=%.0f", best_t, best_val)
    return best_t, best_val


def full_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    value_tp: float = VALUE_TP,
    value_tn: float = VALUE_TN,
    value_fn: float = VALUE_FN,
    value_fp: float = VALUE_FP,
    slo_recall_min: float = SLO_RECALL_MIN,
) -> dict:
    """Calcula métricas completas: estatísticas, negócio e SLO.

    Args:
        y_true:    labels verdadeiros
        y_proba:   probabilidades previstas
        threshold: limiar de classificação aplicado
        value_tp/tn/fn/fp: valores de negócio
        slo_recall_min:    recall mínimo para flag slo_ok

    Returns:
        Dict com roc_auc, pr_auc, recall, precision, f1, tp, tn, fp, fn,
        business_value, threshold, slo_ok.
    """
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * recall / max(prec + recall, 1e-9)
    bv = tp * value_tp + tn * value_tn + fn * value_fn + fp * value_fp

    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "recall": recall,
        "precision": prec,
        "f1": f1,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "business_value": bv,
        "threshold": threshold,
        "slo_ok": recall >= slo_recall_min,
    }
