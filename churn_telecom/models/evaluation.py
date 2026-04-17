"""Model evaluation: metrics, comparison table and cost analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


# ── Estrutura de resultado ───────────────────────────────────────────────────


@dataclass
class ModelMetrics:
    """Conjunto padronizado de métricas para comparação."""

    name: str
    roc_auc: float
    pr_auc: float
    f1: float
    precision: float
    recall: float
    specificity: float  # recall da classe negativa = TN / (TN + FP)
    tn: int
    fp: int
    fn: int
    tp: int

    # Métricas de negócio (preenchidas por CostAnalyzer)
    cost_total: float = 0.0
    churn_avoided: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "ROC-AUC": round(self.roc_auc, 4),
            "PR-AUC": round(self.pr_auc, 4),
            "F1": round(self.f1, 4),
            "Precision": round(self.precision, 4),
            "Recall": round(self.recall, 4),
            "Specificity": round(self.specificity, 4),
            "TN": self.tn,
            "FP": self.fp,
            "FN": self.fn,
            "TP": self.tp,
            "Cost Total (R$)": round(self.cost_total, 2),
            "Churn Avoided (R$)": round(self.churn_avoided, 2),
        }


# ── Calculadora de métricas ───────────────────────────────────────────────────


class MetricsCalculator:
    """
    Calcula o conjunto completo de métricas para um modelo.

    Responsabilidade única: transformar (y_true, y_proba) → ModelMetrics.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def compute(
        self,
        name: str,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> ModelMetrics:
        y_pred = (y_proba >= self.threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        metrics = ModelMetrics(
            name=name,
            roc_auc=float(roc_auc_score(y_true, y_proba)),
            pr_auc=float(average_precision_score(y_true, y_proba)),
            f1=float(f1_score(y_true, y_pred, zero_division=0)),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            specificity=specificity,
            tn=int(tn),
            fp=int(fp),
            fn=int(fn),
            tp=int(tp),
        )
        logger.info(
            "metrics | %s | roc_auc=%.4f | pr_auc=%.4f | f1=%.4f | recall=%.4f",
            name,
            metrics.roc_auc,
            metrics.pr_auc,
            metrics.f1,
            metrics.recall,
        )
        return metrics


# ── Análise de custo de negócio ──────────────────────────────────────────────


@dataclass
class CostConfig:
    """
    Parâmetros financeiros para o cálculo de custo de churn.

    Referência de literatura:
        - Custo de aquisição de novo cliente ≈ 5-7x custo de retenção
        - Receita média mensal por cliente de telecomunicações: R$ 80–150

    Attributes
    ----------
    clv_per_customer : valor médio do ciclo de vida do cliente retido (R$)
    retention_cost   : custo da ação de retenção por cliente abordado (R$)
    fp_cost          : custo de abordar um cliente que NÃO ia sair (R$)
    fn_cost          : custo de perder um cliente que ia sair (R$)
                       ≈ clv_per_customer (receita perdida)
    """

    clv_per_customer: float = 1_200.0  # R$ — receita anual estimada
    retention_cost: float = 80.0  # R$ — custo de ação (ligação, desconto)
    fp_cost: float = 80.0  # = retention_cost (custo de abordar em vão)
    fn_cost: float = 1_200.0  # = clv perdido


class CostAnalyzer:
    """
    Calcula o impacto financeiro de FP e FN para cada modelo.

    Fórmula:
        custo_total = FP × fp_cost + FN × fn_cost
        churn_avoided = TP × (clv_per_customer - retention_cost)
    """

    def __init__(self, config: CostConfig | None = None) -> None:
        self.cfg = config or CostConfig()

    def annotate(self, metrics: ModelMetrics) -> ModelMetrics:
        """Preenche os campos de negócio em `metrics` in-place."""
        metrics.cost_total = (
            metrics.fp * self.cfg.fp_cost + metrics.fn * self.cfg.fn_cost
        )
        metrics.churn_avoided = metrics.tp * (
            self.cfg.clv_per_customer - self.cfg.retention_cost
        )
        logger.info(
            "cost | %s | cost_total=R$%.2f | churn_avoided=R$%.2f",
            metrics.name,
            metrics.cost_total,
            metrics.churn_avoided,
        )
        return metrics

    def tradeoff_summary(self, results: list[ModelMetrics]) -> pd.DataFrame:
        """
        Retorna DataFrame com análise de trade-off FP vs FN.

        Inclui:
        - net_value = churn_avoided - cost_total
        - roi = net_value / cost_total
        """
        rows = []
        for m in results:
            net = m.churn_avoided - m.cost_total
            roi = net / m.cost_total if m.cost_total > 0 else float("inf")
            rows.append(
                {
                    "model": m.name,
                    "FP": m.fp,
                    "FN": m.fn,
                    "cost_FP (R$)": round(m.fp * self.cfg.fp_cost, 2),
                    "cost_FN (R$)": round(m.fn * self.cfg.fn_cost, 2),
                    "cost_total (R$)": round(m.cost_total, 2),
                    "churn_avoided (R$)": round(m.churn_avoided, 2),
                    "net_value (R$)": round(net, 2),
                    "ROI": round(roi, 2),
                }
            )
        df = pd.DataFrame(rows).sort_values("net_value (R$)", ascending=False)
        return df


# ── Tabela comparativa ────────────────────────────────────────────────────────


class ModelComparator:
    """
    Agrega múltiplos ModelMetrics e gera a tabela comparativa.

    Uso:
        comparator = ModelComparator()
        comparator.add(metrics_dummy)
        comparator.add(metrics_lr)
        comparator.add(metrics_mlp)
        df = comparator.summary()
    """

    def __init__(self) -> None:
        self._results: list[ModelMetrics] = []

    def add(self, metrics: ModelMetrics) -> None:
        self._results.append(metrics)
        logger.debug("comparator | modelo adicionado: %s", metrics.name)

    def summary(self) -> pd.DataFrame:
        """DataFrame com todas as métricas, ordenado por ROC-AUC desc."""
        rows = [m.to_dict() for m in self._results]
        df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
        df = df.reset_index(drop=True)
        logger.info("comparator | tabela gerada | modelos=%d", len(df))
        return df

    def best_model_name(self, metric: str = "ROC-AUC") -> str:
        df = self.summary()
        return str(df.iloc[0]["model"])
