"""Testes para MetricsCalculator, CostAnalyzer e ModelComparator."""

from __future__ import annotations

import numpy as np
import pytest

from models.evaluation import (
    CostAnalyzer,
    CostConfig,
    MetricsCalculator,
    ModelComparator,
    ModelMetrics,
)


@pytest.fixture
def y_true() -> np.ndarray:
    return np.array([0, 0, 1, 1, 0, 1, 0, 1])


@pytest.fixture
def y_proba() -> np.ndarray:
    return np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.6])


@pytest.fixture
def metrics(y_true: np.ndarray, y_proba: np.ndarray) -> ModelMetrics:
    return MetricsCalculator(threshold=0.5).compute("test_model", y_true, y_proba)


class TestMetricsCalculator:
    def test_compute_retorna_model_metrics(self, y_true: np.ndarray, y_proba: np.ndarray) -> None:
        calc = MetricsCalculator(threshold=0.5)
        result = calc.compute("model_a", y_true, y_proba)
        assert isinstance(result, ModelMetrics)
        assert result.name == "model_a"

    def test_roc_auc_entre_0_e_1(self, metrics: ModelMetrics) -> None:
        assert 0.0 <= metrics.roc_auc <= 1.0

    def test_pr_auc_entre_0_e_1(self, metrics: ModelMetrics) -> None:
        assert 0.0 <= metrics.pr_auc <= 1.0

    def test_f1_entre_0_e_1(self, metrics: ModelMetrics) -> None:
        assert 0.0 <= metrics.f1 <= 1.0

    def test_matriz_confusao_soma_correta(self, metrics: ModelMetrics, y_true: np.ndarray) -> None:
        assert metrics.tn + metrics.fp + metrics.fn + metrics.tp == len(y_true)

    def test_specificity_calculada(self, metrics: ModelMetrics) -> None:
        assert 0.0 <= metrics.specificity <= 1.0

    def test_to_dict_tem_campos_obrigatorios(self, metrics: ModelMetrics) -> None:
        d = metrics.to_dict()
        for key in ["model", "ROC-AUC", "PR-AUC", "F1", "Recall", "TN", "FP", "FN", "TP"]:
            assert key in d

    def test_threshold_alto_aumenta_especificidade(self, y_true: np.ndarray, y_proba: np.ndarray) -> None:
        calc_alto = MetricsCalculator(threshold=0.95)
        m = calc_alto.compute("high_threshold", y_true, y_proba)
        assert m.fp == 0

    def test_specificity_zero_quando_todos_fp(self) -> None:
        """TN=0, FP=N → specificity=0. Usa y_true com ambas as classes para evitar erro ROC-AUC."""
        y_true = np.array([0, 0, 0, 1])
        y_proba = np.array([0.9, 0.9, 0.9, 0.95])
        calc = MetricsCalculator(threshold=0.5)
        m = calc.compute("all_fp", y_true, y_proba)
        assert m.specificity == 0.0


class TestCostAnalyzer:
    def test_annotate_preenche_cost_total(self, metrics: ModelMetrics) -> None:
        analyzer = CostAnalyzer()
        result = analyzer.annotate(metrics)
        assert result.cost_total >= 0.0

    def test_annotate_preenche_churn_avoided(self, metrics: ModelMetrics) -> None:
        analyzer = CostAnalyzer()
        result = analyzer.annotate(metrics)
        assert result.churn_avoided >= 0.0

    def test_annotate_retorna_mesmo_objeto(self, metrics: ModelMetrics) -> None:
        analyzer = CostAnalyzer()
        result = analyzer.annotate(metrics)
        assert result is metrics

    def test_cost_total_formula(self) -> None:
        cfg = CostConfig(fp_cost=10.0, fn_cost=100.0)
        m = ModelMetrics(
            name="test", roc_auc=0.9, pr_auc=0.8, f1=0.7,
            precision=0.8, recall=0.7, specificity=0.9,
            tn=50, fp=5, fn=2, tp=10,
        )
        CostAnalyzer(cfg).annotate(m)
        assert m.cost_total == pytest.approx(5 * 10.0 + 2 * 100.0)

    def test_churn_avoided_formula(self) -> None:
        cfg = CostConfig(clv_per_customer=1000.0, retention_cost=50.0)
        m = ModelMetrics(
            name="test", roc_auc=0.9, pr_auc=0.8, f1=0.7,
            precision=0.8, recall=0.7, specificity=0.9,
            tn=50, fp=5, fn=2, tp=10,
        )
        CostAnalyzer(cfg).annotate(m)
        assert m.churn_avoided == pytest.approx(10 * (1000.0 - 50.0))

    def test_config_default(self) -> None:
        analyzer = CostAnalyzer()
        assert analyzer.cfg is not None

    def test_tradeoff_summary_retorna_dataframe(self) -> None:
        analyzer = CostAnalyzer()
        m1 = ModelMetrics("A", 0.9, 0.8, 0.7, 0.8, 0.7, 0.9, 50, 5, 2, 10)
        m2 = ModelMetrics("B", 0.85, 0.75, 0.65, 0.75, 0.65, 0.85, 45, 8, 4, 8)
        for m in [m1, m2]:
            analyzer.annotate(m)
        df = analyzer.tradeoff_summary([m1, m2])
        assert len(df) == 2
        assert "net_value (R$)" in df.columns
        assert "ROI" in df.columns

    def test_tradeoff_summary_ordena_por_net_value(self) -> None:
        cfg = CostConfig(fp_cost=10.0, fn_cost=100.0, clv_per_customer=1000.0, retention_cost=50.0)
        analyzer = CostAnalyzer(cfg)
        m_bom = ModelMetrics("Bom", 0.95, 0.9, 0.85, 0.9, 0.85, 0.95, 50, 2, 1, 15)
        m_ruim = ModelMetrics("Ruim", 0.7, 0.6, 0.5, 0.6, 0.5, 0.7, 30, 20, 10, 5)
        for m in [m_bom, m_ruim]:
            analyzer.annotate(m)
        df = analyzer.tradeoff_summary([m_bom, m_ruim])
        assert df.iloc[0]["model"] == "Bom"

    def test_roi_infinito_quando_cost_zero(self) -> None:
        cfg = CostConfig(fp_cost=0.0, fn_cost=0.0)
        analyzer = CostAnalyzer(cfg)
        m = ModelMetrics("zero_cost", 0.9, 0.8, 0.7, 0.8, 0.7, 0.9, 50, 0, 0, 10)
        analyzer.annotate(m)
        df = analyzer.tradeoff_summary([m])
        assert df.iloc[0]["ROI"] == float("inf")


class TestModelComparator:
    def test_add_e_summary(self, metrics: ModelMetrics) -> None:
        comp = ModelComparator()
        comp.add(metrics)
        df = comp.summary()
        assert len(df) == 1
        assert df.iloc[0]["model"] == metrics.name

    def test_summary_ordena_por_roc_auc(self) -> None:
        comp = ModelComparator()
        m_baixo = ModelMetrics("baixo", 0.6, 0.5, 0.4, 0.5, 0.4, 0.6, 50, 10, 8, 5)
        m_alto = ModelMetrics("alto", 0.9, 0.8, 0.7, 0.8, 0.7, 0.9, 50, 3, 2, 10)
        comp.add(m_baixo)
        comp.add(m_alto)
        df = comp.summary()
        assert df.iloc[0]["model"] == "alto"

    def test_best_model_name(self) -> None:
        comp = ModelComparator()
        comp.add(ModelMetrics("A", 0.7, 0.6, 0.5, 0.6, 0.5, 0.7, 40, 8, 6, 6))
        comp.add(ModelMetrics("B", 0.9, 0.8, 0.7, 0.8, 0.7, 0.9, 50, 3, 2, 10))
        assert comp.best_model_name() == "B"
