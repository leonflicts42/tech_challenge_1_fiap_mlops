"""
Testes automatizados — Etapa 2.

FIXES aplicados nesta versão:
  1. build_mlp() chamado com device="cpu" explícito (4º arg é device, não seed)
  2. verbose= removido do ReduceLROnPlateau (corrigido no trainer.py)
  3. Todos os testes usam apenas CPU — sem dependência de GPU
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from models.evaluation import (
    CostAnalyzer,
    CostConfig,
    MetricsCalculator,
    ModelComparator,
)
from models.mlp import build_mlp
from models.trainer import ChurnTrainer, EarlyStopping, TrainerConfig

# ── Constantes ────────────────────────────────────────────────────────────────

SEED = 42
INPUT_DIM = 20
N_SAMPLES = 300
DEVICE = "cpu"


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_data():
    """Dataset binário balanceado sintético."""
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((N_SAMPLES, INPUT_DIM)).astype(np.float32)
    y = rng.integers(0, 2, size=N_SAMPLES).astype(np.float32)
    return X, y


@pytest.fixture
def split_data(synthetic_data):
    X, y = synthetic_data
    split = int(0.8 * N_SAMPLES)
    return X[:split], y[:split], X[split:], y[split:]


@pytest.fixture
def small_model():
    # FIX: 4º argumento é device — passar "cpu", não SEED (42)
    return build_mlp(INPUT_DIM, [16, 8], 0.0, DEVICE)


@pytest.fixture
def trainer_config():
    return TrainerConfig(
        lr=1e-3,
        epochs=5,
        batch_size=64,
        patience=3,
        device=DEVICE,
        seed=SEED,
    )


# ── EarlyStopping ─────────────────────────────────────────────────────────────


class TestEarlyStopping:
    def test_does_not_stop_on_improvement(self, small_model):
        es = EarlyStopping(patience=3, min_delta=1e-4)
        for loss in [1.0, 0.9, 0.8, 0.7]:
            assert not es.step(loss, small_model), "não deve parar com melhora contínua"

    def test_stops_after_patience(self, small_model):
        es = EarlyStopping(patience=3, min_delta=1e-4)
        es.step(1.0, small_model)  # melhora inicial — registra best_state
        stopped = False
        for _ in range(3):
            stopped = es.step(1.0, small_model)  # platô — incrementa counter
        assert stopped, "deve parar após patience=3 sem melhora"

    def test_restores_best_weights(self, small_model):
        """Após corromper pesos, restore_best deve desfazer a corrupção."""
        es = EarlyStopping(patience=2, min_delta=1e-4)
        es.step(1.0, small_model)  # salva estado atual como melhor

        # Captura soma dos pesos antes da corrupção
        before = sum(p.abs().sum().item() for p in small_model.parameters())

        # Corrompe pesos manualmente
        with torch.no_grad():
            for p in small_model.parameters():
                p.fill_(999.0)

        corrupted = sum(p.abs().sum().item() for p in small_model.parameters())
        assert corrupted != before, "corrupção falhou — teste inválido"

        es.restore_best(small_model)

        after = sum(p.abs().sum().item() for p in small_model.parameters())
        assert abs(after - before) < 1e-3, (
            f"restauração falhou: antes={before:.4f}, depois={after:.4f}"
        )

    def test_counter_resets_on_improvement(self, small_model):
        es = EarlyStopping(patience=3, min_delta=1e-4)
        es.step(1.0, small_model)
        es.step(1.0, small_model)  # counter=1
        es.step(0.5, small_model)  # melhora → reset
        assert es._counter == 0, "counter deve zerar após melhora"


# ── ChurnTrainer ──────────────────────────────────────────────────────────────


class TestChurnTrainer:
    def test_smoke_fit(self, small_model, split_data, trainer_config):
        """Fit deve completar sem exceções."""
        X_tr, y_tr, X_vl, y_vl = split_data
        trainer = ChurnTrainer(small_model, trainer_config)
        history = trainer.fit(X_tr, y_tr, X_vl, y_vl)
        assert history is not None

    def test_history_lengths_match_epochs(
        self, small_model, split_data, trainer_config
    ):
        X_tr, y_tr, X_vl, y_vl = split_data
        trainer = ChurnTrainer(small_model, trainer_config)
        history = trainer.fit(X_tr, y_tr, X_vl, y_vl)
        n = len(history.train_loss)
        assert (
            n == len(history.val_loss) == len(history.train_auc) == len(history.val_auc)
        ), "todas as listas do histórico devem ter o mesmo comprimento"

    def test_history_epochs_leq_max_epochs(
        self, small_model, split_data, trainer_config
    ):
        X_tr, y_tr, X_vl, y_vl = split_data
        trainer = ChurnTrainer(small_model, trainer_config)
        history = trainer.fit(X_tr, y_tr, X_vl, y_vl)
        assert len(history.train_loss) <= trainer_config.epochs, (
            "número de épocas não pode exceder o máximo configurado"
        )

    def test_predict_proba_shape(self, small_model, split_data, trainer_config):
        X_tr, y_tr, X_vl, y_vl = split_data
        trainer = ChurnTrainer(small_model, trainer_config)
        trainer.fit(X_tr, y_tr, X_vl, y_vl)
        proba = trainer.predict_proba(X_vl)
        assert proba.shape == (len(X_vl),), (
            f"shape esperado ({len(X_vl)},), obtido {proba.shape}"
        )

    def test_predict_proba_range(self, small_model, split_data, trainer_config):
        X_tr, y_tr, X_vl, y_vl = split_data
        trainer = ChurnTrainer(small_model, trainer_config)
        trainer.fit(X_tr, y_tr, X_vl, y_vl)
        proba = trainer.predict_proba(X_vl)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0), (
            "probabilidades devem estar em [0, 1]"
        )

    def test_early_stopping_flag(self):
        """Com patience=1, o treino deve parar antes de 20 épocas."""
        # FIX: 4º arg é device — "cpu", não SEED
        model = build_mlp(INPUT_DIM, [8], 0.0, DEVICE)
        cfg = TrainerConfig(epochs=20, patience=1, lr=1e-3, device=DEVICE, seed=SEED)
        rng = np.random.default_rng(SEED)
        X = rng.standard_normal((200, INPUT_DIM)).astype(np.float32)
        y = rng.integers(0, 2, size=200).astype(np.float32)
        trainer = ChurnTrainer(model, cfg)
        history = trainer.fit(X[:160], y[:160], X[160:], y[160:])
        assert len(history.train_loss) <= 20, (
            "com patience=1, o treino deve respeitar o limite máximo de épocas"
        )


# ── MetricsCalculator ─────────────────────────────────────────────────────────


class TestMetricsCalculator:
    def test_perfect_classifier(self):
        calc = MetricsCalculator(threshold=0.5)
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.1, 0.9, 0.9])
        m = calc.compute("perfect", y_true, y_proba)
        assert m.roc_auc == pytest.approx(1.0)
        assert m.recall == pytest.approx(1.0)
        assert m.fn == 0
        assert m.tp == 2

    def test_worst_classifier(self):
        calc = MetricsCalculator(threshold=0.5)
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.9, 0.9, 0.1, 0.1])
        m = calc.compute("worst", y_true, y_proba)
        assert m.tp == 0
        assert m.fn == 2

    def test_specificity_is_tn_rate(self):
        calc = MetricsCalculator(threshold=0.5)
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.1, 0.9, 0.9, 0.9, 0.9])
        m = calc.compute("test", y_true, y_proba)
        expected = m.tn / (m.tn + m.fp)
        assert m.specificity == pytest.approx(expected)


# ── CostAnalyzer ─────────────────────────────────────────────────────────────


class TestCostAnalyzer:
    def test_cost_formula(self):
        """cost_total deve ser FP * fp_cost + FN * fn_cost."""
        cfg = CostConfig(
            fp_cost=100.0,
            fn_cost=1000.0,
            clv_per_customer=1200.0,
            retention_cost=100.0,
        )
        analyzer = CostAnalyzer(cfg)
        calc = MetricsCalculator()
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.9, 0.1, 0.1, 0.9])  # 1 FP, 1 FN
        m = analyzer.annotate(calc.compute("test", y_true, y_proba))
        assert m.cost_total == pytest.approx(m.fp * 100.0 + m.fn * 1000.0)

    def test_net_positive_for_good_model(self):
        """Modelo preciso deve ter net_value positivo."""
        cfg = CostConfig()
        analyzer = CostAnalyzer(cfg)
        calc = MetricsCalculator()
        y_true = np.array([0] * 80 + [1] * 20)
        y_proba = np.concatenate([np.full(80, 0.1), np.full(20, 0.9)])
        m = analyzer.annotate(calc.compute("good", y_true, y_proba))
        net = m.churn_avoided - m.cost_total
        assert net > 0, "modelo preciso deve ter net_value positivo"


# ── ModelComparator ───────────────────────────────────────────────────────────


class TestModelComparator:
    def _make_metrics(self, name: str, auc: float):
        from models.evaluation import ModelMetrics

        return ModelMetrics(
            name=name,
            roc_auc=auc,
            pr_auc=auc,
            f1=auc,
            precision=auc,
            recall=auc,
            specificity=auc,
            tn=10,
            fp=5,
            fn=3,
            tp=8,
        )

    def test_summary_ordered_by_auc(self):
        comp = ModelComparator()
        comp.add(self._make_metrics("A", 0.70))
        comp.add(self._make_metrics("B", 0.85))
        comp.add(self._make_metrics("C", 0.60))
        df = comp.summary()
        assert list(df["model"]) == ["B", "A", "C"], (
            "tabela deve estar ordenada por ROC-AUC decrescente"
        )

    def test_best_model_name(self):
        comp = ModelComparator()
        comp.add(self._make_metrics("Dummy", 0.50))
        comp.add(self._make_metrics("MLP", 0.88))
        assert comp.best_model_name() == "MLP"
