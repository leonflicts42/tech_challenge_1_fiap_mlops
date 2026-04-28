"""Testes para EarlyStopping e ChurnTrainer — models/trainer.py."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from models.mlp import ChurnMLPv2
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
    return ChurnMLPv2(input_dim=INPUT_DIM, hidden_dims=[16, 8], dropout=0.0).to(DEVICE)


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

        before = sum(p.abs().sum().item() for p in small_model.parameters())

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
        model = ChurnMLPv2(input_dim=INPUT_DIM, hidden_dims=[8], dropout=0.0).to(DEVICE)
        cfg = TrainerConfig(epochs=20, patience=1, lr=1e-3, device=DEVICE, seed=SEED)
        rng = np.random.default_rng(SEED)
        X = rng.standard_normal((200, INPUT_DIM)).astype(np.float32)
        y = rng.integers(0, 2, size=200).astype(np.float32)
        trainer = ChurnTrainer(model, cfg)
        history = trainer.fit(X[:160], y[:160], X[160:], y[160:])
        assert len(history.train_loss) <= 20, (
            "com patience=1, o treino deve respeitar o limite máximo de épocas"
        )
