"""Smoke tests da arquitetura ChurnMLP.

Cobre:
- Construção e contagem de parâmetros.
- Forward pass com shape correto.
- Backward pass produz gradientes.
- Validação de argumentos inválidos.
- Reprodutibilidade via seed (factory).
- Modos train() / eval() comportam-se de forma diferente (Dropout/BatchNorm).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from churn_telecom.models.mlp import ChurnMLP, build_mlp


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def in_features() -> int:
    return 30


@pytest.fixture
def batch_size() -> int:
    return 16


@pytest.fixture
def model(in_features: int) -> ChurnMLP:
    return ChurnMLP(in_features=in_features, hidden_dims=[64, 32], dropout=0.3)


@pytest.fixture
def x(batch_size: int, in_features: int) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(batch_size, in_features)


# ── Testes de construção ──────────────────────────────────────────────────────


def test_mlp_constroi_com_atributos_corretos(model: ChurnMLP, in_features: int) -> None:
    assert model.in_features == in_features
    assert model.hidden_dims == [64, 32]
    assert model.dropout == 0.3
    assert isinstance(model.output, nn.Linear)
    assert model.output.out_features == 1


def test_mlp_conta_parametros_treinaveis(model: ChurnMLP) -> None:
    # Linear(30→64)=1984+64, BN(64)=128, Linear(64→32)=2080+32, BN(32)=64,
    # Linear(32→1)=32+1
    assert model.count_parameters() > 0
    total_manual = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert model.count_parameters() == total_manual


# ── Testes de forward ─────────────────────────────────────────────────────────


def test_forward_retorna_shape_correto(
    model: ChurnMLP, x: torch.Tensor, batch_size: int
) -> None:
    model.eval()
    logits = model(x)
    assert logits.shape == (batch_size, 1)
    assert logits.dtype == torch.float32


def test_forward_retorna_logits_nao_probabilidades(
    model: ChurnMLP, x: torch.Tensor
) -> None:
    """Logits podem assumir qualquer valor real (sem Sigmoid no forward)."""
    model.eval()
    logits = model(x)
    # Logits não são limitados a [0, 1]; ao menos um valor fora desse range
    # ao longo do batch é estatisticamente esperado com inputs ~N(0,1).
    assert (logits < 0).any() or (logits > 1).any()


# ── Testes de backward ───────────────────────────────────────────────────────


def test_backward_produz_gradientes(model: ChurnMLP, x: torch.Tensor) -> None:
    model.train()
    y = torch.randint(0, 2, (x.shape[0], 1)).float()
    loss_fn = nn.BCEWithLogitsLoss()

    logits = model(x)
    loss = loss_fn(logits, y)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Sem gradiente em {name}"
        assert torch.isfinite(param.grad).all(), f"Gradiente não-finito em {name}"


# ── Testes de validação de argumentos ────────────────────────────────────────


@pytest.mark.parametrize(
    ("in_feat", "hidden", "drop"),
    [
        (0, [16], 0.1),  # in_features inválido
        (10, [], 0.1),  # hidden_dims vazio
        (10, [16, 0], 0.1),  # dim inválida
        (10, [16], 1.0),  # dropout inválido
        (10, [16], -0.1),  # dropout negativo
    ],
)
def test_construtor_rejeita_argumentos_invalidos(
    in_feat: int, hidden: list[int], drop: float
) -> None:
    with pytest.raises(ValueError):
        ChurnMLP(in_features=in_feat, hidden_dims=hidden, dropout=drop)


# ── Reprodutibilidade ────────────────────────────────────────────────────────


def test_factory_e_reprodutivel_com_mesma_seed(in_features: int) -> None:
    m1 = build_mlp(in_features=in_features, seed=123, device="cpu")
    m2 = build_mlp(in_features=in_features, seed=123, device="cpu")
    for p1, p2 in zip(m1.parameters(), m2.parameters(), strict=True):
        assert torch.equal(p1, p2)


# ── Train vs Eval ────────────────────────────────────────────────────────────


def test_train_eval_modes_diferem(model: ChurnMLP, x: torch.Tensor) -> None:
    """Em train() o Dropout introduz estocasticidade; em eval() não."""
    model.eval()
    out_eval_1 = model(x)
    out_eval_2 = model(x)
    assert torch.equal(out_eval_1, out_eval_2)

    model.train()
    torch.manual_seed(1)
    out_train_1 = model(x)
    torch.manual_seed(2)
    out_train_2 = model(x)
    assert not torch.equal(out_train_1, out_train_2)
