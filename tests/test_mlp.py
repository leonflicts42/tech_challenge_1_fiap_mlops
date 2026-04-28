"""Testes para ChurnMLPv2 e ChurnMLPInference — models/mlp.py."""

from __future__ import annotations

import pytest
import torch

from models.mlp import ChurnMLPInference, ChurnMLPv2


@pytest.fixture
def input_dim() -> int:
    return 30


@pytest.fixture
def batch_size() -> int:
    return 8


@pytest.fixture
def x(batch_size: int, input_dim: int) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch_size, input_dim)


class TestChurnMLPv2:
    def test_constroi_com_defaults(self, input_dim: int) -> None:
        model = ChurnMLPv2(input_dim=input_dim, hidden_dims=[64, 32])
        assert model.input_dim == input_dim
        assert model.hidden_dims == [64, 32]

    def test_forward_shape_correto(
        self, input_dim: int, x: torch.Tensor, batch_size: int
    ) -> None:
        model = ChurnMLPv2(input_dim=input_dim, hidden_dims=[64, 32])
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch_size, 1)
        assert out.dtype == torch.float32

    def test_forward_retorna_logits(self, input_dim: int, x: torch.Tensor) -> None:
        model = ChurnMLPv2(input_dim=input_dim, hidden_dims=[64, 32])
        model.eval()
        with torch.no_grad():
            logits = model(x)
        assert (logits < 0).any() or (logits > 1).any()

    def test_skip_connection_com_projecao(self, x: torch.Tensor) -> None:
        """use_skip=True com input_dim != última hidden → usa skip_proj."""
        model = ChurnMLPv2(input_dim=30, hidden_dims=[64, 32], use_skip=True)
        assert model.skip_proj is not None
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (x.shape[0], 1)

    def test_skip_connection_sem_projecao(self) -> None:
        """use_skip=True com input_dim == última hidden → x + skip direto."""
        model = ChurnMLPv2(input_dim=32, hidden_dims=[64, 32], use_skip=True)
        assert model.skip_proj is None
        x = torch.randn(4, 32)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 1)

    def test_sem_skip_connection(self, input_dim: int, x: torch.Tensor) -> None:
        model = ChurnMLPv2(input_dim=input_dim, hidden_dims=[64, 32], use_skip=False)
        assert model.skip_proj is None
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (x.shape[0], 1)

    def test_hidden_dims_default_quando_none(self, input_dim: int) -> None:
        model = ChurnMLPv2(input_dim=input_dim, hidden_dims=None)
        assert model.hidden_dims is not None
        assert len(model.hidden_dims) > 0

    def test_seed_garante_reproducibilidade(self, input_dim: int) -> None:
        m1 = ChurnMLPv2(input_dim=input_dim, hidden_dims=[32], seed=42)
        m2 = ChurnMLPv2(input_dim=input_dim, hidden_dims=[32], seed=42)
        for p1, p2 in zip(m1.parameters(), m2.parameters(), strict=True):
            assert torch.equal(p1, p2)

    def test_backward_produz_gradientes(self, input_dim: int, x: torch.Tensor) -> None:
        model = ChurnMLPv2(input_dim=input_dim, hidden_dims=[32])
        model.train()
        y = torch.randint(0, 2, (x.shape[0], 1)).float()
        loss = torch.nn.BCEWithLogitsLoss()(model(x), y)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"Sem gradiente em {name}"

    def test_tem_parametros_treinaveis(self, input_dim: int) -> None:
        model = ChurnMLPv2(input_dim=input_dim, hidden_dims=[64, 32])
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n > 0

    def test_camada_saida_shape(self, input_dim: int) -> None:
        model = ChurnMLPv2(input_dim=input_dim, hidden_dims=[64, 32])
        assert model.output.out_features == 1


class TestChurnMLPInference:
    def test_constroi_com_defaults(self, input_dim: int) -> None:
        model = ChurnMLPInference(input_dim=input_dim)
        assert model.output.out_features == 1

    def test_constroi_com_hidden_dims_none(self, input_dim: int) -> None:
        model = ChurnMLPInference(input_dim=input_dim, hidden_dims=None)
        assert model.output is not None

    def test_forward_shape_correto(
        self, input_dim: int, x: torch.Tensor, batch_size: int
    ) -> None:
        model = ChurnMLPInference(input_dim=input_dim, hidden_dims=[64, 32])
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (batch_size, 1)

    def test_forward_batch_size_1(self, input_dim: int) -> None:
        model = ChurnMLPInference(input_dim=input_dim, hidden_dims=[64, 32])
        model.eval()
        x = torch.randn(1, input_dim)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1)

    def test_estado_dict_carregavel(self, input_dim: int) -> None:
        m1 = ChurnMLPInference(input_dim=input_dim, hidden_dims=[32])
        state = m1.state_dict()
        m2 = ChurnMLPInference(input_dim=input_dim, hidden_dims=[32])
        m2.load_state_dict(state)
        x = torch.randn(2, input_dim)
        m1.eval()
        m2.eval()
        with torch.no_grad():
            assert torch.allclose(m1(x), m2(x))
