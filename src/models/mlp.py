"""Definição da arquitetura MLP para predição de churn.

Este módulo contém **somente** a definição da rede neural (SRP).
O loop de treino, dataset/loader e otimização de threshold ficam em módulos
separados (`train.py`, `dataset.py`, `threshold.py`).

A saída do `forward` é um **logit puro** (sem Sigmoid), por dois motivos:
1. Permite usar `BCEWithLogitsLoss`, numericamente mais estável
   (combina log-sum-exp internamente).
2. Mantém a rede agnóstica ao threshold de decisão, que é definido
   posteriormente via análise de custo (FP × FN).
"""

from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


class ChurnMLP(nn.Module):
    """MLP feed-forward para classificação binária de churn.

    Cada bloco oculto: ``Linear → BatchNorm1d → ReLU → Dropout``.
    Camada de saída: ``Linear(last_hidden, 1)`` retornando logit.

    Parâmetros
    ----------
    in_features  : número de features de entrada (após pré-processamento).
    hidden_dims  : lista com o tamanho de cada camada oculta. Ex: [64, 32].
    dropout      : probabilidade de dropout aplicada em cada bloco oculto.

    Exemplo
    -------
    >>> import torch
    >>> model = ChurnMLP(in_features=30, hidden_dims=[64, 32], dropout=0.3)
    >>> x = torch.randn(8, 30)
    >>> logits = model(x)
    >>> logits.shape
    torch.Size([8, 1])
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: list[int],
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if in_features <= 0:
            raise ValueError(f"in_features deve ser > 0, recebido: {in_features}")
        if not hidden_dims:
            raise ValueError("hidden_dims não pode ser vazio.")
        if any(h <= 0 for h in hidden_dims):
            raise ValueError(f"hidden_dims devem ser > 0, recebido: {hidden_dims}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout deve estar em [0, 1), recebido: {dropout}")

        self.in_features = in_features
        self.hidden_dims = list(hidden_dims)
        self.dropout = dropout

        layers: list[nn.Module] = []
        prev_dim = in_features
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            prev_dim = hidden_dim

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 1)

        self._init_weights()

        logger.info(
            "ChurnMLP construída | in=%d | hidden=%s | dropout=%.2f | params=%d",
            in_features,
            hidden_dims,
            dropout,
            self.count_parameters(),
        )

    def _init_weights(self) -> None:
        """Inicializa pesos: Kaiming-Normal para Linear+ReLU, zeros nos bias."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Retorna logits de shape (batch_size, 1)."""
        return self.output(self.hidden(x))

    def count_parameters(self) -> int:
        """Retorna o número de parâmetros treináveis."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_mlp(
    in_features: int,
    hidden_dims: list[int] | None = None,
    dropout: float | None = None,
    device: str | None = None,
    seed: int | None = None,
) -> ChurnMLP:
    """Factory para construir a MLP com defaults vindos de ``config``.

    Centraliza o ponto de criação para garantir:
    - Seed fixada antes da inicialização dos pesos (reprodutibilidade).
    - Device coerente com ``config.DEVICE``.
    - Hiperparâmetros default vindos do ``config`` (single source of truth).

    Parâmetros
    ----------
    in_features : número de features de entrada.
    hidden_dims : sobrescreve ``config.MLP_HIDDEN_DIMS`` se fornecido.
    dropout     : sobrescreve ``config.MLP_DROPOUT`` se fornecido.
    device      : sobrescreve ``config.DEVICE`` se fornecido.
    seed        : sobrescreve ``config.RANDOM_STATE`` se fornecido.
    """
    from config import (
        MLP_HIDDEN_DIMS,
        MLP_DROPOUT,
        DEVICE,
        RANDOM_STATE,
    )

    hidden_dims = hidden_dims if hidden_dims is not None else MLP_HIDDEN_DIMS
    dropout = dropout if dropout is not None else MLP_DROPOUT
    device = device if device is not None else DEVICE
    seed = seed if seed is not None else RANDOM_STATE

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = ChurnMLP(
        in_features=in_features,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    logger.info("MLP movida para device=%s | seed=%d", device, seed)
    return model
