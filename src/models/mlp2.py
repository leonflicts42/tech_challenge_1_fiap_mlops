"""
models/mlp.py — Arquitetura ChurnMLPv2.

Define a rede neural MLP usada para predição de churn.
Esta definição é a **fonte única de verdade** da arquitetura — usada tanto
no treinamento (notebooks) quanto na inferência (API).

Importar para carregar pesos salvos:
    from churn_telecom.models.mlp import ChurnMLPv2
    model = ChurnMLPv2(input_dim=30)
    model.load_state_dict(torch.load("models/best_mlp.pt", map_location="cpu"))
    model.eval()
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from config import (
    MLP_DROPOUT,
    MLP_HIDDEN_DIMS,
    N_FEATURES_FINAL,
    RANDOM_STATE,
    get_logger,
)

logger: logging.Logger = get_logger(__name__)


class ChurnMLPv2(nn.Module):
    """Rede MLP para predição binária de churn.

    Arquitetura:
        Input(30) → [Linear → LayerNorm → ReLU → Dropout] × L → Linear(1)

    Decisões de design:
        LayerNorm em vez de BatchNorm:
            Estável com qualquer batch size — crítico para inferência online
            onde o batch é de tamanho 1.

        Skip connection (opcional):
            Conexão residual do input até a penúltima camada.
            Ajuda o gradiente a fluir em redes mais profundas.

        Kaiming-Normal init:
            Inicialização adequada para ativações ReLU — evita vanishing
            gradient nas primeiras épocas.

        Saída: logit puro (sem sigmoid):
            BCEWithLogitsLoss aplica sigmoid internamente durante o treino,
            o que é numericamente mais estável. Na inferência, sigmoid é
            aplicado explicitamente no ChurnPredictor.

    Args:
        input_dim:   número de features de entrada (default: N_FEATURES_FINAL=30)
        hidden_dims: lista com o tamanho de cada camada oculta
        dropout:     taxa de dropout aplicada após cada bloco oculto
        use_skip:    se True, adiciona skip connection input → penúltima camada
        seed:        semente para reprodutibilidade da inicialização

    Exemplo:
        model = ChurnMLPv2(input_dim=30)
        x = torch.randn(64, 30)   # batch de 64 amostras
        logits = model(x)          # shape (64, 1)
        probs  = torch.sigmoid(logits)
    """

    def __init__(
        self,
        input_dim: int = N_FEATURES_FINAL,
        hidden_dims: list[int] | None = None,
        dropout: float = MLP_DROPOUT,
        use_skip: bool = True,
        seed: int = RANDOM_STATE,
    ) -> None:
        super().__init__()

        torch.manual_seed(seed)

        if hidden_dims is None:
            hidden_dims = MLP_HIDDEN_DIMS  # [128, 64, 32]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.use_skip = use_skip

        # ── Blocos ocultos: Linear → LayerNorm → ReLU → Dropout ──────────────
        blocks: list[nn.Module] = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            blocks.append(
                nn.Sequential(
                    nn.Linear(in_dim, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                )
            )
            in_dim = h_dim

        self.blocks = nn.ModuleList(blocks)

        # ── Skip connection: projeta input para o tamanho da última camada ────
        # Ativado apenas se input_dim ≠ hidden_dims[-1]
        if use_skip and input_dim != hidden_dims[-1]:
            self.skip_proj = nn.Linear(input_dim, hidden_dims[-1], bias=False)
        else:
            self.skip_proj = None

        # ── Camada de saída: logit escalar ────────────────────────────────────
        self.output = nn.Linear(hidden_dims[-1], 1)

        # ── Inicialização Kaiming-Normal para ReLU ────────────────────────────
        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "ChurnMLPv2 | input_dim=%d | hidden_dims=%s | dropout=%.1f"
            " | use_skip=%s | params=%d",
            input_dim,
            hidden_dims,
            dropout,
            use_skip,
            n_params,
        )

    def _init_weights(self) -> None:
        """Inicialização Kaiming-Normal para camadas Linear com ReLU."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passa o tensor pelo modelo e retorna o logit.

        Args:
            x: tensor de shape (batch_size, input_dim) — float32

        Returns:
            logit: tensor de shape (batch_size, 1) — float32
                   Para obter probabilidade: torch.sigmoid(logit)
        """
        skip = x  # guarda para skip connection

        for block in self.blocks:
            x = block(x)

        # Aplica skip connection na saída da última camada oculta
        if self.use_skip:
            if self.skip_proj is not None:
                x = x + self.skip_proj(skip)
            else:
                x = x + skip

        return self.output(x)  # logit shape (batch, 1)


class ChurnMLPInference(nn.Module):
    """MLP com Sequential flat — compatível com o state_dict de best_model_mlp.pt.

    Chaves do state_dict: hidden.0 (Linear), hidden.1 (LayerNorm),
    hidden.4 (Linear), hidden.5 (LayerNorm), output (Linear).
    Use esta classe para carregar checkpoints gerados pelo notebook 4.

    Args:
        input_dim:   número de features (default: N_FEATURES_FINAL=30)
        hidden_dims: dimensões das camadas ocultas (winner: [128, 64])
        dropout:     taxa de dropout (winner: 0.15)
    """

    def __init__(
        self,
        input_dim: int = N_FEATURES_FINAL,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers += [nn.Linear(in_dim, h_dim), nn.LayerNorm(h_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h_dim

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.hidden(x))