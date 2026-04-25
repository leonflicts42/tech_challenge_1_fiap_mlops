"""models — MLP, trainer e avaliação."""

from models.mlp import ChurnMLP, build_mlp
from models.trainer import ChurnTrainer, EarlyStopping, TrainerConfig
from models.evaluation import (
    CostAnalyzer,
    CostConfig,
    MetricsCalculator,
    ModelComparator,
    ModelMetrics,
)

__all__ = [
    # MLP (Etapa 1)
    "ChurnMLP",
    "build_mlp",
    # Trainer (Etapa 2)
    "ChurnTrainer",
    "EarlyStopping",
    "TrainerConfig",
    # Evaluation (Etapa 2)
    "MetricsCalculator",
    "ModelMetrics",
    "CostAnalyzer",
    "CostConfig",
    "ModelComparator",
]
