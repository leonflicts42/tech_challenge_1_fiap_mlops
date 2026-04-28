"""models — MLP, trainer e avaliação."""

from models.mlp import ChurnMLPInference, ChurnMLPv2
from models.trainer import ChurnTrainer, EarlyStopping, TrainerConfig
from models.evaluation import (
    CostAnalyzer,
    CostConfig,
    MetricsCalculator,
    ModelComparator,
    ModelMetrics,
)

__all__ = [
    # MLP
    "ChurnMLPv2",
    "ChurnMLPInference",
    # Trainer
    "ChurnTrainer",
    "EarlyStopping",
    "TrainerConfig",
    # Evaluation
    "MetricsCalculator",
    "ModelMetrics",
    "CostAnalyzer",
    "CostConfig",
    "ModelComparator",
]
