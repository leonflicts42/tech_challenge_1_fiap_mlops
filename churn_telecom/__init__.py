"""churn_telecom.models — MLP, trainer e avaliação."""

from churn_telecom.models.mlp import ChurnMLP, build_mlp
from churn_telecom.models.trainer import ChurnTrainer, EarlyStopping, TrainerConfig
from churn_telecom.models.evaluation import (
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