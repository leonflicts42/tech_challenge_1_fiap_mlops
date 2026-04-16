# churn_telecom/config.py
"""Configurações centrais do projeto churn-telecom.

Única fonte de verdade para paths, constantes, logging e MLflow.
"""

from __future__ import annotations

import hashlib
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import torch

if TYPE_CHECKING:
    import pandas as pd

# ── Reprodutibilidade ──────────────────────────────────────────────────────────
RANDOM_STATE: int = 42

# ── Validação do dataset original ─────────────────────────────────────────────
DATA_ROWS: int = 7043
DATA_COLS: int = 33

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS_FIGURES = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

# Garante que os diretórios existam na inicialização
for _dir in (DATA_RAW, DATA_INTERIM, DATA_PROCESSED, REPORTS_FIGURES, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ── Paleta de cores ────────────────────────────────────────────────────────────
CORES: dict[str, str] = {
    "primaria": "skyblue",
    "secundaria": "steelblue",
    "destaque": "coral",
    "neutro": "dimgray",
    "alerta": "#E84C6C",
}

# ── Colunas por categoria ──────────────────────────────────────────────────────
COLS_ID = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
]

COLS_NUM = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
]

COLS_CAT = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
]

COLS_POS = [
    "Churn Score",
    "CLTV",
    "Churn Reason",
]

TARGET = "Churn Value"
TARGET_COL = "churn_value"
LABEL_COL = "Churn Label"

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
MLFLOW_ARTIFACT_URI = (PROJECT_ROOT / "mlartifacts").as_uri()
MLFLOW_EXPERIMENT = "churn-telecom"

# ── Logger interno do módulo config ───────────────────────────────────────────
_logger = logging.getLogger(__name__)


# ── Funções utilitárias ───────────────────────────────────────────────────────


def get_logger(notebook_name: str) -> logging.Logger:
    """Retorna logger configurado por notebook, com handler de arquivo e console."""
    log_file = PROJECT_ROOT / "logs" / f"{notebook_name}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(notebook_name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def to_snake_case(name: str) -> str:
    """Padroniza nome de coluna para snake_case."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def setup_mlflow() -> None:
    """Configura tracking URI e cria o experimento se não existir."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        mlflow.create_experiment(
            name=MLFLOW_EXPERIMENT,
            artifact_location=MLFLOW_ARTIFACT_URI,
        )
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    _logger.info("MLflow tracking URI : %s", MLFLOW_TRACKING_URI)
    _logger.info("MLflow experiment   : %s", MLFLOW_EXPERIMENT)
    _logger.info("MLflow artifact URI : %s", MLFLOW_ARTIFACT_URI)


@contextmanager
def mlflow_run(run_name: str):
    """Context manager que configura MLflow e inicia um run nomeado."""
    setup_mlflow()
    with mlflow.start_run(run_name=run_name) as run:
        yield run


def log_dataset_to_mlflow(
    X: pd.DataFrame,
    y: pd.Series,
    split: str,
    source_path: Path | str,
) -> None:
    """
    Loga dataset no MLflow via log_input + tags de versionamento.

    Parâmetros
    ----------
    X           : features (DataFrame)
    y           : target (Series)
    split       : "train" | "test"
    source_path : caminho do parquet de origem (usado para hash MD5)
    """
    import pandas as pd  # import local para evitar ciclo na inicialização

    source_path = Path(source_path)
    md5 = hashlib.md5(source_path.read_bytes()).hexdigest()

    dataset = mlflow.data.from_pandas(
        pd.concat([X, y], axis=1),
        source=str(source_path),
        name=f"telco_{split}",
        targets=str(y.name),
    )
    mlflow.log_input(dataset, context=split)

    mlflow.set_tags(
        {
            f"dataset.{split}.rows": str(X.shape[0]),
            f"dataset.{split}.cols": str(X.shape[1]),
            f"dataset.{split}.md5": md5,
            f"dataset.{split}.source": source_path.name,
            f"dataset.{split}.churn_rate": f"{y.mean():.4f}",
        }
    )

    _logger.info(
        "Dataset '%s' logado | shape=(%d, %d) | md5=%s",
        f"telco_{split}",
        X.shape[0],
        X.shape[1],
        md5,
    )

# ── PyTorch / MLP ──────────────────────────────────────────────────────────────
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

MLP_HIDDEN_DIMS: list[int] = [64, 32]
MLP_DROPOUT: float = 0.3
MLP_LR: float = 1e-3
MLP_WEIGHT_DECAY: float = 1e-4
MLP_BATCH_SIZE: int = 64
MLP_MAX_EPOCHS: int = 100
MLP_PATIENCE: int = 10
MLP_MONITOR_METRIC: str = "val_pr_auc"   # "val_pr_auc" | "val_loss" | "val_recall"

# ── Validação cruzada ─────────────────────────────────────────────────────────
CV_N_SPLITS: int = 5

# ── Custos de negócio (trade-off FP × FN) ─────────────────────────────────────
COST_FN: float = 500.0   # cliente perdido sem tentativa de retenção
COST_FP: float = 50.0    # campanha de retenção desperdiçada