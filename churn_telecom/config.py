from pathlib import Path

from fastapi import logger

import logging


# ── Reprodutibilidade ──────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Paths ──────────────────────────────────────────────────────────────────────
# Path(__file__) = churn_telecom/config.py
# .parent        = churn_telecom/
# .parent.parent = raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW       = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM   = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS_FIGURES = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR     = PROJECT_ROOT / "models"

# ── Paleta de cores — padrão visual do projeto ────────────────────────────────
CORES = {
    "primaria":   "skyblue",
    "secundaria": "steelblue",
    "destaque":   "coral",
    "neutro":     "dimgray",
    "alerta":     "#E84C6C",
}

# ── Colunas por categoria ──────────────────────────────────────────────────────
COLS_ID = [
    "CustomerID", "Count",
    "Country", "State", "City", "Zip Code", "Lat Long", "Latitude", "Longitude",
]

COLS_NUM = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
]

COLS_CAT = [
    "Gender", "Senior Citizen", "Partner", "Dependents",
    "Phone Service", "Multiple Lines", "Internet Service",
    "Online Security", "Online Backup", "Device Protection",
    "Tech Support", "Streaming TV", "Streaming Movies",
    "Contract", "Paperless Billing", "Payment Method",
]

COLS_POS = [
    "Churn Score",
    "CLTV",
    "Churn Reason",
]

TARGET    = "Churn Value"
LABEL_COL = "Churn Label"

# churn_telecom/config.py
def get_logger(notebook_name: str) -> logging.Logger:
    """Retorna logger configurado para o notebook especificado."""
    log_file = PROJECT_ROOT / "logs" / f"{notebook_name}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(notebook_name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(log_file, encoding="utf-8"))

    return logger

import mlflow
from pathlib import Path

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
MLFLOW_ARTIFACT_URI = (PROJECT_ROOT / "mlartifacts").as_uri()  # file:///C:/...
MLFLOW_EXPERIMENT   = "churn-telecom"

# logger interno do módulo config — independente dos notebooks
_logger = logging.getLogger(__name__)

def setup_mlflow() -> None:
    """Configura tracking URI, artifact URI e cria o experimento se não existir."""
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