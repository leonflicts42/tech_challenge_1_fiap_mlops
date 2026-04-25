"""
config.py — Fonte única de verdade do projeto Churn Telecom.

Todas as constantes, caminhos, parâmetros de negócio, configurações de
MLflow/PyTorch e utilitários compartilhados são definidos aqui.
Nenhum outro módulo deve hardcodar seeds, paths, nomes de colunas ou
limiares de negócio.

Regra de importação:
    from churn_telecom.config import RANDOM_STATE, get_logger, ...
"""

from __future__ import annotations

import hashlib
import logging
import re
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import torch

if TYPE_CHECKING:
    import pandas as pd

# ── Reprodutibilidade ─────────────────────────────────────────────────────────
# Seed único usado em: numpy, random, sklearn (random_state), torch.manual_seed
RANDOM_STATE: int = 42

# ── Validação do dataset original ─────────────────────────────────────────────
DATA_ROWS: int = 7_043
DATA_COLS: int = 33

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM: Path = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"

REPORTS_PATH: Path = PROJECT_ROOT / "reports"
REPORTS_FIGURES: Path = REPORTS_PATH / "figures"
REPORTS_FIGURES_BASELINES: Path = REPORTS_FIGURES / "baselines"
REPORTS_FIGURES_UNIVARIADA: Path = REPORTS_FIGURES / "univariada"
REPORTS_FIGURES_MULTIVARIADA: Path = REPORTS_FIGURES / "multivariada"
REPORTS_FIGURES_MLP: Path = REPORTS_FIGURES / "mlp"
REPORT_FIGURES_CORRELACAO: Path = REPORTS_FIGURES_MLP / "correlacao"

# Caminhos de arquivos finais (não incluídos no mkdir — são arquivos, não pastas)
TRAIN_PATH: Path = DATA_PROCESSED / "train.parquet"
TEST_PATH: Path = DATA_PROCESSED / "test.parquet"
OUTPUT_PATH: Path = DATA_INTERIM / "telco_droped.parquet"

# Criação automática de diretórios na importação do módulo
for _dir in (
    DATA_RAW,
    DATA_INTERIM,
    DATA_PROCESSED,
    MODELS_DIR,
    REPORTS_FIGURES,
    REPORTS_FIGURES_BASELINES,
    REPORTS_FIGURES_UNIVARIADA,
    REPORTS_FIGURES_MULTIVARIADA,
    REPORTS_FIGURES_MLP,
    REPORT_FIGURES_CORRELACAO,
):
    _dir.mkdir(parents=True, exist_ok=True)

# ── Target ────────────────────────────────────────────────────────────────────
TARGET: str = "Churn Value"      # nome original no CSV (pré-snake_case)
LABEL_COL: str = "Churn Label"   # versão string do target — removida no cleaning
TARGET_COL: str = "churn_value"  # nome snake_case usado em todo o pipeline

# ── Parâmetros do split ───────────────────────────────────────────────────────
TEST_SIZE: float = 0.2  # 20% para holdout
N_SPLITS: int = 5       # folds da validação cruzada estratificada

# ── Custos de negócio (R$) ────────────────────────────────────────────────────
# FN: churner não detectado → perde o CLV do cliente
# FP: cliente retido desnecessariamente → custo da campanha de retenção
COST_CLV: float = 2_845.00        # R$ — receita anual média por cliente (CLV)
COST_RETENTION: float = 73.52     # R$ — custo unitário de ação de retenção
COST_FN: float = COST_CLV         # R$ — custo de perder churner não detectado
COST_FP: float = COST_RETENTION   # R$ — custo de abordar não-churner
COST_RATIO: float = COST_FN / COST_FP  # ≈ 38.7 — FN custa ~39× mais que FP

# ── SLO operacional ───────────────────────────────────────────────────────────
# Threshold ótimo selecionado entre os que atendem Recall ≥ SLO_RECALL_MIN
SLO_RECALL_MIN: float = 0.70  # Recall mínimo aceitável em produção

# ── Varredura de threshold ────────────────────────────────────────────────────
THRESHOLD_MIN: float = 0.05
THRESHOLD_MAX: float = 0.94
THRESHOLD_STEP: float = 0.01
REPORTS_JSON_DIR: Path = PROJECT_ROOT / "reports" / "json"
OPTUNA_PARAMS_PATH: Path = REPORTS_JSON_DIR / "optuna_best_params.json"

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI: str = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
MLFLOW_ARTIFACT_URI: str = (PROJECT_ROOT / "mlartifacts").as_uri()
MLFLOW_EXPERIMENT: str = "churn-telecom"

# ── PyTorch / MLP ─────────────────────────────────────────────────────────────
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# Arquitetura
MLP_HIDDEN_DIMS: list[int] = [128, 64, 32]
MLP_DROPOUT: float = 0.3

# Otimização
MLP_LR: float = 1e-3
MLP_WEIGHT_DECAY: float = 1e-4
MLP_EPOCHS: int = 100
MLP_BATCH_SIZE: int = 256  # patch: 256 (melhor para datasets ~5k)

# Early stopping
MLP_PATIENCE: int = 10
MLP_MIN_DELTA: float = 1e-4
MLP_MONITOR_METRIC: str = "val_pr_auc"  # "val_pr_auc" | "val_loss" | "val_recall"

# ── Validação cruzada ─────────────────────────────────────────────────────────
CV_N_SPLITS: int = 5
CV_RANDOM_STATE: int = RANDOM_STATE  # mesma semente global — sem fonte paralela

# ── Paleta de cores (visualizações) ───────────────────────────────────────────
CORES: dict[str, str] = {
    "primaria": "skyblue",
    "secundaria": "steelblue",
    "destaque": "coral",
    "neutro": "dimgray",
    "alerta": "#E84C6C",
}

# ── Colunas por categoria (nomes ORIGINAIS do dataset, pré-snake_case) ────────

# Identificadores e geolocalização — sem poder preditivo, removidos no cleaning
COLS_ID: list[str] = [
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

# Colunas geradas APÓS o evento de churn — data leakage, removidas no cleaning
COLS_POS: list[str] = [
    "Churn Score",
    "CLTV",
    "Churn Reason",
]

# Features numéricas contínuas do dataset original
COLS_NUM: list[str] = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
]

# Features categóricas do dataset original
COLS_CAT: list[str] = [
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

# ── Colunas do pipeline de inferência (snake_case) ────────────────────────────

# Serviços dependentes de internet que recebem normalização semântica:
# "No internet service" → "No" no SemanticNormalizer
NO_SERVICE_COLS: list[str] = [
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]

# Colunas usadas para contar serviços ativos na feature num_services
SERVICES_COLS: list[str] = [
    "phone_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]

# Numéricas → SimpleImputer(median) → log1p → StandardScaler
COLS_NUM_PIPE: list[str] = [
    "tenure_months",
    "monthly_charges",
    "total_charges",      # skewness=0.962 — log1p corrige assimetria
    "num_services",       # criada pelo FeatureEngineer
    "charges_per_month",  # criada pelo FeatureEngineer
]

# Binárias Yes/No → OrdinalEncoder(["No","Yes"]) → No=0, Yes=1 (ordem garantida)
COLS_BINARY_PIPE: list[str] = [
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "paperless_billing",
    "senior_citizen",     # já é 0/1 no raw — OrdinalEncoder preserva semântica
]

# Nominais (3+ categorias) → OneHotEncoder(drop="first") — evita dummy trap
COLS_OHE_PIPE: list[str] = [
    "internet_service",   # DSL / Fiber optic / No
    "contract",           # Month-to-month / One year / Two year
    "payment_method",     # 4 métodos de pagamento
    "gender",             # Male / Female — mantido para análise de viés
    "tenure_group",       # novo / medio / longo — criada pelo FeatureEngineer
]

# Features binárias criadas pelo FeatureEngineer → passthrough (já são int 0/1)
COLS_PASS_PIPE: list[str] = [
    "is_month_to_month",
    "has_security_support",
    "is_fiber_optic",
]

# Todas as features criadas pelo FeatureEngineer
NEW_FEATURES: list[str] = [
    "num_services",
    "charges_per_month",
    "is_month_to_month",
    "tenure_group",
    "has_security_support",
    "is_fiber_optic",
]

# Subset binário das novas features (validação de range {0,1} nos testes)
BINARY_NEW_FEATURES: list[str] = [
    "is_month_to_month",
    "has_security_support",
    "is_fiber_optic",
]

# Número esperado de features após o ColumnTransformer — contrato dos testes
N_FEATURES_FINAL: int = 30

# ── Logger interno do módulo config ───────────────────────────────────────────
_logger = logging.getLogger(__name__)


# ── Utilitários ───────────────────────────────────────────────────────────────


def to_snake_case(name: str) -> str:
    """Converte qualquer string para snake_case.

    Trata espaços, hífens E CamelCase — único ponto de conversão do projeto.
    Usado em notebooks, pipeline e API para garantir consistência de nomes.

    Exemplos:
        >>> to_snake_case("TenureMonths")
        'tenure_months'
        >>> to_snake_case("Monthly Charges")
        'monthly_charges'
        >>> to_snake_case("Churn Value")
        'churn_value'
        >>> to_snake_case("PhoneService")
        'phone_service'
        >>> to_snake_case("  Total  Charges  ")
        'total_charges'
    """
    name = name.strip()
    # CamelCase → insere underscore antes de maiúscula precedida de minúscula
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    # Espaços e hífens → underscore; colapsa múltiplos underscores
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.lower().strip("_")


def get_logger(name: str) -> logging.Logger:
    """Retorna logger configurado com saída em console e arquivo.

    Formato: HH:MM:SS | LEVEL | message

    Reconfigura handlers a cada chamada — evita acúmulo de handlers
    duplicados em notebooks com autoreload (%autoreload 2).

    Args:
        name: identificador do módulo/notebook (ex: "1.04_preprocessing")

    Returns:
        Logger configurado com console handler e file handler.

    Exemplo:
        logger = get_logger(__name__)
        logger.info("Iniciando | shape=%s", df.shape)
    """
    log_file = PROJECT_ROOT / "logs" / f"{name}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.handlers.clear()  # evita duplicação em notebooks com autoreload
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


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
    """Context manager que configura MLflow e inicia um run nomeado.

    Exemplo:
        with mlflow_run("baseline_lr") as run:
            mlflow.log_param("C", 1.0)
            mlflow.log_metric("roc_auc", 0.85)
    """
    setup_mlflow()
    with mlflow.start_run(run_name=run_name) as run:
        yield run


def log_dataset_to_mlflow(
    X: pd.DataFrame,
    y: pd.Series,
    split: str,
    source_path: Path | str,
) -> None:
    """Loga dataset no MLflow via log_input + tags de versionamento MD5.

    Args:
        X:           features (DataFrame)
        y:           target (Series)
        split:       "train" | "test"
        source_path: caminho do parquet de origem (usado para hash MD5)
    """
    import pandas as _pd  # import local evita ciclo na inicialização do módulo

    source_path = Path(source_path)
    md5 = hashlib.md5(source_path.read_bytes()).hexdigest()

    dataset = mlflow.data.from_pandas(
        _pd.concat([X, y], axis=1),
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