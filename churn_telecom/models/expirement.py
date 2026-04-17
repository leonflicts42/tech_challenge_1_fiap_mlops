"""
Experiment runner — Etapa 2.

Treina DummyClassifier, LogisticRegression, RandomForest e ChurnMLP,
registra todos os artefatos no MLflow e gera tabela comparativa.

Uso:
    python -m churn_telecom.models.experiment
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import mlflow
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Importações internas — assumem que o pacote está instalado (pip install -e .)
from churn_telecom.models.evaluation import (
    CostAnalyzer,
    CostConfig,
    MetricsCalculator,
    ModelComparator,
)
from churn_telecom.models.mlp import build_mlp
from churn_telecom.models.trainer import ChurnTrainer, TrainerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

SEED = 42
N_SPLITS = 5
MLFLOW_EXPERIMENT = "churn_telecom_etapa2"
THRESHOLD = 0.5


# ── Helpers ──────────────────────────────────────────────────────────────────


def _load_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Carrega features e target já pré-processados pelo pipeline da Etapa 1.

    Adapte este caminho conforme a estrutura real do projeto.
    """
    data_path = Path("data/processed/features.npy")
    target_path = Path("data/processed/target.npy")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {data_path}\n"
            "Execute o notebook de EDA/preprocessamento antes deste script."
        )

    X = np.load(data_path)
    y = np.load(target_path)
    logger.info(
        "dados carregados | X=%s | y=%s | churn_rate=%.2f%%",
        X.shape,
        y.shape,
        y.mean() * 100,
    )
    return X, y


def _pos_weight(y_train: np.ndarray) -> float:
    """Calcula o peso para a classe positiva (para BCEWithLogitsLoss)."""
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    return float(n_neg / max(n_pos, 1))


# ── Treinamento de baselines sklearn ────────────────────────────────────────


def _train_sklearn_baseline(
    name: str,
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict,
) -> np.ndarray:
    """Treina um estimador sklearn e registra no MLflow. Retorna y_proba."""
    with mlflow.start_run(run_name=name, nested=True):
        mlflow.log_params(params)
        estimator.fit(X_train, y_train)

        if hasattr(estimator, "predict_proba"):
            y_proba = estimator.predict_proba(X_val)[:, 1]
        else:
            # DummyClassifier pode não ter predict_proba com strategy="most_frequent"
            y_proba = estimator.predict(X_val).astype(float)

        from sklearn.metrics import roc_auc_score, average_precision_score

        mlflow.log_metrics(
            {
                "val_roc_auc": float(roc_auc_score(y_val, y_proba)),
                "val_pr_auc": float(average_precision_score(y_val, y_proba)),
            }
        )
        logger.info("%s | treinamento concluído", name)
        return y_proba


# ── Treinamento MLP ──────────────────────────────────────────────────────────


def _train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
) -> tuple[np.ndarray, object]:
    """Treina a ChurnMLP e registra no MLflow. Retorna (y_proba, trainer)."""
    trainer_cfg = TrainerConfig(
        lr=1e-3,
        epochs=100,
        batch_size=256,
        patience=10,
        weight_decay=1e-4,
        device="cpu",
        seed=SEED,
        pos_weight=_pos_weight(y_train),
    )
    mlp_cfg = {
        "hidden_dims": [128, 64, 32],
        "dropout": 0.3,
        "input_dim": input_dim,
    }

    with mlflow.start_run(run_name="ChurnMLP", nested=True):
        mlflow.log_params(
            {
                **mlp_cfg,
                **{
                    "lr": trainer_cfg.lr,
                    "epochs": trainer_cfg.epochs,
                    "batch_size": trainer_cfg.batch_size,
                    "patience": trainer_cfg.patience,
                    "weight_decay": trainer_cfg.weight_decay,
                    "pos_weight": trainer_cfg.pos_weight,
                },
            }
        )

        model = build_mlp(
            input_dim=input_dim,
            hidden_dims=mlp_cfg["hidden_dims"],
            dropout=mlp_cfg["dropout"],
            seed=SEED,
        )
        trainer = ChurnTrainer(model, trainer_cfg)
        history = trainer.fit(X_train, y_train, X_val, y_val)

        mlflow.log_metrics(
            {
                "best_epoch": history.best_epoch,
                "stopped_early": int(history.stopped_early),
            }
        )

        y_proba = trainer.predict_proba(X_val)
        logger.info(
            "MLP | stopped_early=%s | best_epoch=%d",
            history.stopped_early,
            history.best_epoch,
        )
        return y_proba, trainer


# ── CV estratificado para avaliação robusta ───────────────────────────────────


def _cv_evaluate(
    name: str,
    get_estimator_fn,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = N_SPLITS,
) -> np.ndarray:
    """
    Validação cruzada estratificada — retorna probabilidades OOF (out-of-fold).

    Usado para calcular métricas finais com estimativas não enviesadas.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof_proba = np.zeros(len(y), dtype=float)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_vl = X[train_idx], X[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]  # noqa: F841

        estimator = get_estimator_fn()
        estimator.fit(X_tr, y_tr)

        if hasattr(estimator, "predict_proba"):
            oof_proba[val_idx] = estimator.predict_proba(X_vl)[:, 1]
        else:
            oof_proba[val_idx] = estimator.predict(X_vl).astype(float)

        logger.info("CV | %s | fold %d/%d concluído", name, fold, n_splits)

    return oof_proba


# ── Runner principal ──────────────────────────────────────────────────────────


def run_experiment() -> None:
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    X, y = _load_data()
    input_dim = X.shape[1]

    # Split treino/val estratificado (80/20) para o run final
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    calc = MetricsCalculator(threshold=THRESHOLD)
    cost_cfg = CostConfig()
    cost_analyzer = CostAnalyzer(cost_cfg)
    comparator = ModelComparator()

    with mlflow.start_run(run_name="etapa2_comparacao"):
        mlflow.log_params(
            {
                "seed": SEED,
                "n_splits_cv": N_SPLITS,
                "threshold": THRESHOLD,
                "cost_fp": cost_cfg.fp_cost,
                "cost_fn": cost_cfg.fn_cost,
                "clv": cost_cfg.clv_per_customer,
            }
        )

        # ── Dummy ────────────────────────────────────────────────────────────
        dummy_proba = _train_sklearn_baseline(
            name="DummyClassifier",
            estimator=DummyClassifier(strategy="stratified", random_state=SEED),
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            params={"strategy": "stratified"},
        )
        m_dummy = calc.compute("DummyClassifier", y_val, dummy_proba)
        comparator.add(cost_analyzer.annotate(m_dummy))

        # ── Logistic Regression ───────────────────────────────────────────────
        lr_pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=1.0,
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=SEED,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        lr_proba = _train_sklearn_baseline(
            name="LogisticRegression",
            estimator=lr_pipe,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            params={"C": 1.0, "class_weight": "balanced"},
        )
        m_lr = calc.compute("LogisticRegression", y_val, lr_proba)
        comparator.add(cost_analyzer.annotate(m_lr))

        # ── Random Forest ─────────────────────────────────────────────────────
        rf_proba = _train_sklearn_baseline(
            name="RandomForest",
            estimator=RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
            ),
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            params={"n_estimators": 200, "max_depth": 8},
        )
        m_rf = calc.compute("RandomForest", y_val, rf_proba)
        comparator.add(cost_analyzer.annotate(m_rf))

        # ── Gradient Boosting ─────────────────────────────────────────────────
        gb_proba = _train_sklearn_baseline(
            name="GradientBoosting",
            estimator=GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=SEED,
            ),
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            params={"n_estimators": 200, "max_depth": 4, "lr": 0.05},
        )
        m_gb = calc.compute("GradientBoosting", y_val, gb_proba)
        comparator.add(cost_analyzer.annotate(m_gb))

        # ── MLP ───────────────────────────────────────────────────────────────
        mlp_proba, _ = _train_mlp(X_train, y_train, X_val, y_val, input_dim)
        m_mlp = calc.compute("ChurnMLP", y_val, mlp_proba)
        comparator.add(cost_analyzer.annotate(m_mlp))

        # ── Tabela comparativa ────────────────────────────────────────────────
        summary_df = comparator.summary()
        tradeoff_df = cost_analyzer.tradeoff_summary([m_dummy, m_lr, m_rf, m_gb, m_mlp])

        # Salvar como artefatos CSV
        Path("models").mkdir(exist_ok=True)
        summary_path = Path("models/model_comparison.csv")
        tradeoff_path = Path("models/cost_tradeoff.csv")
        summary_df.to_csv(summary_path, index=False)
        tradeoff_df.to_csv(tradeoff_path, index=False)

        mlflow.log_artifact(str(summary_path), artifact_path="reports")
        mlflow.log_artifact(str(tradeoff_path), artifact_path="reports")

        # Log métricas finais do melhor modelo no run pai
        best = summary_df.iloc[0]
        mlflow.log_metrics(
            {
                "best_roc_auc": float(best["ROC-AUC"]),
                "best_pr_auc": float(best["PR-AUC"]),
                "best_f1": float(best["F1"]),
            }
        )
        mlflow.log_param("best_model", best["model"])

        logger.info("\n%s", summary_df.to_string(index=False))
        logger.info("\nTrade-off de custo:\n%s", tradeoff_df.to_string(index=False))
        logger.info("Melhor modelo: %s (ROC-AUC=%.4f)", best["model"], best["ROC-AUC"])


if __name__ == "__main__":
    run_experiment()
