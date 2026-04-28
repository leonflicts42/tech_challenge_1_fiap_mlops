"""
src/train_mlp.py — Re-treinamento automatizado do MLP ChurnMLPv2.

Lê os melhores hiperparâmetros registrados pelo Optuna
(reports/json/optuna_best_params.json), treina o modelo nos dados processados,
registra tudo no MLflow e promove a versão a 'champion' se superar o incumbente
em valor de negócio.

Uso:
    uv run python src/train_mlp.py
    uv run python src/train_mlp.py --params-path reports/json/optuna_best_params.json
    uv run python src/train_mlp.py --run-name retrain_drift_2025_01_15

Gatilhos automáticos (ver seção de deploy em docs/deploy_architecture.md):
    • Detecção de data drift (PSI ou KS acima de limiar no /monitor endpoint)
    • Detecção de model drift (Recall mensal < SLO_RECALL_MIN no plano de monitoramento)
    • Agendamento periódico (cron / Airflow DAG / GitHub Actions schedule)
    • Push de novos dados processados no repositório (CI/CD workflow_dispatch)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split

from config import (
    DATA_PROCESSED,
    DEVICE,
    MLFLOW_EXPERIMENT,
    MODELS_DIR,
    N_FEATURES_FINAL,
    RANDOM_STATE,
    REPORTS_FIGURES_MLP,
    REPORTS_JSON_DIR,
    SLO_RECALL_MIN,
    TARGET_COL,
    TEST_SIZE,
    get_logger,
    setup_mlflow,
)
from models.mlp import ChurnMLPInference
from models.trainer import ChurnTrainer, TrainerConfig
from utils.business import find_best_threshold, full_metrics
from utils.plots import (
    plot_all_pr_curves,
    plot_all_roc_curves,
    plot_confusion_matrix_grid,
    plot_f1_threshold_curves,
    save_training_curves,
)

logger = get_logger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────

OPTUNA_PARAMS_PATH: Path = REPORTS_JSON_DIR / "optuna_best_params.json"
MODEL_REGISTRY_NAME: str = "ChurnMLP"
MODEL_ALIAS: str = "champion"
# Mais épocas no treino final do que no CV — o early stopping controla o limite real
_EPOCHS_RETRAIN: int = 200


# ── Utilitários de dados ──────────────────────────────────────────────────────


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = DATA_PROCESSED / "train.parquet"
    test_path = DATA_PROCESSED / "test.parquet"

    for p in (train_path, test_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {p}\n"
                "Execute o notebook 2_vab_preprocessing antes de treinar."
            )

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    logger.info(
        "Dados carregados | treino=%s | teste=%s | churn_train=%.2f%% | churn_test=%.2f%%",
        train_df.shape,
        test_df.shape,
        train_df[TARGET_COL].mean() * 100,
        test_df[TARGET_COL].mean() * 100,
    )
    return train_df, test_df


# ── Utilitários de hiperparâmetros ────────────────────────────────────────────


def _load_best_params(params_path: Path) -> dict:
    if not params_path.exists():
        raise FileNotFoundError(
            f"Hiperparâmetros não encontrados: {params_path}\n"
            "Execute o notebook 4_vab_mlp_vs_baselines para gerar este arquivo."
        )

    with open(params_path, encoding="utf-8") as f:
        all_params = json.load(f)

    mlp_entry = all_params.get("MLP", {})
    best_params = mlp_entry.get("best_params", {})

    if not best_params:
        raise ValueError(f"Chave 'MLP.best_params' não encontrada em {params_path}.")

    logger.info(
        "Hiperparâmetros carregados | cv_best_value=%.0f | n_trials=%d | params=%s",
        mlp_entry.get("cv_best_value", 0.0),
        mlp_entry.get("n_trials", 0),
        json.dumps(best_params, default=str),
    )
    return best_params


def _parse_hidden_dims(params: dict) -> list[int]:
    """Reconstrói hidden_dims a partir do formato do Optuna (n_layers + dim_i)."""
    n_layers = int(params["n_layers"])
    return [int(params[f"dim_{i}"]) for i in range(n_layers)]


# ── Modelo compatível com a API ───────────────────────────────────────────────


def _build_model(hidden_dims: list[int], dropout: float) -> ChurnMLPInference:
    """Constrói ChurnMLPInference com init Kaiming-Normal e seed fixada.

    ChurnMLPInference é usado (não ChurnMLPv2) porque seu state_dict é
    diretamente compatível com ChurnPredictor._load_model() — mesma estrutura
    de chaves que o notebook 4 produziu (hidden.0, hidden.1, ..., output).
    """
    torch.manual_seed(RANDOM_STATE)

    model = ChurnMLPInference(
        input_dim=N_FEATURES_FINAL,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )

    # Kaiming-Normal para camadas Linear com ReLU
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Modelo construído | ChurnMLPInference | hidden_dims=%s | dropout=%.2f | params=%d",
        hidden_dims,
        dropout,
        n_params,
    )
    return model


# ── Geração de plots ──────────────────────────────────────────────────────────


def _generate_plots(
    history: object,
    y_test: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict[str, Path]:
    """Gera e salva todos os gráficos de desempenho em reports/figures/mlp/."""
    REPORTS_FIGURES_MLP.mkdir(parents=True, exist_ok=True)

    proba_dict = {"MLP": y_proba}
    preds_dict = {"MLP": (y_proba >= threshold).astype(int)}
    paths: dict[str, Path] = {}

    # Curvas de treinamento (loss + AUC por época)
    curves_path = REPORTS_FIGURES_MLP / "training_curves.png"
    save_training_curves(
        train_losses=history.train_loss,
        val_losses=history.val_loss,
        output_path=curves_path,
        train_aucs=history.train_auc,
        val_aucs=history.val_auc,
        best_epoch=history.best_epoch,
    )
    paths["training_curves"] = curves_path

    # Curva ROC
    roc_path = REPORTS_FIGURES_MLP / "comparativo_roc_curves.png"
    plot_all_roc_curves(
        models_probas=proba_dict,
        y_true=y_test,
        save_path=roc_path,
    )
    paths["roc_curves"] = roc_path

    # Curva Precision-Recall
    pr_path = REPORTS_FIGURES_MLP / "comparativo_pr_curves.png"
    plot_all_pr_curves(
        models_probas=proba_dict,
        y_true=y_test,
        save_path=pr_path,
    )
    paths["pr_curves"] = pr_path

    # Matriz de confusão
    cm_path = REPORTS_FIGURES_MLP / "comparativo_confusion_matrices.png"
    plot_confusion_matrix_grid(
        models_preds=preds_dict,
        y_true=y_test,
        save_path=cm_path,
    )
    paths["confusion_matrix"] = cm_path

    # F1 vs. threshold
    f1_path = REPORTS_FIGURES_MLP / "comparativo_f1_threshold.png"
    plot_f1_threshold_curves(
        models_probas=proba_dict,
        y_true=y_test,
        save_path=f1_path,
    )
    paths["f1_threshold"] = f1_path

    logger.info(
        "Plots gerados | dir=%s | arquivos=%s",
        REPORTS_FIGURES_MLP,
        [p.name for p in paths.values()],
    )
    return paths


# ── MLflow: registro e promoção ───────────────────────────────────────────────


def _get_champion_business_value(client: MlflowClient) -> float | None:
    """Retorna o business_value do champion atual, ou None se não existir."""
    try:
        mv = client.get_model_version_by_alias(MODEL_REGISTRY_NAME, MODEL_ALIAS)
        tag = mv.tags.get("business_value_test")
        return float(tag) if tag else None
    except Exception:
        return None


def _register_and_promote(
    client: MlflowClient,
    run_id: str,
    metrics: dict,
    dataset_meta: dict,
    best_params: dict,
) -> str:
    """Registra o modelo e promove a 'champion' se superar o incumbente."""
    model_uri = f"runs:/{run_id}/model_artifact"
    mv = mlflow.register_model(model_uri=model_uri, name=MODEL_REGISTRY_NAME)

    tags = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset_md5_train": dataset_meta["train_md5"],
        "dataset_md5_test": dataset_meta["test_md5"],
        "n_train": str(dataset_meta["n_train"]),
        "recall_test": f"{metrics['recall']:.4f}",
        "roc_auc_test": f"{metrics['roc_auc']:.4f}",
        "pr_auc_test": f"{metrics['pr_auc']:.4f}",
        "business_value_test": f"{metrics['business_value']:.2f}",
        "threshold": f"{metrics['threshold']:.4f}",
        "slo_ok": str(metrics["slo_ok"]),
        "hidden_dims": str(_parse_hidden_dims(best_params)),
        "dropout": str(best_params.get("dropout", "")),
    }
    for k, v in tags.items():
        client.set_model_version_tag(MODEL_REGISTRY_NAME, mv.version, k, v)

    current_bv = _get_champion_business_value(client)
    new_bv = metrics["business_value"]
    slo_ok = metrics["slo_ok"]

    if slo_ok and (current_bv is None or new_bv > current_bv):
        client.set_registered_model_alias(MODEL_REGISTRY_NAME, MODEL_ALIAS, mv.version)
        client.set_model_version_tag(
            MODEL_REGISTRY_NAME, mv.version, "stage", MODEL_ALIAS
        )
        logger.info(
            "Novo champion promovido | version=%s | bv_novo=%.0f | bv_anterior=%s",
            mv.version,
            new_bv,
            f"{current_bv:.0f}" if current_bv is not None else "nenhum",
        )
    else:
        reason = (
            "SLO não atendido"
            if not slo_ok
            else f"bv={new_bv:.0f} não supera champion={current_bv:.0f}"
        )
        logger.info(
            "Modelo registrado como challenger (não promovido) | reason=%s | version=%s",
            reason,
            mv.version,
        )

    logger.info(
        "Registro concluído | name=%s | version=%s | alias=%s",
        MODEL_REGISTRY_NAME,
        mv.version,
        MODEL_ALIAS
        if (slo_ok and (current_bv is None or new_bv > current_bv))
        else "challenger",
    )
    return mv.version


# ── Pipeline principal ────────────────────────────────────────────────────────


def train(
    params_path: Path | str | None = None,
    run_name: str | None = None,
    experiment_name: str | None = None,
) -> dict:
    """Executa o pipeline completo de re-treinamento do MLP.

    Pode ser chamado diretamente por código de detecção de drift, agendadores
    ou pela CLI. Retorna um dict com métricas, versão registrada e paths dos
    artefatos para que o chamador possa tomar decisões downstream.

    Args:
        params_path:     caminho para optuna_best_params.json
                         (default: reports/json/optuna_best_params.json)
        run_name:        nome do run no MLflow
                         (default: retrain_<timestamp UTC>)
        experiment_name: nome do experimento MLflow
                         (default: MLFLOW_EXPERIMENT de config.py)

    Returns:
        {
          "run_id":          str,
          "model_version":   str,
          "metrics":         dict,   # métricas completas no conjunto de teste
          "dataset_meta":    dict,   # MD5 e tamanhos dos datasets
          "plot_paths":      dict,   # paths dos PNGs gerados
          "model_promoted":  bool,   # True se virou novo champion
        }
    """
    params_path = Path(params_path) if params_path else OPTUNA_PARAMS_PATH
    experiment_name = experiment_name or MLFLOW_EXPERIMENT
    run_name = (
        run_name or f"retrain_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    )

    logger.info("=" * 60)
    logger.info("Re-treinamento iniciado | run_name=%s | device=%s", run_name, DEVICE)
    logger.info("=" * 60)

    # ── [1] Configurar MLflow ─────────────────────────────────────────────────
    setup_mlflow()
    mlflow.set_experiment(experiment_name)

    # ── [2] Carregar e inspecionar dados ──────────────────────────────────────
    train_df, test_df = _load_data()

    train_path = DATA_PROCESSED / "train.parquet"
    test_path = DATA_PROCESSED / "test.parquet"

    dataset_meta: dict = {
        "train_md5": _md5(train_path),
        "test_md5": _md5(test_path),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_features": N_FEATURES_FINAL,
        "churn_rate_train": float(train_df[TARGET_COL].mean()),
        "churn_rate_test": float(test_df[TARGET_COL].mean()),
    }

    feature_cols = [c for c in train_df.columns if c != TARGET_COL]
    X_dev = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_dev = train_df[TARGET_COL].to_numpy(dtype=np.float32)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_test = test_df[TARGET_COL].to_numpy(dtype=np.float32)

    # ── [3] Carregar hiperparâmetros do Optuna ────────────────────────────────
    best_params = _load_best_params(params_path)
    hidden_dims = _parse_hidden_dims(best_params)
    dropout = float(best_params["dropout"])
    lr = float(best_params["lr"])
    batch_size = int(best_params["batch_size"])
    weight_decay = float(best_params["weight_decay"])
    pos_weight = float(best_params["pos_weight"])
    patience = int(best_params["patience"])

    # ── [4] Dividir dev em treino/validação ───────────────────────────────────
    X_tr, X_vl, y_tr, y_vl = train_test_split(
        X_dev,
        y_dev,
        test_size=TEST_SIZE,
        stratify=y_dev,
        random_state=RANDOM_STATE,
    )
    logger.info(
        "Split | treino=%d | validação=%d | teste=%d | SLO_recall>=%.2f",
        len(X_tr),
        len(X_vl),
        len(X_test),
        SLO_RECALL_MIN,
    )

    # ── [5] Abrir run MLflow e executar pipeline ───────────────────────────────
    mlflow.end_run()
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info("MLflow run aberto | run_id=%s", run_id)

        # ── [5a] Registrar datasets ───────────────────────────────────────────
        train_ds = mlflow.data.from_pandas(
            train_df,
            source=str(train_path),
            name="telco_churn_train",
            targets=TARGET_COL,
        )
        test_ds = mlflow.data.from_pandas(
            test_df,
            source=str(test_path),
            name="telco_churn_test",
            targets=TARGET_COL,
        )
        mlflow.log_input(train_ds, context="training")
        mlflow.log_input(test_ds, context="evaluation")

        mlflow.log_params(
            {
                # Dataset
                "dataset_train_md5": dataset_meta["train_md5"],
                "dataset_test_md5": dataset_meta["test_md5"],
                "n_train": dataset_meta["n_train"],
                "n_test": dataset_meta["n_test"],
                "n_features": dataset_meta["n_features"],
                "churn_rate_train": round(dataset_meta["churn_rate_train"], 4),
                "churn_rate_test": round(dataset_meta["churn_rate_test"], 4),
                # Arquitetura
                "hidden_dims": str(hidden_dims),
                "dropout": dropout,
                "use_skip": best_params.get("use_skip", False),
                # Treinamento
                "lr": lr,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "pos_weight": round(pos_weight, 4),
                "patience": patience,
                "epochs_max": _EPOCHS_RETRAIN,
                "random_state": RANDOM_STATE,
                "device": DEVICE,
            }
        )

        # ── [5b] Construir modelo e treinar ───────────────────────────────────
        model = _build_model(hidden_dims, dropout)

        trainer_cfg = TrainerConfig(
            lr=lr,
            epochs=_EPOCHS_RETRAIN,
            batch_size=batch_size,
            patience=patience,
            weight_decay=weight_decay,
            pos_weight=pos_weight,
            device=DEVICE,
            seed=RANDOM_STATE,
        )

        logger.info(
            "Treinamento iniciado | epochs_max=%d | patience=%d | batch=%d",
            _EPOCHS_RETRAIN,
            patience,
            batch_size,
        )
        trainer = ChurnTrainer(model, trainer_cfg)
        history = trainer.fit(X_tr, y_tr, X_vl, y_vl)

        mlflow.log_metrics(
            {
                "best_epoch": history.best_epoch,
                "stopped_early": int(history.stopped_early),
                "final_val_loss": history.val_loss[-1]
                if history.val_loss
                else float("nan"),
                "final_val_auc": history.val_auc[-1]
                if history.val_auc
                else float("nan"),
            }
        )

        logger.info(
            "Treinamento concluído | best_epoch=%d | stopped_early=%s | val_auc=%.4f",
            history.best_epoch,
            history.stopped_early,
            history.val_auc[-1] if history.val_auc else float("nan"),
        )

        # ── [5c] Avaliar no conjunto de teste ─────────────────────────────────
        y_proba = trainer.predict_proba(X_test)
        best_threshold, _ = find_best_threshold(y_test, y_proba)
        metrics = full_metrics(y_test, y_proba, best_threshold)

        mlflow.log_metrics(
            {
                "test_roc_auc": metrics["roc_auc"],
                "test_pr_auc": metrics["pr_auc"],
                "test_recall": metrics["recall"],
                "test_precision": metrics["precision"],
                "test_f1": metrics["f1"],
                "test_fn": float(metrics["fn"]),
                "test_fp": float(metrics["fp"]),
                "test_tp": float(metrics["tp"]),
                "test_tn": float(metrics["tn"]),
                "test_business_value": metrics["business_value"],
                "test_threshold": metrics["threshold"],
                "test_slo_ok": float(metrics["slo_ok"]),
            }
        )

        logger.info(
            "Avaliação no teste | ROC-AUC=%.4f | PR-AUC=%.4f | Recall=%.4f"
            " | BV=%.0f | SLO_ok=%s | threshold=%.2f",
            metrics["roc_auc"],
            metrics["pr_auc"],
            metrics["recall"],
            metrics["business_value"],
            metrics["slo_ok"],
            metrics["threshold"],
        )

        # ── [5d] Gerar e logar plots ──────────────────────────────────────────
        plot_paths = _generate_plots(history, y_test, y_proba, best_threshold)
        for label, path in plot_paths.items():
            mlflow.log_artifact(str(path), artifact_path="plots")

        # ── [5e] Salvar artefatos do modelo ───────────────────────────────────
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        pt_path = MODELS_DIR / "best_model_mlp.pt"
        torch.save(model.state_dict(), pt_path)

        # Loga o state_dict como artefato bruto (para auditoria)
        mlflow.log_artifact(str(pt_path), artifact_path="model_artifact")
        # Loga o modelo completo no formato MLflow (para mlflow.pytorch.load_model)
        mlflow.pytorch.log_model(model, name="model_mlflow")

        # Relatório do modelo vencedor (formato compatível com predictor.py)
        winner_report = {
            "model": "MLP",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "best_params": best_params,
            "test_metrics": metrics,
            "cost_function": {
                "TP": float(2845.0),
                "TN": float(73.52),
                "FN": float(-2845.0),
                "FP": float(-73.52),
            },
            "dataset": dataset_meta,
        }
        winner_path = REPORTS_JSON_DIR / "winner_model_report.json"
        with open(winner_path, "w", encoding="utf-8") as f:
            json.dump(winner_report, f, indent=2, ensure_ascii=False, default=str)
        mlflow.log_artifact(str(winner_path), artifact_path="reports")
        mlflow.log_artifact(str(params_path), artifact_path="reports")

        logger.info("Artefatos salvos | model=%s | report=%s", pt_path, winner_path)

        # ── [5f] Registrar no MLflow Model Registry ───────────────────────────
        client = MlflowClient()
        prev_champion_bv = _get_champion_business_value(client)
        version = _register_and_promote(
            client, run_id, metrics, dataset_meta, best_params
        )
        model_promoted = metrics["slo_ok"] and (
            prev_champion_bv is None or metrics["business_value"] > prev_champion_bv
        )

    logger.info("=" * 60)
    logger.info(
        "Re-treinamento finalizado | run_id=%s | version=%s | promoted=%s",
        run_id,
        version,
        model_promoted,
    )
    logger.info("=" * 60)

    return {
        "run_id": run_id,
        "model_version": version,
        "metrics": metrics,
        "dataset_meta": dataset_meta,
        "plot_paths": {k: str(v) for k, v in plot_paths.items()},
        "model_promoted": model_promoted,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_mlp",
        description="Re-treina o MLP de churn com os melhores hiperparâmetros do Optuna.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--params-path",
        type=Path,
        default=OPTUNA_PARAMS_PATH,
        help="Caminho para optuna_best_params.json",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Nome do run no MLflow (default: retrain_<timestamp>)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=MLFLOW_EXPERIMENT,
        help="Nome do experimento MLflow",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    result = train(
        params_path=args.params_path,
        run_name=args.run_name,
        experiment_name=args.experiment,
    )

    print("\n── Resultado do re-treinamento ────────────────────────")  # noqa: T201
    print(f"  run_id:         {result['run_id']}")  # noqa: T201
    print(f"  model_version:  {result['model_version']}")  # noqa: T201
    print(f"  promoted:       {result['model_promoted']}")  # noqa: T201
    print(f"  ROC-AUC:        {result['metrics']['roc_auc']:.4f}")  # noqa: T201
    print(f"  Recall:         {result['metrics']['recall']:.4f}")  # noqa: T201
    print(f"  Business Value: {result['metrics']['business_value']:,.0f}")  # noqa: T201
    print(f"  SLO ok:         {result['metrics']['slo_ok']}")  # noqa: T201
    print("───────────────────────────────────────────────────────\n")  # noqa: T201

    sys.exit(0 if result["metrics"]["slo_ok"] else 1)


if __name__ == "__main__":
    main()
