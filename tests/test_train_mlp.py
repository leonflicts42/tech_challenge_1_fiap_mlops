"""Testes para src/train_mlp.py — pipeline de re-treinamento automatizado."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

import train_mlp
from train_mlp import (
    MODEL_ALIAS,
    MODEL_REGISTRY_NAME,
    _build_model,
    _build_parser,
    _generate_plots,
    _get_champion_business_value,
    _load_best_params,
    _md5,
    _parse_hidden_dims,
    _register_and_promote,
)


# ── Fixtures compartilhadas ───────────────────────────────────────────────────


@pytest.fixture
def minimal_params() -> dict:
    """Hiperparâmetros mínimos no formato gerado pelo Optuna."""
    return {
        "n_layers": 2,
        "dim_0": 32,
        "dim_1": 16,
        "dropout": 0.1,
        "lr": 1e-3,
        "batch_size": 64,
        "weight_decay": 1e-4,
        "pos_weight": 2.0,
        "patience": 5,
        "use_skip": False,
    }


@pytest.fixture
def params_file(tmp_path: Path, minimal_params: dict) -> Path:
    """Arquivo JSON com estrutura completa do Optuna."""
    data = {
        "MLP": {
            "best_params": minimal_params,
            "cv_best_value": 800000.0,
            "n_trials": 50,
        }
    }
    p = tmp_path / "optuna_best_params.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture
def fake_history():
    """Objeto TrainHistory sintético."""
    return SimpleNamespace(
        train_loss=[0.9, 0.7, 0.5],
        val_loss=[0.95, 0.75, 0.55],
        train_auc=[0.6, 0.75, 0.85],
        val_auc=[0.58, 0.72, 0.82],
        best_epoch=3,
        stopped_early=False,
    )


@pytest.fixture
def synthetic_probas() -> tuple[np.ndarray, np.ndarray]:
    """Labels e probabilidades sintéticos com ambas as classes."""
    rng = np.random.default_rng(42)
    y_true = np.array([0] * 70 + [1] * 30, dtype=np.float32)
    y_proba = rng.uniform(0.0, 1.0, size=100).astype(np.float32)
    y_proba[y_true == 1] = np.clip(y_proba[y_true == 1] + 0.3, 0, 1)
    return y_true, y_proba


@pytest.fixture
def sample_metrics() -> dict:
    return {
        "roc_auc": 0.85,
        "pr_auc": 0.66,
        "recall": 0.99,
        "precision": 0.36,
        "f1": 0.53,
        "tp": 368,
        "tn": 393,
        "fp": 640,
        "fn": 4,
        "business_value": 1_017_420.0,
        "threshold": 0.16,
        "slo_ok": True,
    }


@pytest.fixture
def sample_dataset_meta() -> dict:
    return {
        "train_md5": "abc123",
        "test_md5": "def456",
        "n_train": 5000,
        "n_test": 1000,
        "n_features": 30,
        "churn_rate_train": 0.264,
        "churn_rate_test": 0.265,
    }


# ── _md5 ─────────────────────────────────────────────────────────────────────


class TestMd5:
    def test_hash_correto(self, tmp_path: Path) -> None:
        f = tmp_path / "file.bin"
        f.write_bytes(b"hello world")
        expected = hashlib.md5(b"hello world").hexdigest()
        assert _md5(f) == expected

    def test_hash_diferente_para_conteudos_distintos(self, tmp_path: Path) -> None:
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        a.write_bytes(b"aaa")
        b.write_bytes(b"bbb")
        assert _md5(a) != _md5(b)

    def test_hash_determinístico(self, tmp_path: Path) -> None:
        f = tmp_path / "f.bin"
        f.write_bytes(b"reproducivel")
        assert _md5(f) == _md5(f)


# ── _parse_hidden_dims ────────────────────────────────────────────────────────


class TestParseHiddenDims:
    def test_duas_camadas(self) -> None:
        params = {"n_layers": 2, "dim_0": 128, "dim_1": 64}
        assert _parse_hidden_dims(params) == [128, 64]

    def test_uma_camada(self) -> None:
        params = {"n_layers": 1, "dim_0": 256}
        assert _parse_hidden_dims(params) == [256]

    def test_tres_camadas(self) -> None:
        params = {"n_layers": 3, "dim_0": 64, "dim_1": 32, "dim_2": 16}
        assert _parse_hidden_dims(params) == [64, 32, 16]

    def test_preserva_ordem(self) -> None:
        params = {"n_layers": 3, "dim_0": 32, "dim_1": 128, "dim_2": 64}
        assert _parse_hidden_dims(params) == [32, 128, 64]


# ── _load_best_params ─────────────────────────────────────────────────────────


class TestLoadBestParams:
    def test_carrega_params_validos(
        self, params_file: Path, minimal_params: dict
    ) -> None:
        result = _load_best_params(params_file)
        assert result == minimal_params

    def test_arquivo_inexistente_levanta_filenotfounderror(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(FileNotFoundError, match="Hiperparâmetros não encontrados"):
            _load_best_params(tmp_path / "inexistente.json")

    def test_chave_mlp_ausente_levanta_valueerror(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"RandomForest": {}}), encoding="utf-8")
        with pytest.raises(ValueError, match="MLP.best_params"):
            _load_best_params(p)

    def test_best_params_vazio_levanta_valueerror(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.json"
        p.write_text(json.dumps({"MLP": {"best_params": {}}}), encoding="utf-8")
        with pytest.raises(ValueError, match="MLP.best_params"):
            _load_best_params(p)

    def test_retorna_apenas_best_params_nao_o_wrapper(
        self, params_file: Path, minimal_params: dict
    ) -> None:
        result = _load_best_params(params_file)
        assert "cv_best_value" not in result
        assert "n_trials" not in result


# ── _load_data ────────────────────────────────────────────────────────────────


class TestLoadData:
    def test_arquivo_treino_ausente_levanta_filenotfounderror(
        self, tmp_path: Path
    ) -> None:
        with patch("train_mlp.DATA_PROCESSED", tmp_path):
            with pytest.raises(FileNotFoundError):
                train_mlp._load_data()

    def test_arquivo_teste_ausente_levanta_filenotfounderror(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "train.parquet").write_bytes(b"")
        with patch("train_mlp.DATA_PROCESSED", tmp_path):
            with pytest.raises(FileNotFoundError):
                train_mlp._load_data()

    def test_retorna_dois_dataframes(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"churn_value": [0, 1, 0], "feat": [1.0, 2.0, 3.0]})
        df.to_parquet(tmp_path / "train.parquet")
        df.to_parquet(tmp_path / "test.parquet")
        with patch("train_mlp.DATA_PROCESSED", tmp_path):
            train_df, test_df = train_mlp._load_data()
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)


# ── _build_model ──────────────────────────────────────────────────────────────


class TestBuildModel:
    def test_retorna_churn_mlp_inference(self) -> None:
        from models.mlp import ChurnMLPInference

        model = _build_model([32, 16], dropout=0.1)
        assert isinstance(model, ChurnMLPInference)

    def test_forward_shape_correto(self) -> None:
        model = _build_model([32, 16], dropout=0.0)
        model.eval()
        x = torch.randn(4, 30)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 1)

    def test_tem_parametros_treinaveis(self) -> None:
        model = _build_model([32], dropout=0.1)
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n > 0

    def test_bias_inicializado_em_zero(self) -> None:
        model = _build_model([32], dropout=0.0)
        for module in model.modules():
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                assert torch.all(module.bias == 0.0)

    def test_reproducivel_com_mesma_seed(self) -> None:
        m1 = _build_model([32, 16], dropout=0.1)
        m2 = _build_model([32, 16], dropout=0.1)
        for p1, p2 in zip(m1.parameters(), m2.parameters(), strict=True):
            assert torch.equal(p1, p2)

    def test_hidden_dims_variados_nao_levantam_excecao(self) -> None:
        for dims in [[64], [32, 16], [128, 64, 32]]:
            model = _build_model(dims, dropout=0.2)
            assert model is not None


# ── _generate_plots ───────────────────────────────────────────────────────────


class TestGeneratePlots:
    def test_retorna_cinco_chaves(
        self,
        tmp_path: Path,
        fake_history: object,
        synthetic_probas: tuple,
    ) -> None:
        y_true, y_proba = synthetic_probas
        plot_fns = [
            "train_mlp.save_training_curves",
            "train_mlp.plot_all_roc_curves",
            "train_mlp.plot_all_pr_curves",
            "train_mlp.plot_confusion_matrix_grid",
            "train_mlp.plot_f1_threshold_curves",
        ]
        with patch("train_mlp.REPORTS_FIGURES_MLP", tmp_path):
            with patch.multiple(
                "train_mlp", **{fn.split(".")[-1]: MagicMock() for fn in plot_fns}
            ):
                paths = _generate_plots(fake_history, y_true, y_proba, threshold=0.16)

        assert set(paths.keys()) == {
            "training_curves",
            "roc_curves",
            "pr_curves",
            "confusion_matrix",
            "f1_threshold",
        }

    def test_save_training_curves_chamado_com_historico(
        self,
        tmp_path: Path,
        fake_history: object,
        synthetic_probas: tuple,
    ) -> None:
        y_true, y_proba = synthetic_probas
        with patch("train_mlp.REPORTS_FIGURES_MLP", tmp_path):
            with (
                patch("train_mlp.save_training_curves") as mock_curves,
                patch("train_mlp.plot_all_roc_curves"),
                patch("train_mlp.plot_all_pr_curves"),
                patch("train_mlp.plot_confusion_matrix_grid"),
                patch("train_mlp.plot_f1_threshold_curves"),
            ):
                _generate_plots(fake_history, y_true, y_proba, threshold=0.16)

        mock_curves.assert_called_once()
        kwargs = mock_curves.call_args.kwargs
        assert kwargs["train_losses"] == fake_history.train_loss
        assert kwargs["val_losses"] == fake_history.val_loss
        assert kwargs["best_epoch"] == fake_history.best_epoch

    def test_todas_as_funcoes_de_plot_chamadas(
        self,
        tmp_path: Path,
        fake_history: object,
        synthetic_probas: tuple,
    ) -> None:
        y_true, y_proba = synthetic_probas
        mocks = {
            "save_training_curves": MagicMock(),
            "plot_all_roc_curves": MagicMock(),
            "plot_all_pr_curves": MagicMock(),
            "plot_confusion_matrix_grid": MagicMock(),
            "plot_f1_threshold_curves": MagicMock(),
        }
        with patch("train_mlp.REPORTS_FIGURES_MLP", tmp_path):
            with patch.multiple("train_mlp", **mocks):
                _generate_plots(fake_history, y_true, y_proba, threshold=0.16)

        for name, mock in mocks.items():
            mock.assert_called_once(), f"{name} não foi chamado"

    def test_preds_aplicam_threshold_corretamente(
        self,
        tmp_path: Path,
        fake_history: object,
    ) -> None:
        y_true = np.array([0, 1, 0, 1], dtype=np.float32)
        y_proba = np.array([0.1, 0.9, 0.2, 0.8], dtype=np.float32)
        captured_preds: dict = {}

        def capture_cm(models_preds, y_true, save_path):  # noqa: ARG001
            captured_preds.update(models_preds)

        with patch("train_mlp.REPORTS_FIGURES_MLP", tmp_path):
            with (
                patch("train_mlp.save_training_curves"),
                patch("train_mlp.plot_all_roc_curves"),
                patch("train_mlp.plot_all_pr_curves"),
                patch("train_mlp.plot_confusion_matrix_grid", side_effect=capture_cm),
                patch("train_mlp.plot_f1_threshold_curves"),
            ):
                _generate_plots(fake_history, y_true, y_proba, threshold=0.5)

        np.testing.assert_array_equal(captured_preds["MLP"], np.array([0, 1, 0, 1]))


# ── _get_champion_business_value ──────────────────────────────────────────────


class TestGetChampionBusinessValue:
    def _make_client(self, tag_value: str | None) -> MagicMock:
        client = MagicMock()
        mv = MagicMock()
        mv.tags = {"business_value_test": tag_value} if tag_value is not None else {}
        client.get_model_version_by_alias.return_value = mv
        return client

    def test_retorna_float_quando_tag_presente(self) -> None:
        client = self._make_client("1017420.56")
        result = _get_champion_business_value(client)
        assert result == pytest.approx(1_017_420.56)

    def test_retorna_none_quando_tag_ausente(self) -> None:
        client = self._make_client(None)
        result = _get_champion_business_value(client)
        assert result is None

    def test_retorna_none_quando_alias_nao_existe(self) -> None:
        client = MagicMock()
        client.get_model_version_by_alias.side_effect = Exception("alias not found")
        result = _get_champion_business_value(client)
        assert result is None

    def test_consulta_o_alias_correto(self) -> None:
        client = self._make_client("500.0")
        _get_champion_business_value(client)
        client.get_model_version_by_alias.assert_called_once_with(
            MODEL_REGISTRY_NAME, MODEL_ALIAS
        )


# ── _register_and_promote ─────────────────────────────────────────────────────


class TestRegisterAndPromote:
    def _make_client_and_mv(self, champion_bv: float | None = None) -> tuple:
        client = MagicMock()
        mv = MagicMock()
        mv.version = "3"
        mv.tags = (
            {"business_value_test": str(champion_bv)} if champion_bv is not None else {}
        )
        client.get_model_version_by_alias.return_value = mv
        return client, mv

    def _base_metrics(self, slo_ok: bool = True, bv: float = 1_000_000.0) -> dict:
        return {
            "recall": 0.99,
            "roc_auc": 0.85,
            "pr_auc": 0.66,
            "business_value": bv,
            "threshold": 0.16,
            "slo_ok": slo_ok,
        }

    def _base_dataset_meta(self) -> dict:
        return {
            "train_md5": "aaa",
            "test_md5": "bbb",
            "n_train": 5000,
        }

    def _base_params(self) -> dict:
        return {"n_layers": 2, "dim_0": 128, "dim_1": 64, "dropout": 0.15}

    @patch("train_mlp.mlflow.register_model")
    def test_promove_quando_slo_ok_e_sem_champion_anterior(
        self, mock_register: MagicMock
    ) -> None:
        client, mv = self._make_client_and_mv(champion_bv=None)
        mock_register.return_value = mv
        client.get_model_version_by_alias.side_effect = Exception("no alias")

        version = _register_and_promote(
            client,
            "run_abc",
            self._base_metrics(slo_ok=True, bv=1_000_000.0),
            self._base_dataset_meta(),
            self._base_params(),
        )

        client.set_registered_model_alias.assert_called_once_with(
            MODEL_REGISTRY_NAME, MODEL_ALIAS, mv.version
        )
        assert version == mv.version

    @patch("train_mlp.mlflow.register_model")
    def test_promove_quando_bv_supera_champion(self, mock_register: MagicMock) -> None:
        client, mv = self._make_client_and_mv(champion_bv=900_000.0)
        mock_register.return_value = mv

        _register_and_promote(
            client,
            "run_abc",
            self._base_metrics(slo_ok=True, bv=1_000_000.0),
            self._base_dataset_meta(),
            self._base_params(),
        )

        client.set_registered_model_alias.assert_called_once()

    @patch("train_mlp.mlflow.register_model")
    def test_nao_promove_quando_bv_inferior_ao_champion(
        self, mock_register: MagicMock
    ) -> None:
        client, mv = self._make_client_and_mv(champion_bv=1_500_000.0)
        mock_register.return_value = mv

        _register_and_promote(
            client,
            "run_abc",
            self._base_metrics(slo_ok=True, bv=1_000_000.0),
            self._base_dataset_meta(),
            self._base_params(),
        )

        client.set_registered_model_alias.assert_not_called()

    @patch("train_mlp.mlflow.register_model")
    def test_nao_promove_quando_slo_nao_atendido(
        self, mock_register: MagicMock
    ) -> None:
        client, mv = self._make_client_and_mv(champion_bv=None)
        mock_register.return_value = mv
        client.get_model_version_by_alias.side_effect = Exception("no alias")

        _register_and_promote(
            client,
            "run_abc",
            self._base_metrics(slo_ok=False, bv=2_000_000.0),
            self._base_dataset_meta(),
            self._base_params(),
        )

        client.set_registered_model_alias.assert_not_called()

    @patch("train_mlp.mlflow.register_model")
    def test_tags_obrigatorias_sao_definidas(self, mock_register: MagicMock) -> None:
        client, mv = self._make_client_and_mv(champion_bv=None)
        mock_register.return_value = mv
        client.get_model_version_by_alias.side_effect = Exception("no alias")

        _register_and_promote(
            client,
            "run_abc",
            self._base_metrics(),
            self._base_dataset_meta(),
            self._base_params(),
        )

        tag_keys = {
            call_args.args[2]
            for call_args in client.set_model_version_tag.call_args_list
        }
        for required in (
            "trained_at",
            "recall_test",
            "roc_auc_test",
            "business_value_test",
            "threshold",
            "slo_ok",
        ):
            assert required in tag_keys, f"Tag obrigatória ausente: {required}"

    @patch("train_mlp.mlflow.register_model")
    def test_retorna_versao_registrada(self, mock_register: MagicMock) -> None:
        client, mv = self._make_client_and_mv()
        mock_register.return_value = mv
        mv.version = "7"

        version = _register_and_promote(
            client,
            "run_abc",
            self._base_metrics(),
            self._base_dataset_meta(),
            self._base_params(),
        )

        assert version == "7"


# ── _build_parser ─────────────────────────────────────────────────────────────


class TestBuildParser:
    def test_defaults(self) -> None:
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.params_path == train_mlp.OPTUNA_PARAMS_PATH
        assert args.run_name is None
        assert args.experiment == train_mlp.MLFLOW_EXPERIMENT

    def test_aceita_params_path(self, tmp_path: Path) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--params-path", str(tmp_path / "params.json")])
        assert args.params_path == tmp_path / "params.json"

    def test_aceita_run_name(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--run-name", "drift_retrain"])
        assert args.run_name == "drift_retrain"

    def test_aceita_experiment(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--experiment", "meu-experimento"])
        assert args.experiment == "meu-experimento"


# ── train() — smoke tests com mocks pesados ───────────────────────────────────


def _minimal_train_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {f"feat_{i}": rng.standard_normal(200) for i in range(30)}
    data["churn_value"] = rng.integers(0, 2, size=200).astype(float)
    return pd.DataFrame(data)


@pytest.fixture
def mock_train_dependencies(tmp_path: Path, minimal_params: dict, sample_metrics: dict):
    """Patchea todos os efeitos colaterais pesados de train()."""
    train_df = _minimal_train_df()
    test_df = _minimal_train_df()

    mock_history = SimpleNamespace(
        train_loss=[0.8, 0.6],
        val_loss=[0.85, 0.65],
        train_auc=[0.65, 0.80],
        val_auc=[0.62, 0.78],
        best_epoch=2,
        stopped_early=True,
    )

    mock_trainer = MagicMock()
    mock_trainer.fit.return_value = mock_history
    mock_trainer.predict_proba.return_value = np.full(40, 0.8, dtype=np.float32)

    mock_mv = MagicMock()
    mock_mv.version = "1"
    mock_mv.tags = {}

    mock_run = MagicMock()
    mock_run.__enter__ = MagicMock(return_value=mock_run)
    mock_run.__exit__ = MagicMock(return_value=False)
    mock_run.info.run_id = "test-run-id-abc"

    plot_paths = {
        "training_curves": tmp_path / "tc.png",
        "roc_curves": tmp_path / "roc.png",
        "pr_curves": tmp_path / "pr.png",
        "confusion_matrix": tmp_path / "cm.png",
        "f1_threshold": tmp_path / "f1.png",
    }

    patches = [
        patch("train_mlp._load_data", return_value=(train_df, test_df)),
        patch("train_mlp._load_best_params", return_value=minimal_params),
        patch("train_mlp.setup_mlflow"),
        patch("train_mlp.mlflow.set_experiment"),
        patch("train_mlp.mlflow.end_run"),
        patch("train_mlp.mlflow.start_run", return_value=mock_run),
        patch("train_mlp.mlflow.log_input"),
        patch("train_mlp.mlflow.log_params"),
        patch("train_mlp.mlflow.log_metrics"),
        patch("train_mlp.mlflow.log_artifact"),
        patch("train_mlp.mlflow.pytorch.log_model"),
        patch("train_mlp.mlflow.register_model", return_value=mock_mv),
        patch("train_mlp.mlflow.data.from_pandas", return_value=MagicMock()),
        patch("train_mlp.ChurnTrainer", return_value=mock_trainer),
        patch("train_mlp.find_best_threshold", return_value=(0.16, 900_000.0)),
        patch("train_mlp.full_metrics", return_value=sample_metrics),
        patch("train_mlp._generate_plots", return_value=plot_paths),
        patch("train_mlp.torch.save"),
        patch(
            "train_mlp.MlflowClient",
            return_value=MagicMock(
                get_model_version_by_alias=MagicMock(side_effect=Exception("no alias"))
            ),
        ),
        patch("train_mlp._md5", return_value="mockhash_abc123"),
        patch("train_mlp.DATA_PROCESSED", tmp_path),
        patch("train_mlp.MODELS_DIR", tmp_path),
        patch("train_mlp.REPORTS_JSON_DIR", tmp_path),
        patch("train_mlp.OPTUNA_PARAMS_PATH", tmp_path / "optuna_best_params.json"),
    ]

    # Cria o arquivo de parâmetros no tmp_path para o winner_report
    (tmp_path / "optuna_best_params.json").write_text(
        json.dumps({"MLP": {"best_params": minimal_params}}), encoding="utf-8"
    )

    for p in patches:
        p.start()

    yield {"history": mock_history, "trainer": mock_trainer, "mv": mock_mv}

    for p in patches:
        p.stop()


class TestTrain:
    def test_smoke_retorna_dict_com_chaves_obrigatorias(
        self, mock_train_dependencies: dict
    ) -> None:
        result = train_mlp.train(run_name="test_run")
        assert set(result.keys()) == {
            "run_id",
            "model_version",
            "metrics",
            "dataset_meta",
            "plot_paths",
            "model_promoted",
        }

    def test_run_id_preenchido(self, mock_train_dependencies: dict) -> None:
        result = train_mlp.train(run_name="test_run")
        assert result["run_id"] == "test-run-id-abc"

    def test_model_promoted_true_quando_slo_ok_sem_champion(
        self, mock_train_dependencies: dict
    ) -> None:
        result = train_mlp.train(run_name="test_run")
        assert result["model_promoted"] is True

    def test_model_promoted_false_quando_slo_nao_atendido(
        self, mock_train_dependencies: dict, sample_metrics: dict
    ) -> None:
        failing_metrics = {**sample_metrics, "slo_ok": False}
        with patch("train_mlp.full_metrics", return_value=failing_metrics):
            result = train_mlp.train(run_name="test_run")
        assert result["model_promoted"] is False

    def test_trainer_fit_chamado_com_arrays_numpy(
        self, mock_train_dependencies: dict
    ) -> None:
        train_mlp.train(run_name="test_run")
        mock_trainer = mock_train_dependencies["trainer"]
        mock_trainer.fit.assert_called_once()
        X_tr, y_tr, X_vl, y_vl = mock_trainer.fit.call_args.args
        assert isinstance(X_tr, np.ndarray)
        assert isinstance(y_tr, np.ndarray)

    def test_generate_plots_chamado(self, mock_train_dependencies: dict) -> None:
        with patch("train_mlp._generate_plots", return_value={}) as mock_plots:
            train_mlp.train(run_name="test_run")
        mock_plots.assert_called_once()

    def test_winner_report_json_escrito(
        self, mock_train_dependencies: dict, tmp_path: Path
    ) -> None:
        train_mlp.train(run_name="test_run")
        report_path = tmp_path / "winner_model_report.json"
        assert report_path.exists()
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
        assert report["model"] == "MLP"
        assert "test_metrics" in report
        assert "dataset" in report

    def test_run_name_default_contem_timestamp(
        self, mock_train_dependencies: dict
    ) -> None:
        with patch("train_mlp.mlflow.start_run") as mock_start:
            mock_run = MagicMock()
            mock_run.__enter__ = MagicMock(return_value=mock_run)
            mock_run.__exit__ = MagicMock(return_value=False)
            mock_run.info.run_id = "abc"
            mock_start.return_value = mock_run
            train_mlp.train()  # sem run_name → gera timestamp
        call_kwargs = mock_start.call_args.kwargs
        assert "retrain_" in call_kwargs.get("run_name", "")


# ── main() ────────────────────────────────────────────────────────────────────


class TestMain:
    def test_exit_0_quando_slo_ok(self, mock_train_dependencies: dict) -> None:
        with patch("sys.argv", ["train_mlp"]):
            with pytest.raises(SystemExit) as exc:
                train_mlp.main()
        assert exc.value.code == 0

    def test_exit_1_quando_slo_falha(
        self, mock_train_dependencies: dict, sample_metrics: dict
    ) -> None:
        failing = {**sample_metrics, "slo_ok": False}
        with patch("train_mlp.full_metrics", return_value=failing):
            with patch("sys.argv", ["train_mlp"]):
                with pytest.raises(SystemExit) as exc:
                    train_mlp.main()
        assert exc.value.code == 1
