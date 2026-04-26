"""Testes para ChurnPredictor — api/predictor.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from api.predictor import ChurnPredictor, _patched_check_unknown
from api.schemas import ChurnRequest


# ── Fixture: predictor com internals mockados ─────────────────────────────────

@pytest.fixture
def mock_predictor(valid_payload: dict) -> ChurnPredictor:
    """ChurnPredictor com _load_* mockados para não precisar de arquivos reais."""
    with (
        patch.object(ChurnPredictor, "_load_preprocessor") as p,
        patch.object(ChurnPredictor, "_load_model") as m,
        patch.object(ChurnPredictor, "_load_threshold") as t,
    ):
        p.return_value = MagicMock()
        m.return_value = MagicMock()
        t.return_value = 0.5

        pred = ChurnPredictor(
            preprocessor_path=Path("fake.pkl"),
            model_path=Path("fake.pt"),
        )

    # configura comportamento realista
    pred._preprocessor = MagicMock()
    pred._preprocessor.transform.return_value = np.zeros((1, 30), dtype=np.float64)

    pred._model = MagicMock()
    pred._model.return_value = torch.tensor([[0.8]])
    pred._threshold = 0.5
    pred.model_version = "test-v1"

    return pred


@pytest.fixture
def churn_request(valid_payload: dict) -> ChurnRequest:
    return ChurnRequest(**valid_payload)


# ── Testes do pipeline predict() ──────────────────────────────────────────────

class TestChurnPredictorPredict:
    def test_predict_retorna_churn_response(
        self, mock_predictor: ChurnPredictor, churn_request: ChurnRequest
    ) -> None:
        from api.schemas import ChurnResponse
        result = mock_predictor.predict(churn_request)
        assert isinstance(result, ChurnResponse)

    def test_predict_churn_probability_no_range(
        self, mock_predictor: ChurnPredictor, churn_request: ChurnRequest
    ) -> None:
        result = mock_predictor.predict(churn_request)
        assert 0.0 <= result.churn_probability <= 1.0

    def test_predict_label_churn_quando_prob_alta(
        self, mock_predictor: ChurnPredictor, churn_request: ChurnRequest
    ) -> None:
        mock_predictor._model.return_value = torch.tensor([[3.0]])  # sigmoid ≈ 0.95
        mock_predictor._threshold = 0.5
        result = mock_predictor.predict(churn_request)
        assert result.churn_label == "churn"

    def test_predict_label_no_churn_quando_prob_baixa(
        self, mock_predictor: ChurnPredictor, churn_request: ChurnRequest
    ) -> None:
        mock_predictor._model.return_value = torch.tensor([[-3.0]])  # sigmoid ≈ 0.05
        mock_predictor._threshold = 0.5
        result = mock_predictor.predict(churn_request)
        assert result.churn_label == "no_churn"

    def test_predict_cost_churn_e_positivo(
        self, mock_predictor: ChurnPredictor, churn_request: ChurnRequest
    ) -> None:
        mock_predictor._model.return_value = torch.tensor([[3.0]])
        result = mock_predictor.predict(churn_request)
        assert result.cost_estimate_brl > 0.0

    def test_predict_cost_no_churn_e_zero(
        self, mock_predictor: ChurnPredictor, churn_request: ChurnRequest
    ) -> None:
        mock_predictor._model.return_value = torch.tensor([[-3.0]])
        result = mock_predictor.predict(churn_request)
        assert result.cost_estimate_brl == 0.0

    def test_predict_threshold_no_response(
        self, mock_predictor: ChurnPredictor, churn_request: ChurnRequest
    ) -> None:
        mock_predictor._threshold = 0.16
        result = mock_predictor.predict(churn_request)
        assert result.threshold_used == 0.16

    def test_predict_model_version_no_response(
        self, mock_predictor: ChurnPredictor, churn_request: ChurnRequest
    ) -> None:
        result = mock_predictor.predict(churn_request)
        assert result.model_version == "test-v1"

    def test_predict_preprocessor_chamado_uma_vez(
        self, mock_predictor: ChurnPredictor, churn_request: ChurnRequest
    ) -> None:
        mock_predictor.predict(churn_request)
        mock_predictor._preprocessor.transform.assert_called_once()

    def test_predict_shape_errado_levanta_value_error(
        self, mock_predictor: ChurnPredictor, churn_request: ChurnRequest
    ) -> None:
        mock_predictor._preprocessor.transform.return_value = np.zeros((1, 15))
        with pytest.raises(ValueError, match="Shape inesperado"):
            mock_predictor.predict(churn_request)

    def test_predict_sparse_matrix_e_convertida(
        self, mock_predictor: ChurnPredictor, churn_request: ChurnRequest
    ) -> None:
        from scipy.sparse import csr_matrix
        mock_predictor._preprocessor.transform.return_value = csr_matrix(
            np.zeros((1, 30))
        )
        result = mock_predictor.predict(churn_request)
        assert result is not None

    def test_predict_multiple_lines_no_phone_service_normalizado(
        self, mock_predictor: ChurnPredictor, valid_payload: dict
    ) -> None:
        valid_payload["multiple_lines"] = "No phone service"
        req = ChurnRequest(**valid_payload)
        result = mock_predictor.predict(req)
        assert result is not None


# ── Testes do carregamento de threshold ──────────────────────────────────────

class TestLoadThreshold:
    def test_threshold_default_quando_sem_arquivos(self, mock_predictor: ChurnPredictor) -> None:
        with (
            patch("api.predictor.MODELS_DIR") as mock_dir,
            patch("api.predictor.PROJECT_ROOT") as mock_root,
        ):
            mock_dir.glob.return_value = []
            mock_root.__truediv__ = lambda self, other: Path("/nao/existe") / other
            result = mock_predictor._load_threshold(0.42)
        assert result == 0.42

    def test_threshold_carregado_de_optuna(self, tmp_path: Path, mock_predictor: ChurnPredictor) -> None:
        optuna_file = tmp_path / "optuna_best_params_01.json"
        optuna_file.write_text(json.dumps({"threshold": 0.23}))

        with patch("api.predictor.MODELS_DIR", tmp_path):
            result = mock_predictor._load_threshold(0.5)
        assert result == pytest.approx(0.23)

    def test_threshold_invalido_em_optuna_usa_default(
        self, tmp_path: Path, mock_predictor: ChurnPredictor
    ) -> None:
        optuna_file = tmp_path / "optuna_best_params_bad.json"
        optuna_file.write_text("not valid json {{{{")

        fake_root = tmp_path / "project"
        fake_root.mkdir()

        with (
            patch("api.predictor.MODELS_DIR", tmp_path),
            patch("api.predictor.PROJECT_ROOT", fake_root),
        ):
            result = mock_predictor._load_threshold(0.5)
        assert result == 0.5


# ── Testes do FileNotFoundError ───────────────────────────────────────────────

class TestChurnPredictorFileNotFound:
    def test_preprocessor_nao_encontrado_levanta_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            ChurnPredictor(
                preprocessor_path=Path("/nao/existe/preprocessor.pkl"),
                model_path=Path("/nao/existe/model.pt"),
            )

    def test_model_nao_encontrado_levanta_file_not_found(self, tmp_path: Path) -> None:
        fake_pkl = tmp_path / "preprocessor.pkl"
        import joblib
        import sklearn.preprocessing
        joblib.dump(sklearn.preprocessing.StandardScaler(), fake_pkl)

        with pytest.raises(FileNotFoundError):
            ChurnPredictor(
                preprocessor_path=fake_pkl,
                model_path=Path("/nao/existe/model.pt"),
            )


# ── Testes do monkey-patch sklearn ───────────────────────────────────────────

class TestPatchedCheckUnknown:
    def test_retorna_diff_quando_type_error(self) -> None:
        values = np.array(["a", "b", "c"])
        known = np.array(["a", "b"])
        result = _patched_check_unknown(values, known)
        assert "c" in result

    def test_retorna_mask_quando_return_mask_true(self) -> None:
        values = np.array(["a", "b", "c"])
        known = np.array(["a", "b"])
        diff, mask = _patched_check_unknown(values, known, return_mask=True)
        assert "c" in diff

    def test_delega_para_original_sem_type_error(self) -> None:
        values = np.array([1.0, 2.0, 3.0])
        known = np.array([1.0, 2.0])
        result = _patched_check_unknown(values, known)
        assert 3.0 in result
