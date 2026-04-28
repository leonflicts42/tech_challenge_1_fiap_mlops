"""Testes dos endpoints HTTP da API."""

from __future__ import annotations


from fastapi.testclient import TestClient

from main import app


class TestHealthEndpoint:
    def test_health_200_com_mocks(self, test_client: TestClient) -> None:
        r = test_client.get("/api/v1/health")
        assert r.status_code == 200

    def test_health_retorna_ok(self, test_client: TestClient) -> None:
        assert test_client.get("/api/v1/health").json()["status"] == "ok"

    def test_health_503_sem_predictor(self, test_client: TestClient) -> None:
        app.state.predictor = None
        r = test_client.get("/api/v1/health")
        assert r.status_code == 503

    def test_health_model_loaded_false_quando_model_none(
        self, test_client: TestClient
    ) -> None:
        app.state.predictor._model = None
        r = test_client.get("/api/v1/health")
        assert r.status_code == 200
        assert r.json()["model_loaded"] is False

    def test_health_preprocessor_loaded_false_quando_preprocessor_none(
        self, test_client: TestClient
    ) -> None:
        app.state.predictor._preprocessor = None
        r = test_client.get("/api/v1/health")
        assert r.status_code == 200
        assert r.json()["preprocessor_loaded"] is False


class TestPredictEndpoint:
    def test_payload_valido_retorna_200(
        self, test_client: TestClient, valid_payload: dict
    ) -> None:
        r = test_client.post("/api/v1/predict", json=valid_payload)
        assert r.status_code == 200

    def test_schema_de_resposta_correto(
        self, test_client: TestClient, valid_payload: dict
    ) -> None:
        body = test_client.post("/api/v1/predict", json=valid_payload).json()
        assert "churn_probability" in body
        assert "churn_label" in body
        assert "threshold_used" in body
        assert "cost_estimate_brl" in body
        assert "model_version" in body

    def test_churn_label_valido(
        self, test_client: TestClient, valid_payload: dict
    ) -> None:
        label = test_client.post("/api/v1/predict", json=valid_payload).json()[
            "churn_label"
        ]
        assert label in ("churn", "no_churn")

    def test_gender_invalido_retorna_422(
        self, test_client: TestClient, valid_payload: dict
    ) -> None:
        valid_payload["gender"] = "Other"
        assert (
            test_client.post("/api/v1/predict", json=valid_payload).status_code == 422
        )

    def test_campo_faltando_retorna_422(
        self, test_client: TestClient, valid_payload: dict
    ) -> None:
        del valid_payload["tenure_months"]
        assert (
            test_client.post("/api/v1/predict", json=valid_payload).status_code == 422
        )

    def test_tenure_acima_de_72_retorna_422(
        self, test_client: TestClient, valid_payload: dict
    ) -> None:
        valid_payload["tenure_months"] = 99
        assert (
            test_client.post("/api/v1/predict", json=valid_payload).status_code == 422
        )

    def test_resposta_tem_header_request_id(
        self, test_client: TestClient, valid_payload: dict
    ) -> None:
        r = test_client.post("/api/v1/predict", json=valid_payload)
        assert "x-request-id" in r.headers

    def test_predict_503_sem_predictor(
        self, test_client: TestClient, valid_payload: dict
    ) -> None:
        app.state.predictor = None
        r = test_client.post("/api/v1/predict", json=valid_payload)
        assert r.status_code == 503

    def test_predict_chama_predictor_uma_vez(
        self, test_client: TestClient, valid_payload: dict
    ) -> None:
        app.state.predictor.predict.reset_mock()
        test_client.post("/api/v1/predict", json=valid_payload)
        app.state.predictor.predict.assert_called_once()


class TestPayloadVariants:
    """Verifica que payloads com 19 e 33 campos retornam 200 com predição válida."""

    def _assert_predict_ok(self, r) -> None:
        assert r.status_code == 200
        body = r.json()
        assert "churn_probability" in body
        assert "churn_label" in body
        assert "threshold_used" in body
        assert "cost_estimate_brl" in body
        assert "model_version" in body
        assert body["churn_label"] in ("churn", "no_churn")
        assert 0.0 <= body["churn_probability"] <= 1.0

    def test_19_features_retorna_200_e_predicao_valida(
        self, test_client: TestClient, valid_payload: dict
    ) -> None:
        assert len(valid_payload) == 19
        self._assert_predict_ok(test_client.post("/api/v1/predict", json=valid_payload))

    def test_33_features_retorna_200_e_predicao_valida(
        self, test_client: TestClient, raw_payload_33: dict
    ) -> None:
        assert len(raw_payload_33) == 33
        self._assert_predict_ok(
            test_client.post("/api/v1/predict", json=raw_payload_33)
        )

    def test_33_features_descarta_colunas_extras(
        self, test_client: TestClient, raw_payload_33: dict
    ) -> None:
        """Colunas como CustomerID, Churn Label, CLTV devem ser ignoradas silenciosamente."""
        r = test_client.post("/api/v1/predict", json=raw_payload_33)
        assert r.status_code == 200

    def test_valores_em_lowercase_sao_normalizados(
        self, test_client: TestClient, valid_payload: dict
    ) -> None:
        """Aceita 'yes', 'no', 'dsl', 'month-to-month' (qualquer casing)."""
        valid_payload["gender"] = "female"
        valid_payload["senior_citizen"] = "yes"
        valid_payload["internet_service"] = "fiber optic"
        valid_payload["contract"] = "two year"
        valid_payload["payment_method"] = "mailed check"
        self._assert_predict_ok(test_client.post("/api/v1/predict", json=valid_payload))

    def test_valores_em_uppercase_sao_normalizados(
        self, test_client: TestClient, valid_payload: dict
    ) -> None:
        valid_payload["gender"] = "MALE"
        valid_payload["internet_service"] = "DSL"
        valid_payload["contract"] = "ONE YEAR"
        self._assert_predict_ok(test_client.post("/api/v1/predict", json=valid_payload))

    def test_chaves_em_title_case_sao_normalizadas(
        self, test_client: TestClient
    ) -> None:
        """Chaves originais do dataset IBM Telco (ex.: 'Tenure Months') devem funcionar."""
        payload = {
            "Gender": "Female",
            "Senior Citizen": "No",
            "Partner": "Yes",
            "Dependents": "No",
            "Tenure Months": 6,
            "Phone Service": "Yes",
            "Multiple Lines": "No",
            "Internet Service": "DSL",
            "Online Security": "No",
            "Online Backup": "No",
            "Device Protection": "No",
            "Tech Support": "No",
            "Streaming TV": "No",
            "Streaming Movies": "No",
            "Contract": "Month-to-month",
            "Paperless Billing": "Yes",
            "Payment Method": "Electronic check",
            "Monthly Charge": 45.0,
            "Total Charges": 270.0,
        }
        self._assert_predict_ok(test_client.post("/api/v1/predict", json=payload))

    def test_33_features_e_19_features_chamam_predictor_uma_vez_cada(
        self, test_client: TestClient, valid_payload: dict, raw_payload_33: dict
    ) -> None:
        app.state.predictor.predict.reset_mock()
        test_client.post("/api/v1/predict", json=valid_payload)
        test_client.post("/api/v1/predict", json=raw_payload_33)
        assert app.state.predictor.predict.call_count == 2


class TestRootEndpoint:
    def test_root_retorna_links(self, test_client: TestClient) -> None:
        r = test_client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert "docs" in body
        assert "health" in body
        assert "predict" in body


class TestPredictErrorHandlers:
    def test_value_error_no_pipeline_retorna_422(
        self, test_client: TestClient, valid_payload: dict
    ) -> None:
        app.state.predictor.predict.side_effect = ValueError("shape inesperado")
        r = test_client.post("/api/v1/predict", json=valid_payload)
        assert r.status_code == 422
        app.state.predictor.predict.side_effect = None

    def test_excecao_generica_retorna_500(
        self, test_client: TestClient, valid_payload: dict
    ) -> None:
        app.state.predictor.predict.side_effect = RuntimeError("erro inesperado")
        r = test_client.post("/api/v1/predict", json=valid_payload)
        assert r.status_code == 500
        app.state.predictor.predict.side_effect = None
