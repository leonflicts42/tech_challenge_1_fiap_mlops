"""Testes de validação do schema Pydantic — sem dependência de API ou artefatos."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.schemas import ChurnRequest, ChurnResponse


class TestChurnRequestValidation:
    def test_valid_payload_creates_instance(self, valid_payload: dict) -> None:
        req = ChurnRequest(**valid_payload)
        assert req.gender == "Female"
        assert req.tenure_months == 12

    def test_rejeita_gender_invalido(self, valid_payload: dict) -> None:
        valid_payload["gender"] = "Other"
        with pytest.raises(ValidationError):
            ChurnRequest(**valid_payload)

    def test_rejeita_tenure_acima_de_72(self, valid_payload: dict) -> None:
        valid_payload["tenure_months"] = 73
        with pytest.raises(ValidationError):
            ChurnRequest(**valid_payload)

    def test_rejeita_monthly_charges_negativo(self, valid_payload: dict) -> None:
        valid_payload["monthly_charges"] = -1.0
        with pytest.raises(ValidationError):
            ChurnRequest(**valid_payload)

    def test_rejeita_contract_invalido(self, valid_payload: dict) -> None:
        valid_payload["contract"] = "Weekly"
        with pytest.raises(ValidationError):
            ChurnRequest(**valid_payload)

    def test_rejeita_payment_method_invalido(self, valid_payload: dict) -> None:
        valid_payload["payment_method"] = "Crypto"
        with pytest.raises(ValidationError):
            ChurnRequest(**valid_payload)

    def test_rejeita_senior_citizen_invalido(self, valid_payload: dict) -> None:
        valid_payload["senior_citizen"] = "Maybe"
        with pytest.raises(ValidationError):
            ChurnRequest(**valid_payload)

    def test_rejeita_internet_service_invalido(self, valid_payload: dict) -> None:
        valid_payload["internet_service"] = "5G"
        with pytest.raises(ValidationError):
            ChurnRequest(**valid_payload)

    def test_aceita_no_internet_service_em_servicos(self, valid_payload: dict) -> None:
        valid_payload["online_security"] = "No internet service"
        req = ChurnRequest(**valid_payload)
        assert req.online_security == "No internet service"

    def test_model_dump_tem_19_campos(self, valid_payload: dict) -> None:
        req = ChurnRequest(**valid_payload)
        assert len(req.model_dump()) == 19

    def test_aceita_tenure_zero(self, valid_payload: dict) -> None:
        valid_payload["tenure_months"] = 0
        req = ChurnRequest(**valid_payload)
        assert req.tenure_months == 0

    def test_aceita_total_charges_zero(self, valid_payload: dict) -> None:
        valid_payload["total_charges"] = 0.0
        req = ChurnRequest(**valid_payload)
        assert req.total_charges == 0.0

    def test_normalize_inputs_ignora_nao_dict(self) -> None:
        """normalize_inputs retorna inalterado se não for dict."""
        result = ChurnRequest.normalize_inputs("not_a_dict")
        assert result == "not_a_dict"

    def test_internet_inconsistency_nao_rejeita_requisicao(
        self, valid_payload: dict
    ) -> None:
        """internet_service=No com serviço ativo não levanta erro (SemanticNormalizer corrige)."""
        valid_payload["internet_service"] = "No"
        valid_payload["online_security"] = "Yes"
        req = ChurnRequest(**valid_payload)
        assert req.internet_service == "No"
        assert req.online_security == "Yes"


class TestChurnResponseSchema:
    def test_response_valida(self) -> None:
        r = ChurnResponse(
            churn_probability=0.83,
            churn_label="churn",
            threshold_used=0.16,
            cost_estimate_brl=73.52,
            model_version="v1",
        )
        assert r.churn_label == "churn"

    def test_rejeita_probabilidade_acima_de_1(self) -> None:
        with pytest.raises(ValidationError):
            ChurnResponse(
                churn_probability=1.5,
                churn_label="churn",
                threshold_used=0.5,
                cost_estimate_brl=0.0,
                model_version="v1",
            )

    def test_rejeita_label_invalido(self) -> None:
        with pytest.raises(ValidationError):
            ChurnResponse(
                churn_probability=0.5,
                churn_label="maybe",
                threshold_used=0.5,
                cost_estimate_brl=0.0,
                model_version="v1",
            )
