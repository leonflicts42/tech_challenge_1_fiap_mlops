import os
import sys
from unittest.mock import MagicMock

# Ignora arquivos "copy" com espaços no nome (backups acidentais)
collect_ignore_glob = ["*copy*"]

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from api.schemas import ChurnRequest, ChurnResponse  # noqa: E402
from main import app  # noqa: E402

VALID_PAYLOAD: dict = {
    "gender": "Female",
    "senior_citizen": "No",
    "partner": "Yes",
    "dependents": "No",
    "tenure_months": 12,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "DSL",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 29.85,
    "total_charges": 358.20,
}


def _make_mock_predictor() -> MagicMock:
    pred = MagicMock()
    pred._model = MagicMock()
    pred._preprocessor = MagicMock()
    pred.model_version = "test-v1"
    pred._threshold = 0.16
    pred.predict.return_value = ChurnResponse(
        churn_probability=0.83,
        churn_label="churn",
        threshold_used=0.16,
        cost_estimate_brl=73.52,
        model_version="test-v1",
    )
    return pred


RAW_PAYLOAD_33: dict = {
    "CustomerID": "8779-QRDMV",
    "Count": 1,
    "Country": "United States",
    "State": "California",
    "City": "Los Angeles",
    "Zip Code": 90001,
    "Lat Long": "34.052235, -118.243683",
    "Latitude": 34.052235,
    "Longitude": -118.243683,
    "Gender": "male",
    "Senior Citizen": "no",
    "Partner": "yes",
    "Dependents": "no",
    "Tenure Months": 12,
    "Phone Service": "yes",
    "Multiple Lines": "no",
    "Internet Service": "dsl",
    "Online Security": "no",
    "Online Backup": "no",
    "Device Protection": "no",
    "Tech Support": "no",
    "Streaming TV": "no",
    "Streaming Movies": "no",
    "Contract": "month-to-month",
    "Paperless Billing": "yes",
    "Payment Method": "electronic check",
    "Monthly Charge": 29.85,
    "Total Charges": 358.20,
    "Churn Label": "No",
    "Churn Value": 0,
    "Churn Score": 26,
    "CLTV": 5433,
    "Churn Reason": "",
}


@pytest.fixture
def valid_payload() -> dict:
    return VALID_PAYLOAD.copy()


@pytest.fixture
def raw_payload_33() -> dict:
    return RAW_PAYLOAD_33.copy()


@pytest.fixture
def test_client():
    """TestClient com predictor mockado no app.state."""
    with TestClient(app, raise_server_exceptions=False) as client:
        app.state.predictor = _make_mock_predictor()
        yield client
