"""
api/router.py — Endpoints da API de predição de churn.

Isola as rotas do objeto FastAPI principal (main.py), seguindo o princípio
de separação de responsabilidades. O router é incluído em main.py via
app.include_router().

Endpoints:
    GET  /health   → liveness probe (status do modelo e preprocessor)
    POST /predict  → predição de churn para um cliente
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, Request, status

from api.schemas import ChurnRequest, ChurnResponse
from config import get_logger

logger: logging.Logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["churn"])


# ── GET /health ───────────────────────────────────────────────────────────────


@router.get(
    "/health",
    summary="Liveness probe",
    description=(
        "Verifica se a API está ativa e se os artefatos de ML "
        "(modelo e preprocessor) foram carregados corretamente."
    ),
    response_model=dict,
    status_code=status.HTTP_200_OK,
)
async def health(request: Request) -> dict:
    """Retorna o status de saúde da API.

    Verifica a presença de app.state.predictor para confirmar que
    o lifespan carregou os artefatos com sucesso.

    Returns:
        JSON com status, flags de artefatos carregados e versão do modelo.

    Raises:
        503: se o predictor não estiver disponível (falha no lifespan)
    """
    predictor = getattr(request.app.state, "predictor", None)

    if predictor is None:
        logger.warning("health | predictor não disponível em app.state")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "unavailable",
                "model_loaded": False,
                "preprocessor_loaded": False,
                "detail": (
                    "Artefatos de ML não carregados — "
                    "verifique os logs de startup."
                ),
            },
        )

    model_loaded = predictor._model is not None
    preprocessor_loaded = predictor._preprocessor is not None

    logger.info(
        "health | status=ok | model_loaded=%s | preprocessor_loaded=%s",
        model_loaded,
        preprocessor_loaded,
    )

    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "preprocessor_loaded": preprocessor_loaded,
        "model_version": predictor.model_version,
        "threshold": predictor._threshold,
    }


# ── POST /predict ─────────────────────────────────────────────────────────────


@router.post(
    "/predict",
    summary="Predição de churn",
    description=(
        "Recebe os dados brutos de um cliente e retorna a probabilidade "
        "de churn, a classificação e a estimativa de custo de negócio.\n\n"
        "O pipeline interno aplica:\n"
        "1. Normalização de nomes de colunas (snake_case)\n"
        "2. Limpeza semântica (SemanticNormalizer)\n"
        "3. Feature engineering (FeatureEngineer — 6 features)\n"
        "4. Transformação estatística (preprocessor.pkl)\n"
        "5. Inferência (ChurnMLPv2)\n"
    ),
    response_model=ChurnResponse,
    status_code=status.HTTP_200_OK,
    responses={
        422: {"description": "Payload inválido — campo ausente ou fora do domínio"},
        503: {"description": "Modelo não disponível"},
    },
)
async def predict(request: Request, payload: ChurnRequest) -> ChurnResponse:
    """Executa o pipeline completo de predição para um cliente.

    Args:
        request: objeto Request do FastAPI (acesso a app.state)
        payload: dados do cliente validados pelo Pydantic

    Returns:
        ChurnResponse com probabilidade, label, threshold e custo estimado

    Raises:
        503: se o predictor não estiver disponível
        500: se ocorrer erro interno no pipeline
    """
    predictor = getattr(request.app.state, "predictor", None)

    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo não disponível. Verifique os logs de startup.",
        )

    start = time.perf_counter()

    try:
        response = predictor.predict(payload)
    except ValueError as exc:
        logger.error("predict | ValueError no pipeline: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Erro no pipeline de transformação: {exc}",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("predict | erro inesperado: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno na predição. Verifique os logs.",
        ) from exc

    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "predict | churn_prob=%.4f | label=%s | latency_ms=%.1f",
        response.churn_probability,
        response.churn_label,
        elapsed_ms,
    )

    return response