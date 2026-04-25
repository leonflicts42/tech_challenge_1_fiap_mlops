"""
api/main.py — Aplicação FastAPI de predição de churn.

Responsabilidades:
    - Criar o app FastAPI com metadata (título, versão, docs)
    - Lifespan: carregar ChurnPredictor UMA ÚNICA VEZ na subida
    - Registrar middleware de latência (LatencyMiddleware)
    - Incluir o router com os endpoints /health e /predict
    - Handler de exceções para erros de validação Pydantic

Uso:
    # Desenvolvimento
    uvicorn api.main:app --reload --port 8000

    # Produção (via Makefile)
    make run
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.middleware import LatencyMiddleware
from api.predictor import ChurnPredictor
from api.router import router
from config import MODELS_DIR, get_logger

logger: logging.Logger = get_logger(__name__)

# ── Lifespan: carrega artefatos antes de aceitar requisições ──────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida da aplicação.

    ON STARTUP:
        Instancia o ChurnPredictor, que internamente carrega:
        - preprocessor.pkl  (ColumnTransformer sklearn)
        - best_mlp.pt       (pesos da ChurnMLPv2)
        - optuna_best_params_*.json (threshold ótimo, se existir)

        Se qualquer artefato estiver ausente, a aplicação falha na subida
        com um erro claro — em vez de falhar silenciosamente na primeira
        requisição.

        O predictor é armazenado em app.state.predictor para ser acessado
        pelos endpoints via request.app.state.predictor.

    ON SHUTDOWN:
        Limpeza de recursos (extensível).
    """
    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Churn API — iniciando")
    logger.info("MODELS_DIR: %s", MODELS_DIR)

    try:
        app.state.predictor = ChurnPredictor(
            preprocessor_path=MODELS_DIR / "preprocessor.pkl",
            model_path=MODELS_DIR / "best_model_mlp.pt",
        )
        logger.info("ChurnPredictor carregado com sucesso")
    except FileNotFoundError as exc:
        # Falha na subida: artefatos não encontrados
        # A API sobe mas retorna 503 em /predict e /health
        logger.error("FALHA NO STARTUP — artefato ausente: %s", exc)
        app.state.predictor = None
    except Exception as exc:  # noqa: BLE001
        logger.error("FALHA NO STARTUP — erro inesperado: %s", exc, exc_info=True)
        app.state.predictor = None

    logger.info("=" * 60)

    yield  # A API fica ativa aqui

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Churn API — encerrando")
    app.state.predictor = None


# ── Criação da aplicação ──────────────────────────────────────────────────────

app = FastAPI(
    title="Churn Telecom API",
    description=(
        "API de predição de churn para o setor de telecomunicações.\n\n"
        "Modelo: **ChurnMLPv2** (PyTorch) treinado no dataset IBM Telco.\n\n"
        "Pipeline interno: `to_snake_case` → `SemanticNormalizer` "
        "→ `FeatureEngineer` → `preprocessor.pkl` → `ChurnMLPv2`"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ────────────────────────────────────────────────────────────────
# Registrar ANTES do router para interceptar todas as requisições
app.add_middleware(LatencyMiddleware)


# ── Exception handlers ────────────────────────────────────────────────────────


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Formata erros de validação Pydantic em JSON estruturado.

    Retorna 422 com detalhes legíveis sobre o campo inválido, o valor
    recebido e a mensagem de erro — sem expor stack traces.

    Exemplo de resposta:
        {
          "error": "validation_error",
          "detail": [
            {
              "field": "tenure_months",
              "message": "Input should be less than or equal to 72",
              "received": 100
            }
          ],
          "request_id": "uuid-da-requisição"
        }
    """
    request_id = getattr(getattr(request, "state", None), "request_id", "unknown")

    errors = []
    for error in exc.errors():
        field = " → ".join(str(loc) for loc in error.get("loc", []))
        errors.append(
            {
                "field": field,
                "message": error.get("msg", ""),
                "received": error.get("input"),
            }
        )

    logger.warning(
        "validation_error | n_errors=%d | request_id=%s | fields=%s",
        len(errors),
        request_id,
        [e["field"] for e in errors],
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={
            "error": "validation_error",
            "detail": errors,
            "request_id": request_id,
        },
    )


# ── Router ────────────────────────────────────────────────────────────────────
app.include_router(router)


# ── Rota raiz (redirect para docs) ───────────────────────────────────────────


@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {
        "message": "Churn Telecom API",
        "docs": "/docs",
        "health": "/api/v1/health",
        "predict": "/api/v1/predict",
    }
