"""
api/middleware.py — Middleware de latência e rastreabilidade.

Intercepta cada requisição para:
    1. Gerar um UUID único (request_id) por requisição
    2. Medir o tempo de processamento em milissegundos
    3. Injetar X-Process-Time e X-Request-ID nos headers da resposta
    4. Logar o evento completo (método, path, status, latência, request_id)

O request_id é armazenado em request.state.request_id para que outros
componentes (exception_handlers, router) possam referenciá-lo nos logs,
permitindo rastrear uma requisição de ponta a ponta nos logs.
"""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from config import get_logger

logger: logging.Logger = get_logger(__name__)


class LatencyMiddleware(BaseHTTPMiddleware):
    """Middleware que mede latência e injeta headers de rastreabilidade.

    Headers adicionados à resposta:
        X-Request-ID:    UUID único por requisição (rastreabilidade)
        X-Process-Time:  tempo de processamento em milissegundos

    Atributos em request.state:
        request_id:  mesmo UUID do header X-Request-ID
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.perf_counter()

        try:
            response: Response = await call_next(request)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "request | %s %s | ERROR | %.1fms | request_id=%s | exc=%s",
                request.method,
                request.url.path,
                elapsed_ms,
                request_id,
                exc,
            )
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{elapsed_ms:.2f}ms"

        logger.info(
            "request | %s %s | %d | %.1fms | request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id,
        )

        return response