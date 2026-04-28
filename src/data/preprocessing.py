"""
preprocessing.py — Transformadores de limpeza semântica para o pipeline de inferência.

Contém o SemanticNormalizer: um sklearn transformer que encapsula as duas
operações de limpeza que devem ser aplicadas ANTES do FeatureEngineer e do
ColumnTransformer, tanto no treino quanto em produção (API).

Por que é um transformer separado?
    - Isolamento de responsabilidade: limpeza != engenharia != encoding
    - Testável independentemente
    - Serializado no Pipeline sklearn → comportamento idêntico em treino e produção
    - fit() é no-op: sem estado aprendido, aplicável a qualquer split
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from config import NO_SERVICE_COLS, get_logger

logger: logging.Logger = get_logger(__name__)


class SemanticNormalizer(BaseEstimator, TransformerMixin):
    """Normaliza valores semânticos redundantes e corrige inconsistências lógicas.

    Operação 1 — Normalização "No internet service" → "No":
        Seis colunas de serviços dependentes de internet possuem 3 valores
        possíveis: "Yes", "No" e "No internet service". Os dois últimos têm
        o mesmo significado preditivo (cliente não possui o serviço), então
        são consolidados em "No" para reduzir cardinalidade no OHE.

        Colunas afetadas (definidas em config.NO_SERVICE_COLS):
            online_security, online_backup, device_protection,
            tech_support, streaming_tv, streaming_movies

    Operação 2 — Correção de inconsistências lógicas:
        Se internet_service = "No", nenhum serviço de internet pode estar
        ativo ("Yes"). Inconsistências deste tipo são erros de entrada de
        dados no CRM e são corrigidas para "No".

    Importante:
        - fit() é no-op: este transformer não aprende nenhum estado dos dados
        - transform() opera apenas em cópias (não modifica o DataFrame original)
        - Seguro para uso em produção: opera apenas em strings, sem estatísticas
    """

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> SemanticNormalizer:
        """No-op: nenhum estado é aprendido nesta etapa.

        Args:
            X: DataFrame com colunas brutas (snake_case).
            y: Ignorado. Presente por compatibilidade com sklearn Pipeline.

        Returns:
            self
        """
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Aplica normalização semântica e correção de inconsistências.

        Args:
            X: DataFrame com colunas brutas em snake_case. Deve conter as
               colunas definidas em NO_SERVICE_COLS. Colunas ausentes são
               ignoradas com warning (não levantam exceção — robusto a
               subconjuntos de features).
            y: Ignorado.

        Returns:
            Novo DataFrame com os mesmos índices, colunas normalizadas.
            O DataFrame original não é modificado.
        """
        X_out = X.copy()

        # ── Operação 1: "No internet service" → "No" ──────────────────────────
        total_substituicoes = 0

        for col in NO_SERVICE_COLS:
            if col not in X_out.columns:
                logger.warning(
                    "SemanticNormalizer | coluna ausente (ignorada): '%s'", col
                )
                continue

            n_afetados = int((X_out[col] == "No internet service").sum())

            if n_afetados > 0:
                X_out[col] = X_out[col].replace({"No internet service": "No"})
                total_substituicoes += n_afetados
                logger.debug(
                    "SemanticNormalizer | %s | 'No internet service' → 'No' | n=%d",
                    col,
                    n_afetados,
                )

        if total_substituicoes > 0:
            logger.info(
                "SemanticNormalizer | normalização semântica | "
                "total_substituições=%d em %d colunas",
                total_substituicoes,
                len(NO_SERVICE_COLS),
            )

        # ── Operação 2: inconsistências lógicas (sem internet, mas serviço ativo) ──
        internet_col = "internet_service"

        if internet_col in X_out.columns:
            mask_sem_internet = X_out[internet_col] == "No"
            total_correcoes = 0

            for col in NO_SERVICE_COLS:
                if col not in X_out.columns:
                    continue

                n_inconsistentes = int((mask_sem_internet & (X_out[col] != "No")).sum())

                if n_inconsistentes > 0:
                    X_out.loc[mask_sem_internet, col] = "No"
                    total_correcoes += n_inconsistentes
                    logger.warning(
                        "SemanticNormalizer | %s | "
                        "%d inconsistências corrigidas "
                        "(sem internet mas serviço ativo)",
                        col,
                        n_inconsistentes,
                    )

            if total_correcoes == 0:
                logger.debug(
                    "SemanticNormalizer | nenhuma inconsistência lógica encontrada"
                )
        else:
            logger.warning(
                "SemanticNormalizer | '%s' ausente — "
                "validação de inconsistências lógicas ignorada",
                internet_col,
            )

        return X_out

    def get_feature_names_out(
        self,
        input_features: list[str] | None = None,
    ) -> list[str]:
        """Retorna os nomes das features de saída.

        SemanticNormalizer não adiciona nem remove colunas — os nomes são
        idênticos à entrada. Necessário para compatibilidade com
        Pipeline.get_feature_names_out() no sklearn ≥ 1.3.

        Args:
            input_features: lista de nomes da entrada. Se None, retorna lista vazia.

        Returns:
            Lista de nomes de features (idêntica à entrada).
        """
        if input_features is None:
            return []
        return list(input_features)
