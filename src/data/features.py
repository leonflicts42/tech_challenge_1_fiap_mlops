"""
features.py — FeatureEngineer: criação das 6 features derivadas.

Encapsula toda a lógica de feature engineering do notebook 2_vab_preprocessing
em um sklearn transformer. Deve ser aplicado APÓS o SemanticNormalizer e
ANTES do ColumnTransformer (preprocessor.pkl).

Ordem no pipeline de inferência:
    SemanticNormalizer → FeatureEngineer → ColumnTransformer

Por que é um transformer separado?
    - Reproduzível: mesma lógica em treino, validação e produção (API)
    - Testável: cada feature pode ser validada isoladamente
    - fit() é no-op: sem estado aprendido → seguro para qualquer split
    - Não remove colunas originais: ColumnTransformer decide o que usar
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from config import (
    BINARY_NEW_FEATURES,
    NEW_FEATURES,
    SERVICES_COLS,
    get_logger,
)

logger: logging.Logger = get_logger(__name__)

# ── Constantes de colunas de entrada (snake_case) ─────────────────────────────
_TENURE_COL = "tenure_months"
_MONTHLY_COL = "monthly_charges"
_CONTRACT_COL = "contract"
_INTERNET_COL = "internet_service"
_SECURITY_COL = "online_security"
_SUPPORT_COL = "tech_support"


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Cria as 6 features derivadas a partir das colunas brutas.

    Features criadas:
        1. num_services (int, range [0,7]):
           Contagem de serviços ativos dentre os 7 definidos em SERVICES_COLS.
           Hipótese: mais serviços → maior switching cost → menor churn.
           Correlação com target: -0.08 (sinal fraco mas sinergista com is_fiber_optic).

        2. charges_per_month (float):
           monthly_charges / (tenure_months + 1).
           Captura percepção de "custo-benefício" — clientes novos com planos
           caros têm valores altos. Correlação com target: +0.41.

        3. is_month_to_month (int, {0,1}):
           1 se contract = "Month-to-month", 0 caso contrário.
           Cramer's V = 0.41. Churn rate month-to-month: 42.7% vs 6.8% outros.

        4. tenure_group (str, {"novo","medio","longo"}):
           Bucketing de tenure_months em 3 faixas:
               - "novo":  0–12 meses  (churn ~47%)
               - "medio": 13–48 meses (churn ~24%)
               - "longo": >48 meses   (churn  ~9%)
           Captura não-linearidade que modelos lineares não capturam.

        5. has_security_support (int, {0,1}):
           1 se online_security = "Yes" OU tech_support = "Yes".
           Consolida duas features correlacionadas (Cramer's V ≈ 0.33–0.34).
           Correlação com target: -0.18.

        6. is_fiber_optic (int, {0,1}):
           1 se internet_service = "Fiber optic".
           Paradoxo: maior churn entre clientes de fibra (41.8% vs 14.4%).
           Correlação com target: +0.31.

    Importante:
        - fit() é no-op: nenhum estado é aprendido
        - As colunas originais são mantidas no DataFrame de saída
        - Encoding (OrdinalEncoder, OHE) é responsabilidade do ColumnTransformer
    """

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> FeatureEngineer:
        """No-op: nenhum estado é aprendido nesta etapa.

        Args:
            X: DataFrame com colunas brutas em snake_case.
            y: Ignorado. Presente por compatibilidade com sklearn Pipeline.

        Returns:
            self
        """
        return self

    def transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Cria as 6 features derivadas e as adiciona ao DataFrame.

        Args:
            X: DataFrame com colunas brutas em snake_case. Deve conter as
               colunas de entrada usadas pelas features (tenure_months,
               monthly_charges, contract, internet_service, online_security,
               tech_support e as colunas de SERVICES_COLS).
            y: Ignorado.

        Returns:
            Novo DataFrame com as colunas originais mais as 6 features criadas.
            O DataFrame original não é modificado.

        Raises:
            ValueError: se uma coluna obrigatória estiver ausente.
        """
        X_out = X.copy()

        self._validate_input_columns(X_out)

        X_out = self._add_num_services(X_out)
        X_out = self._add_charges_per_month(X_out)
        X_out = self._add_is_month_to_month(X_out)
        X_out = self._add_tenure_group(X_out)
        X_out = self._add_has_security_support(X_out)
        X_out = self._add_is_fiber_optic(X_out)

        logger.info(
            "FeatureEngineer | %d features criadas | shape: %s → %s",
            len(NEW_FEATURES),
            X.shape,
            X_out.shape,
        )

        return X_out

    # ── Features individuais ───────────────────────────────────────────────────

    def _add_num_services(self, X: pd.DataFrame) -> pd.DataFrame:
        """Conta quantos dos 7 serviços estão ativos ("Yes") por cliente."""
        cols_present = [c for c in SERVICES_COLS if c in X.columns]
        X["num_services"] = (X[cols_present] == "Yes").sum(axis=1).astype(int)
        logger.debug(
            "FeatureEngineer | num_services | min=%d | max=%d | média=%.2f",
            X["num_services"].min(),
            X["num_services"].max(),
            X["num_services"].mean(),
        )
        return X

    def _add_charges_per_month(self, X: pd.DataFrame) -> pd.DataFrame:
        """Custo relativo ao tempo de casa: monthly_charges / (tenure + 1)."""
        X["charges_per_month"] = (X[_MONTHLY_COL] / (X[_TENURE_COL] + 1)).round(4)
        logger.debug(
            "FeatureEngineer | charges_per_month | min=%.2f | max=%.2f | média=%.2f",
            X["charges_per_month"].min(),
            X["charges_per_month"].max(),
            X["charges_per_month"].mean(),
        )
        return X

    def _add_is_month_to_month(self, X: pd.DataFrame) -> pd.DataFrame:
        """Flag binária: 1 se contrato é Month-to-month."""
        X["is_month_to_month"] = (X[_CONTRACT_COL] == "Month-to-month").astype(int)
        logger.debug(
            "FeatureEngineer | is_month_to_month | n_mtm=%d (%.1f%%)",
            X["is_month_to_month"].sum(),
            X["is_month_to_month"].mean() * 100,
        )
        return X

    def _add_tenure_group(self, X: pd.DataFrame) -> pd.DataFrame:
        """Segmenta tenure_months em 3 faixas de risco."""
        X["tenure_group"] = pd.cut(
            X[_TENURE_COL],
            bins=[0, 12, 48, float("inf")],
            labels=["novo", "medio", "longo"],
            right=True,
            include_lowest=True,
        ).astype(str)
        logger.debug(
            "FeatureEngineer | tenure_group | distribuição: %s",
            X["tenure_group"].value_counts().to_dict(),
        )
        return X

    def _add_has_security_support(self, X: pd.DataFrame) -> pd.DataFrame:
        """Flag: 1 se tem online_security="Yes" OU tech_support="Yes"."""
        X["has_security_support"] = (
            (X[_SECURITY_COL] == "Yes") | (X[_SUPPORT_COL] == "Yes")
        ).astype(int)
        logger.debug(
            "FeatureEngineer | has_security_support | n=%d (%.1f%%)",
            X["has_security_support"].sum(),
            X["has_security_support"].mean() * 100,
        )
        return X

    def _add_is_fiber_optic(self, X: pd.DataFrame) -> pd.DataFrame:
        """Flag: 1 se internet_service="Fiber optic"."""
        X["is_fiber_optic"] = (X[_INTERNET_COL] == "Fiber optic").astype(int)
        logger.debug(
            "FeatureEngineer | is_fiber_optic | n=%d (%.1f%%)",
            X["is_fiber_optic"].sum(),
            X["is_fiber_optic"].mean() * 100,
        )
        return X

    # ── Validação de entrada ───────────────────────────────────────────────────

    def _validate_input_columns(self, X: pd.DataFrame) -> None:
        """Valida que as colunas obrigatórias estão presentes.

        Raises:
            ValueError: lista de colunas ausentes se houver alguma.
        """
        required = [
            _TENURE_COL,
            _MONTHLY_COL,
            _CONTRACT_COL,
            _INTERNET_COL,
            _SECURITY_COL,
            _SUPPORT_COL,
            *SERVICES_COLS,
        ]
        missing = [c for c in required if c not in X.columns]
        if missing:
            raise ValueError(
                "FeatureEngineer.transform() — "
                f"colunas obrigatórias ausentes: {missing}"
            )

    def get_feature_names_out(
        self,
        input_features: list[str] | None = None,
    ) -> list[str]:
        """Retorna os nomes das features de saída.

        FeatureEngineer adiciona as 6 novas features ao final das colunas de
        entrada. Necessário para compatibilidade com Pipeline.get_feature_names_out()
        no sklearn ≥ 1.3.

        Args:
            input_features: lista de nomes da entrada.

        Returns:
            Lista com colunas de entrada + 6 novas features.
        """
        base = list(input_features) if input_features is not None else []
        # Adiciona apenas as features que ainda não estão na lista
        extras = [f for f in NEW_FEATURES if f not in base]
        return base + extras

    # ── Validação de saída (para testes) ──────────────────────────────────────

    def validate_output(self, X_out: pd.DataFrame) -> None:
        """Valida contratos de saída do transformer.

        Usado principalmente em testes automatizados e no notebook de validação.
        Levanta AssertionError se qualquer contrato for violado.

        Args:
            X_out: DataFrame retornado por transform().

        Raises:
            AssertionError: se algum contrato for violado.
        """
        # 1. Todas as 6 features foram criadas
        for feat in NEW_FEATURES:
            assert feat in X_out.columns, f"Feature ausente: '{feat}'"

        # 2. Nenhuma feature nova tem nulos
        nulos = X_out[NEW_FEATURES].isnull().sum()
        nulos_problema = nulos[nulos > 0]
        assert len(nulos_problema) == 0, (
            f"Nulos em features novas: {nulos_problema.to_dict()}"
        )

        # 3. Features binárias contêm apenas {0, 1}
        for feat in BINARY_NEW_FEATURES:
            valores = set(X_out[feat].unique())
            assert valores.issubset({0, 1}), (
                f"'{feat}' contém valores inesperados: {valores}"
            )

        # 4. num_services no range [0, len(SERVICES_COLS)]
        assert X_out["num_services"].between(0, len(SERVICES_COLS)).all(), (
            f"num_services fora do range [0, {len(SERVICES_COLS)}]"
        )

        # 5. tenure_group contém apenas os 3 valores esperados
        valores_tg = set(X_out["tenure_group"].unique())
        esperados_tg = {"novo", "medio", "longo"}
        assert valores_tg.issubset(esperados_tg), (
            f"tenure_group contém valores inesperados: {valores_tg - esperados_tg}"
        )

        # 6. charges_per_month não tem infinitos ou negativos
        assert (X_out["charges_per_month"] >= 0).all(), (
            "charges_per_month contém valores negativos"
        )
        assert X_out["charges_per_month"].notna().all(), "charges_per_month contém NaN"

        logger.info("FeatureEngineer.validate_output() | todos os contratos OK")
