"""Testes para SemanticNormalizer — data/preprocessing.py."""

from __future__ import annotations

import pandas as pd
import pytest

from data.preprocessing import SemanticNormalizer


@pytest.fixture
def normalizer() -> SemanticNormalizer:
    return SemanticNormalizer()


@pytest.fixture
def base_df() -> pd.DataFrame:
    return pd.DataFrame([{
        "internet_service": "DSL",
        "online_security": "No internet service",
        "online_backup": "No internet service",
        "device_protection": "No",
        "tech_support": "Yes",
        "streaming_tv": "No internet service",
        "streaming_movies": "No",
    }])


class TestSemanticNormalizerFit:
    def test_fit_retorna_self(self, normalizer: SemanticNormalizer, base_df: pd.DataFrame) -> None:
        result = normalizer.fit(base_df)
        assert result is normalizer

    def test_fit_com_y_retorna_self(self, normalizer: SemanticNormalizer, base_df: pd.DataFrame) -> None:
        result = normalizer.fit(base_df, y=pd.Series([0]))
        assert result is normalizer


class TestSemanticNormalizerTransform:
    def test_no_internet_service_vira_no(self, normalizer: SemanticNormalizer, base_df: pd.DataFrame) -> None:
        out = normalizer.transform(base_df)
        assert out["online_security"].iloc[0] == "No"
        assert out["online_backup"].iloc[0] == "No"
        assert out["streaming_tv"].iloc[0] == "No"

    def test_nao_modifica_valores_ja_corretos(self, normalizer: SemanticNormalizer, base_df: pd.DataFrame) -> None:
        out = normalizer.transform(base_df)
        assert out["tech_support"].iloc[0] == "Yes"
        assert out["device_protection"].iloc[0] == "No"

    def test_nao_modifica_dataframe_original(self, normalizer: SemanticNormalizer, base_df: pd.DataFrame) -> None:
        original_val = base_df["online_security"].iloc[0]
        normalizer.transform(base_df)
        assert base_df["online_security"].iloc[0] == original_val

    def test_inconsistencia_sem_internet_corrige_servicos(self, normalizer: SemanticNormalizer) -> None:
        """internet_service=No mas tech_support=Yes → corrige para No."""
        df = pd.DataFrame([{
            "internet_service": "No",
            "online_security": "Yes",
            "online_backup": "No",
            "device_protection": "Yes",
            "tech_support": "Yes",
            "streaming_tv": "No",
            "streaming_movies": "Yes",
        }])
        out = normalizer.transform(df)
        assert out["online_security"].iloc[0] == "No"
        assert out["device_protection"].iloc[0] == "No"
        assert out["tech_support"].iloc[0] == "No"
        assert out["streaming_movies"].iloc[0] == "No"

    def test_sem_inconsistencia_nao_altera_nada(self, normalizer: SemanticNormalizer) -> None:
        df = pd.DataFrame([{
            "internet_service": "No",
            "online_security": "No",
            "online_backup": "No",
            "device_protection": "No",
            "tech_support": "No",
            "streaming_tv": "No",
            "streaming_movies": "No",
        }])
        out = normalizer.transform(df)
        for col in ["online_security", "online_backup", "tech_support"]:
            assert out[col].iloc[0] == "No"

    def test_coluna_ausente_e_ignorada(self, normalizer: SemanticNormalizer) -> None:
        df = pd.DataFrame([{"internet_service": "DSL", "online_security": "No internet service"}])
        out = normalizer.transform(df)
        assert out["online_security"].iloc[0] == "No"

    def test_sem_internet_service_col_ignora_validacao(self, normalizer: SemanticNormalizer) -> None:
        df = pd.DataFrame([{"online_security": "No internet service", "tech_support": "Yes"}])
        out = normalizer.transform(df)
        assert out["online_security"].iloc[0] == "No"

    def test_multiplas_linhas(self, normalizer: SemanticNormalizer) -> None:
        df = pd.DataFrame([
            {"internet_service": "DSL", "online_security": "No internet service",
             "online_backup": "Yes", "device_protection": "No",
             "tech_support": "No", "streaming_tv": "No internet service",
             "streaming_movies": "No"},
            {"internet_service": "No", "online_security": "Yes",
             "online_backup": "No", "device_protection": "No",
             "tech_support": "Yes", "streaming_tv": "No", "streaming_movies": "No"},
        ])
        out = normalizer.transform(df)
        assert out["online_security"].iloc[0] == "No"
        assert out["streaming_tv"].iloc[0] == "No"
        assert out["online_security"].iloc[1] == "No"
        assert out["tech_support"].iloc[1] == "No"


class TestSemanticNormalizerGetFeatureNamesOut:
    def test_retorna_mesmas_colunas(self, normalizer: SemanticNormalizer) -> None:
        cols = ["a", "b", "c"]
        assert normalizer.get_feature_names_out(cols) == cols

    def test_none_retorna_lista_vazia(self, normalizer: SemanticNormalizer) -> None:
        assert normalizer.get_feature_names_out(None) == []
