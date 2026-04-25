"""Testes do FeatureEngineer — cobertura das 6 features derivadas."""

from __future__ import annotations

import pandas as pd
import pytest

from data.features import FeatureEngineer


# ── Fixtures ──────────────────────────────────────────────────────────────────

BASE_ROW: dict = {
    "tenure_months": 24,
    "monthly_charges": 65.0,
    "total_charges": 1560.0,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "Yes",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "No",
    "phone_service": "Yes",
    "multiple_lines": "No",
    "gender": "Female",
    "senior_citizen": "No",
    "partner": "Yes",
    "dependents": "No",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
}


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame([BASE_ROW.copy()])


@pytest.fixture
def eng() -> FeatureEngineer:
    return FeatureEngineer()


# ── fit / interface ───────────────────────────────────────────────────────────


def test_fit_retorna_self(eng: FeatureEngineer, df: pd.DataFrame) -> None:
    assert eng.fit(df) is eng


def test_fit_com_y_retorna_self(eng: FeatureEngineer, df: pd.DataFrame) -> None:
    import pandas as _pd

    y = _pd.Series([0])
    assert eng.fit(df, y) is eng


def test_transform_adiciona_6_colunas(eng: FeatureEngineer, df: pd.DataFrame) -> None:
    out = eng.transform(df)
    assert out.shape[1] == df.shape[1] + 6


def test_transform_nao_modifica_original(
    eng: FeatureEngineer, df: pd.DataFrame
) -> None:
    cols_antes = list(df.columns)
    eng.transform(df)
    assert list(df.columns) == cols_antes


def test_get_feature_names_out_adiciona_6(eng: FeatureEngineer) -> None:
    base = ["a", "b"]
    out = eng.get_feature_names_out(base)
    assert len(out) == len(base) + 6


def test_get_feature_names_out_sem_input(eng: FeatureEngineer) -> None:
    out = eng.get_feature_names_out(None)
    assert len(out) == 6


def test_get_feature_names_out_sem_duplicatas(eng: FeatureEngineer) -> None:
    base = ["num_services", "charges_per_month"]
    out = eng.get_feature_names_out(base)
    assert out.count("num_services") == 1


# ── num_services ─────────────────────────────────────────────────────────────


def test_num_services_conta_corretamente(eng: FeatureEngineer) -> None:
    # phone=Yes, online_backup=Yes, streaming_tv=Yes → 3 ativos
    out = eng.transform(pd.DataFrame([BASE_ROW.copy()]))
    assert out["num_services"].iloc[0] == 3


def test_num_services_zero_quando_sem_servicos(eng: FeatureEngineer) -> None:
    row = BASE_ROW.copy()
    for col in [
        "phone_service",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
    ]:
        row[col] = "No"
    out = eng.transform(pd.DataFrame([row]))
    assert out["num_services"].iloc[0] == 0


def test_num_services_maximo(eng: FeatureEngineer) -> None:
    row = BASE_ROW.copy()
    for col in [
        "phone_service",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
    ]:
        row[col] = "Yes"
    out = eng.transform(pd.DataFrame([row]))
    assert out["num_services"].iloc[0] == 7


# ── charges_per_month ─────────────────────────────────────────────────────────


def test_charges_per_month_formula(eng: FeatureEngineer) -> None:
    # 65.0 / (24 + 1) = 2.6
    out = eng.transform(pd.DataFrame([BASE_ROW.copy()]))
    assert abs(out["charges_per_month"].iloc[0] - 65.0 / 25) < 1e-3


def test_charges_per_month_tenure_zero(eng: FeatureEngineer) -> None:
    row = BASE_ROW.copy()
    row["tenure_months"] = 0
    out = eng.transform(pd.DataFrame([row]))
    # 65.0 / (0 + 1) = 65.0
    assert abs(out["charges_per_month"].iloc[0] - 65.0) < 1e-3


def test_charges_per_month_nao_negativo(eng: FeatureEngineer) -> None:
    out = eng.transform(pd.DataFrame([BASE_ROW.copy()]))
    assert (out["charges_per_month"] >= 0).all()


# ── is_month_to_month ─────────────────────────────────────────────────────────


def test_is_month_to_month_verdadeiro(eng: FeatureEngineer) -> None:
    out = eng.transform(pd.DataFrame([BASE_ROW.copy()]))
    assert out["is_month_to_month"].iloc[0] == 1


def test_is_month_to_month_falso_para_anual(eng: FeatureEngineer) -> None:
    row = BASE_ROW.copy()
    row["contract"] = "One year"
    out = eng.transform(pd.DataFrame([row]))
    assert out["is_month_to_month"].iloc[0] == 0


def test_is_month_to_month_falso_para_bienal(eng: FeatureEngineer) -> None:
    row = BASE_ROW.copy()
    row["contract"] = "Two year"
    out = eng.transform(pd.DataFrame([row]))
    assert out["is_month_to_month"].iloc[0] == 0


# ── tenure_group ──────────────────────────────────────────────────────────────


def test_tenure_group_novo(eng: FeatureEngineer) -> None:
    row = BASE_ROW.copy()
    row["tenure_months"] = 6
    out = eng.transform(pd.DataFrame([row]))
    assert out["tenure_group"].iloc[0] == "novo"


def test_tenure_group_medio(eng: FeatureEngineer) -> None:
    row = BASE_ROW.copy()
    row["tenure_months"] = 24
    out = eng.transform(pd.DataFrame([row]))
    assert out["tenure_group"].iloc[0] == "medio"


def test_tenure_group_longo(eng: FeatureEngineer) -> None:
    row = BASE_ROW.copy()
    row["tenure_months"] = 60
    out = eng.transform(pd.DataFrame([row]))
    assert out["tenure_group"].iloc[0] == "longo"


def test_tenure_group_limite_12(eng: FeatureEngineer) -> None:
    row = BASE_ROW.copy()
    row["tenure_months"] = 12
    out = eng.transform(pd.DataFrame([row]))
    assert out["tenure_group"].iloc[0] == "novo"


def test_tenure_group_limite_13(eng: FeatureEngineer) -> None:
    row = BASE_ROW.copy()
    row["tenure_months"] = 13
    out = eng.transform(pd.DataFrame([row]))
    assert out["tenure_group"].iloc[0] == "medio"


def test_tenure_group_so_valores_esperados(eng: FeatureEngineer) -> None:
    rows = [BASE_ROW.copy() for _ in range(3)]
    rows[0]["tenure_months"] = 5
    rows[1]["tenure_months"] = 30
    rows[2]["tenure_months"] = 65
    out = eng.transform(pd.DataFrame(rows))
    assert set(out["tenure_group"].unique()).issubset({"novo", "medio", "longo"})


# ── has_security_support ──────────────────────────────────────────────────────


def test_has_security_support_via_security(eng: FeatureEngineer) -> None:
    row = BASE_ROW.copy()
    row["online_security"] = "Yes"
    row["tech_support"] = "No"
    out = eng.transform(pd.DataFrame([row]))
    assert out["has_security_support"].iloc[0] == 1


def test_has_security_support_via_tech(eng: FeatureEngineer) -> None:
    row = BASE_ROW.copy()
    row["online_security"] = "No"
    row["tech_support"] = "Yes"
    out = eng.transform(pd.DataFrame([row]))
    assert out["has_security_support"].iloc[0] == 1


def test_has_security_support_ambos_nao(eng: FeatureEngineer) -> None:
    row = BASE_ROW.copy()
    row["online_security"] = "No"
    row["tech_support"] = "No"
    out = eng.transform(pd.DataFrame([row]))
    assert out["has_security_support"].iloc[0] == 0


# ── is_fiber_optic ────────────────────────────────────────────────────────────


def test_is_fiber_optic_verdadeiro(eng: FeatureEngineer) -> None:
    out = eng.transform(pd.DataFrame([BASE_ROW.copy()]))
    assert out["is_fiber_optic"].iloc[0] == 1


def test_is_fiber_optic_falso_para_dsl(eng: FeatureEngineer) -> None:
    row = BASE_ROW.copy()
    row["internet_service"] = "DSL"
    out = eng.transform(pd.DataFrame([row]))
    assert out["is_fiber_optic"].iloc[0] == 0


def test_is_fiber_optic_falso_sem_internet(eng: FeatureEngineer) -> None:
    row = BASE_ROW.copy()
    row["internet_service"] = "No"
    out = eng.transform(pd.DataFrame([row]))
    assert out["is_fiber_optic"].iloc[0] == 0


# ── validate_output ───────────────────────────────────────────────────────────


def test_validate_output_passa_em_dados_validos(
    eng: FeatureEngineer, df: pd.DataFrame
) -> None:
    out = eng.transform(df)
    eng.validate_output(out)  # não deve levantar exceção


def test_validate_output_falha_sem_feature(
    eng: FeatureEngineer, df: pd.DataFrame
) -> None:
    out = eng.transform(df)
    out = out.drop(columns=["num_services"])
    with pytest.raises(AssertionError):
        eng.validate_output(out)


def test_validate_output_falha_com_nulo(eng: FeatureEngineer, df: pd.DataFrame) -> None:
    out = eng.transform(df)
    out.loc[0, "is_fiber_optic"] = None
    with pytest.raises(AssertionError):
        eng.validate_output(out)


def test_validate_output_falha_binaria_fora_de_range(
    eng: FeatureEngineer, df: pd.DataFrame
) -> None:
    out = eng.transform(df)
    out.loc[0, "is_month_to_month"] = 2
    with pytest.raises(AssertionError):
        eng.validate_output(out)


def test_validate_output_falha_num_services_negativo(
    eng: FeatureEngineer, df: pd.DataFrame
) -> None:
    out = eng.transform(df)
    out.loc[0, "num_services"] = -1
    with pytest.raises(AssertionError):
        eng.validate_output(out)


def test_validate_output_falha_charges_negativo(
    eng: FeatureEngineer, df: pd.DataFrame
) -> None:
    out = eng.transform(df)
    out.loc[0, "charges_per_month"] = -5.0
    with pytest.raises(AssertionError):
        eng.validate_output(out)


def test_validate_output_falha_tenure_group_invalido(
    eng: FeatureEngineer, df: pd.DataFrame
) -> None:
    out = eng.transform(df)
    out.loc[0, "tenure_group"] = "invalido"
    with pytest.raises(AssertionError):
        eng.validate_output(out)


# ── coluna ausente → ValueError ───────────────────────────────────────────────


def test_transform_levanta_value_error_sem_coluna_obrigatoria(
    eng: FeatureEngineer,
) -> None:
    df_incompleto = pd.DataFrame([{"gender": "Female"}])
    with pytest.raises(ValueError, match="colunas obrigatórias ausentes"):
        eng.transform(df_incompleto)


# ── batch de múltiplas linhas ─────────────────────────────────────────────────


def test_transform_batch_preserva_shape(eng: FeatureEngineer) -> None:
    rows = [BASE_ROW.copy() for _ in range(5)]
    df_batch = pd.DataFrame(rows)
    out = eng.transform(df_batch)
    assert out.shape == (5, df_batch.shape[1] + 6)
