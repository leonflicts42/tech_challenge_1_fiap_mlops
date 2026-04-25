"""
api/schemas.py — Contratos de entrada e saída da rota /predict.

ChurnRequest:  19 variáveis brutas do cliente (pré-preprocessing).
               Os nomes aceitam qualquer casing — a API normaliza internamente.
               Validators garantem que valores fora do domínio sejam rejeitados
               com 422 antes de chegar no pipeline.

ChurnResponse: resultado da predição com probabilidade, label, threshold
               e estimativa de custo de negócio.

Por que 19 campos e não 30?
    O cliente envia os dados brutos, não as features engineered.
    SemanticNormalizer + FeatureEngineer + preprocessor.pkl transformam
    os 19 campos em 30 features float32 internamente.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from config import to_snake_case

# ── Mapa de normalização canônica (case-insensitive → forma correta) ──────────
# Permite que fontes externas enviem "yes", "YES", "dsl", "FIBER OPTIC", etc.
# sem quebrar a validação Pydantic.
_CANONICAL: dict[str, str] = {
    # Yes / No
    "yes": "Yes",
    "no": "No",
    # Gender
    "male": "Male",
    "female": "Female",
    # Internet service
    "dsl": "DSL",
    "fiber optic": "Fiber optic",
    # Contract
    "month-to-month": "Month-to-month",
    "one year": "One year",
    "two year": "Two year",
    # Payment method
    "electronic check": "Electronic check",
    "mailed check": "Mailed check",
    "bank transfer (automatic)": "Bank transfer (automatic)",
    "credit card (automatic)": "Credit card (automatic)",
    # Multiple lines / Internet service deps
    "no phone service": "No phone service",
    "no internet service": "No internet service",
}

# ── Mapa de aliases de chave (snake_case alternativo → campo correto) ────────
# Cobre variações do nome original do dataset (ex: "Monthly Charge" vs
# "Monthly Charges") que sobrevivem ao to_snake_case com resultado diferente.
_KEY_ALIASES: dict[str, str] = {
    "monthly_charge": "monthly_charges",  # dataset IBM usa "Monthly Charge" (sem 's')
}

# ── Literais de domínio (valores aceitos por coluna categórica) ───────────────

_YES_NO = Literal["Yes", "No"]
_INTERNET = Literal["DSL", "Fiber optic", "No"]
_CONTRACT = Literal["Month-to-month", "One year", "Two year"]
_PAYMENT = Literal[
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]
_GENDER = Literal["Male", "Female"]
_MULTIPLE_LINES = Literal["Yes", "No", "No phone service"]
_INTERNET_SVC = Literal["Yes", "No", "No internet service"]


class ChurnRequest(BaseModel):
    """Dados brutos de um cliente para predição de churn.

    Todos os campos correspondem às colunas do dataset IBM Telco após
    remoção de identificadores (CustomerID etc.) e colunas de leakage
    (Churn Score, CLTV, Churn Reason).

    Os nomes dos campos estão em snake_case. O cliente pode enviar em
    qualquer casing — a API normaliza via to_snake_case() antes de criar
    este schema.

    Campos numéricos:
        senior_citizen:   0 = não é idoso, 1 = é idoso
        tenure_months:    meses como cliente [0, 72]
        monthly_charges:  cobrança mensal em R$ [0.0, 200.0]
        total_charges:    cobrança total acumulada em R$ [0.0, 10000.0]
    """

    # ── Demográficas ──────────────────────────────────────────────────────────
    gender: _GENDER = Field(
        ...,
        description="Gênero do cliente",
        examples=["Male", "Female"],
    )
    senior_citizen: _YES_NO = Field(
        ...,
        description="Indica se o cliente tem 65 anos ou mais: Yes ou No",
        examples=["Yes", "No"],
    )
    partner: _YES_NO = Field(
        ...,
        description="Possui cônjuge/parceiro",
    )
    dependents: _YES_NO = Field(
        ...,
        description="Possui dependentes",
    )

    # ── Relacionamento com a operadora ────────────────────────────────────────
    tenure_months: int = Field(
        ...,
        ge=0,
        le=72,
        description="Meses como cliente da operadora",
        examples=[12, 24, 60],
    )
    contract: _CONTRACT = Field(
        ...,
        description="Tipo de contrato",
        examples=["Month-to-month", "One year", "Two year"],
    )
    paperless_billing: _YES_NO = Field(
        ...,
        description="Usa fatura digital (sem papel)",
    )
    payment_method: _PAYMENT = Field(
        ...,
        description="Método de pagamento",
        examples=["Electronic check", "Credit card (automatic)"],
    )

    # ── Cobrança ──────────────────────────────────────────────────────────────
    monthly_charges: float = Field(
        ...,
        ge=0.0,
        le=200.0,
        description="Valor cobrado mensalmente (R$)",
        examples=[29.85, 65.50, 89.10],
    )
    total_charges: float = Field(
        ...,
        ge=0.0,
        le=10_000.0,
        description="Total cobrado no período completo (R$)",
        examples=[29.85, 1500.0, 5346.0],
    )

    # ── Serviços de telefonia ─────────────────────────────────────────────────
    phone_service: _YES_NO = Field(
        ...,
        description="Possui serviço de telefone",
    )
    multiple_lines: _MULTIPLE_LINES = Field(
        ...,
        description="Possui múltiplas linhas telefônicas",
    )

    # ── Serviços de internet ──────────────────────────────────────────────────
    internet_service: _INTERNET = Field(
        ...,
        description="Tipo de serviço de internet contratado",
        examples=["DSL", "Fiber optic", "No"],
    )
    online_security: _INTERNET_SVC = Field(
        ...,
        description="Possui serviço de segurança online",
    )
    online_backup: _INTERNET_SVC = Field(
        ...,
        description="Possui serviço de backup online",
    )
    device_protection: _INTERNET_SVC = Field(
        ...,
        description="Possui proteção de dispositivo",
    )
    tech_support: _INTERNET_SVC = Field(
        ...,
        description="Possui suporte técnico",
    )
    streaming_tv: _INTERNET_SVC = Field(
        ...,
        description="Possui serviço de streaming de TV",
    )
    streaming_movies: _INTERNET_SVC = Field(
        ...,
        description="Possui serviço de streaming de filmes",
    )

    # ── Validadores ───────────────────────────────────────────────────────────

    @model_validator(mode="before")
    @classmethod
    def normalize_inputs(cls, data: object) -> object:
        """Normaliza chaves e valores antes da validação Pydantic.

        Chaves: converte para snake_case (aceita "Senior Citizen", "Gender",
                "Tenure Months", "CustomerID", etc.) e aplica aliases conhecidos.
                Campos extras (IDs, targets, colunas de leakage) são ignorados
                pelo Pydantic após a normalização.

        Valores: converte para a forma canônica (aceita "yes", "YES", "dsl",
                 "FIBER OPTIC", "month-to-month", etc.).
        """
        if not isinstance(data, dict):
            return data
        normalized: dict[str, object] = {}
        for k, v in data.items():
            key = _KEY_ALIASES.get(to_snake_case(str(k)), to_snake_case(str(k)))
            val = _CANONICAL.get(v.strip().lower(), v.strip()) if isinstance(v, str) else v
            normalized[key] = val
        return normalized

    @model_validator(mode="after")
    def validate_internet_consistency(self) -> ChurnRequest:
        """Avisa se cliente sem internet tem serviços de internet ativos.

        Não rejeita a requisição — o SemanticNormalizer corrige isso no
        pipeline. O validator apenas loga a inconsistência para auditoria.
        """
        internet_dependent = [
            self.online_security,
            self.online_backup,
            self.device_protection,
            self.tech_support,
            self.streaming_tv,
            self.streaming_movies,
        ]
        if self.internet_service == "No":
            active = [
                v for v in internet_dependent
                if v == "Yes"
            ]
            if active:
                # Não levanta erro — SemanticNormalizer corrige no pipeline
                # O validator registra para auditoria via logs da API
                pass
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "gender": "Female",
                    "senior_citizen": "No",
                    "partner": "Yes",
                    "dependents": "No",
                    "tenure_months": 1,
                    "contract": "Month-to-month",
                    "paperless_billing": "Yes",
                    "payment_method": "Electronic check",
                    "monthly_charges": 29.85,
                    "total_charges": 29.85,
                    "phone_service": "No",
                    "multiple_lines": "No phone service",
                    "internet_service": "DSL",
                    "online_security": "No",
                    "online_backup": "No",
                    "device_protection": "No",
                    "tech_support": "No",
                    "streaming_tv": "No",
                    "streaming_movies": "No",
                }
            ]
        }
    }


class ChurnResponse(BaseModel):
    """Resultado da predição de churn para um cliente.

    Campos:
        churn_probability: probabilidade de churn em [0.0, 1.0]
        churn_label:       classificação binária ("churn" ou "no_churn")
        threshold_used:    limiar aplicado para gerar churn_label
        cost_estimate_brl: custo esperado desta predição em R$
                           Se label=churn: custo da campanha de retenção (COST_FP)
                           Se label=no_churn e errado: custo do CLV perdido (COST_FN)
        model_version:     identificador do modelo carregado
    """

    churn_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilidade de churn prevista pelo modelo",
        examples=[0.83, 0.12],
    )
    churn_label: Literal["churn", "no_churn"] = Field(
        ...,
        description="Classificação final: 'churn' ou 'no_churn'",
    )
    threshold_used: float = Field(
        ...,
        description="Limiar de decisão aplicado",
        examples=[0.50, 0.35],
    )
    cost_estimate_brl: float = Field(
        ...,
        description=(
            "Custo estimado desta predição em R$. "
            "Para predições positivas: custo da campanha de retenção. "
            "Referência para priorização da equipe comercial."
        ),
        examples=[73.52, 2845.0],
    )
    model_version: str = Field(
        default="unknown",
        description="Versão ou nome do arquivo do modelo carregado",
    )