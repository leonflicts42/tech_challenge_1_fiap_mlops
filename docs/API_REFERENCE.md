# Churn Telecom API — Referência Técnica

> Documento de manutenção para a API de predição de churn do dataset IBM Telco.  
> Gerado em: 2026-04-25 | Versão da API: 1.0.0

---

## Sumário

1. [Visão Geral](#1-visão-geral)
2. [Estrutura de Arquivos](#2-estrutura-de-arquivos)
3. [Fluxo Completo dos Dados](#3-fluxo-completo-dos-dados)
4. [Regras de Negócio](#4-regras-de-negócio)
5. [Endpoints](#5-endpoints)
6. [Schema de Entrada — ChurnRequest](#6-schema-de-entrada--churnrequest)
7. [Schema de Saída — ChurnResponse](#7-schema-de-saída--churnresponse)
8. [Normalização de Entrada](#8-normalização-de-entrada)
9. [Pipeline de Transformação Interna](#9-pipeline-de-transformação-interna)
10. [Arquitetura do Modelo](#10-arquitetura-do-modelo)
11. [Middleware e Rastreabilidade](#11-middleware-e-rastreabilidade)
12. [Tratamento de Erros](#12-tratamento-de-erros)
13. [Cobertura de Testes](#13-cobertura-de-testes)
14. [Configuração e Artefatos](#14-configuração-e-artefatos)
15. [Como Executar](#15-como-executar)

---

## 1. Visão Geral

A API recebe dados brutos de clientes de telecomunicações e retorna a **probabilidade de churn**, a **classificação binária** e uma **estimativa de custo de negócio**.

O cliente pode enviar o payload de duas formas equivalentes:

| Formato | Campos | Exemplo de chave |
|---------|--------|-----------------|
| **19 features** | Apenas as colunas de entrada do modelo | `"gender"`, `"tenure_months"` |
| **33 features** | Dataset IBM Telco completo (com IDs, targets, leakage) | `"Gender"`, `"Tenure Months"`, `"Churn Value"` |

Em ambos os casos a resposta é idêntica. Colunas extras (IDs, targets, leakage) são **silenciosamente descartadas** pelo Pydantic. Nomes de colunas com espaços, maiúsculas ou camelCase são **automaticamente normalizados** para snake_case antes da validação.

> **Por que `Churn Label` / `Churn Value` não são campos obrigatórios?**  
> Porque são o **target** — a variável que o modelo está sendo contratado para prever. Incluí-los no input seria data leakage. O modelo produz justamente esses valores como saída.

---

## 2. Estrutura de Arquivos

```
src/
├── main.py                  # App FastAPI + lifespan + exception handlers
├── config.py                # Constantes globais, paths, custos, utilitários
├── api/
│   ├── router.py            # Endpoints GET /health e POST /predict
│   ├── schemas.py           # ChurnRequest e ChurnResponse (Pydantic v2)
│   ├── predictor.py         # ChurnPredictor: orquestra o pipeline de inferência
│   └── middleware.py        # LatencyMiddleware: X-Request-ID e X-Process-Time
├── data/
│   ├── preprocessing.py     # SemanticNormalizer (sklearn transformer)
│   └── features.py          # FeatureEngineer (sklearn transformer, 6 features)
└── models/
    ├── mlp.py               # ChurnMLP com BatchNorm (treinamento/baseline)
    └── mlp2.py              # ChurnMLPv2 + ChurnMLPInference com LayerNorm (produção)

tests/
├── conftest.py              # Fixtures globais: valid_payload, raw_payload_33, test_client
├── test_api.py              # Testes dos endpoints HTTP
├── test_schema.py           # Testes de validação Pydantic (sem API)
├── test_smoke.py            # Smoke tests: API sobe e responde
├── test_mlp.py              # Testes da arquitetura ChurnMLP
├── test_main.py             # Testes do app FastAPI
└── test_etapa2.py           # Testes de preprocessing e feature engineering

models/
├── preprocessor.pkl         # ColumnTransformer sklearn serializado
└── best_model_mlp.pt        # Pesos da ChurnMLPv2 (LayerNorm)

reports/json/
└── winner_model_report.json # Threshold ótimo e métricas do modelo vencedor
```

---

## 3. Fluxo Completo dos Dados

```
Cliente (19 ou 33 colunas, qualquer casing)
        │
        ▼
┌─────────────────────────────────────────┐
│  LatencyMiddleware                       │
│  • Gera X-Request-ID (UUID)             │
│  • Inicia timer                         │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  ChurnRequest (Pydantic v2)             │
│  model_validator(mode="before"):        │
│  1. Normaliza chaves → snake_case       │
│     "Tenure Months" → tenure_months     │
│     "Monthly Charge" → monthly_charges  │  (alias)
│  2. Normaliza valores → forma canônica  │
│     "yes" / "YES" → "Yes"              │
│     "dsl" / "DSL" → "DSL"             │
│     "month-to-month" → "Month-to-month"│
│  3. Pydantic descarta campos extras     │
│     CustomerID, Churn Value, CLTV…      │
│  4. Valida domínios (Literal types)     │
│     gender ∈ {Male, Female}             │
│     tenure_months ∈ [0, 72]            │
│     contract ∈ {Month-to-month, ...}   │
└─────────────────────────────────────────┘
        │ 422 se domínio inválido
        ▼
┌─────────────────────────────────────────┐
│  ChurnPredictor.predict()               │
│                                         │
│  [1] model_dump() → DataFrame (1 linha) │
│                                         │
│  [2] Normaliza multiple_lines:          │
│      "No phone service" → "No"          │
│                                         │
│  [3] SemanticNormalizer.transform()     │
│      "No internet service" → "No"       │
│      (6 colunas de serviços)            │
│      Corrige: sem internet + serviço    │
│      ativo → força "No"                 │
│                                         │
│  [4] FeatureEngineer.transform()        │
│      Cria 6 features derivadas:         │
│      • num_services                     │
│      • charges_per_month               │
│      • is_month_to_month               │
│      • tenure_group                     │
│      • has_security_support            │
│      • is_fiber_optic                  │
│      DataFrame: 19 cols → 25 cols       │
│                                         │
│  [5] preprocessor.pkl.transform()       │
│      ColumnTransformer sklearn:         │
│      • log1p + StandardScaler (num)     │
│      • OrdinalEncoder (binário)         │
│      • OneHotEncoder (categórico)       │
│      • passthrough (flags binárias)     │
│      Saída: array float64 (1, 30)       │
│                                         │
│  [6] array → tensor float32             │
│      (converte sparse se necessário)    │
│                                         │
│  [7] ChurnMLPv2.forward() → logit       │
│      Input(30) → [Linear→LayerNorm      │
│      →ReLU→Dropout]×2 → Linear(1)      │
│                                         │
│  [8] sigmoid(logit) → probabilidade     │
│                                         │
│  [9] prob ≥ threshold → label + custo   │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  ChurnResponse                          │
│  • churn_probability: float [0.0, 1.0] │
│  • churn_label: "churn" | "no_churn"   │
│  • threshold_used: float               │
│  • cost_estimate_brl: float            │
│  • model_version: str                  │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  LatencyMiddleware (response)           │
│  • Injeta X-Request-ID no header        │
│  • Injeta X-Process-Time no header      │
│  • Loga: método, path, status, latência │
└─────────────────────────────────────────┘
        │
        ▼
      Cliente (200 OK)
```

---

## 4. Regras de Negócio

### 4.1 Threshold de Decisão

O threshold de decisão (limiar de classificação) **não é fixo em 0.5**. É carregado na seguinte ordem de prioridade:

1. **Arquivo Optuna** (`models/optuna_best_params_*.json`) — gerado durante tuning em runtime
2. **Relatório do modelo vencedor** (`reports/json/winner_model_report.json`) — campo `test_metrics.threshold`
3. **Padrão** (`0.50`) — usado apenas se nenhum dos anteriores existir

O threshold atual do modelo vencedor é **0.16**, resultado de otimização Optuna que maximiza o score de custo de negócio (não acurácia nem F1).

### 4.2 Estimativa de Custo

| Predição | Custo | Justificativa |
|----------|-------|---------------|
| `churn` (positivo) | R$ 73,52 | Custo da campanha de retenção (COST_FP) |
| `no_churn` (negativo) | R$ 0,00 | Nenhuma ação tomada |

A assimetria intencional entre os custos guia o threshold para baixo (0.16): um falso negativo (cliente que churn não foi detectado) custa R$ 2.845,00 (CLV perdido), enquanto um falso positivo (campanha de retenção desnecessária) custa apenas R$ 73,52. Vale errar para o lado da retenção.

### 4.3 Colunas Aceitas vs. Rejeitadas

| Tipo | Exemplos | Comportamento |
|------|---------|---------------|
| Features de entrada (19) | `gender`, `tenure_months`, `contract` | **Obrigatórias** — 422 se ausentes |
| Identificadores | `CustomerID`, `Count`, `Zip Code` | Silenciosamente descartadas |
| Leakage / targets | `Churn Label`, `Churn Value`, `Churn Score`, `CLTV` | Silenciosamente descartadas |
| Metadados geográficos | `City`, `State`, `Latitude`, `Longitude` | Silenciosamente descartadas |

### 4.4 Normalização de Valores de String

A API aceita qualquer casing nos valores categóricos:

| Enviado | Normalizado para |
|---------|-----------------|
| `"yes"`, `"YES"`, `"Yes"` | `"Yes"` |
| `"no"`, `"NO"` | `"No"` |
| `"male"`, `"MALE"` | `"Male"` |
| `"female"`, `"FEMALE"` | `"Female"` |
| `"dsl"`, `"DSL"` | `"DSL"` |
| `"fiber optic"`, `"FIBER OPTIC"` | `"Fiber optic"` |
| `"month-to-month"`, `"MONTH-TO-MONTH"` | `"Month-to-month"` |
| `"one year"`, `"ONE YEAR"` | `"One year"` |
| `"two year"`, `"TWO YEAR"` | `"Two year"` |
| `"electronic check"` | `"Electronic check"` |
| `"mailed check"` | `"Mailed check"` |
| `"bank transfer (automatic)"` | `"Bank transfer (automatic)"` |
| `"credit card (automatic)"` | `"Credit card (automatic)"` |

---

## 5. Endpoints

### `GET /api/v1/health`

Liveness probe. Verifica se a API está ativa e os artefatos carregados.

**Resposta 200:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "preprocessor_loaded": true,
  "model_version": "best_model_mlp",
  "threshold": 0.16
}
```

**Resposta 503** (artefatos não carregados no startup):
```json
{
  "status": "unavailable",
  "model_loaded": false,
  "preprocessor_loaded": false,
  "detail": "Artefatos de ML não carregados — verifique os logs de startup."
}
```

---

### `POST /api/v1/predict`

Predição de churn para um cliente.

**Resposta 200:**
```json
{
  "churn_probability": 0.847321,
  "churn_label": "churn",
  "threshold_used": 0.16,
  "cost_estimate_brl": 73.52,
  "model_version": "best_model_mlp"
}
```

**Resposta 422** (campo inválido ou fora do domínio):
```json
{
  "error": "validation_error",
  "detail": [
    {
      "field": "tenure_months",
      "message": "Input should be less than or equal to 72",
      "received": 99
    }
  ],
  "request_id": "uuid-da-requisição"
}
```

**Resposta 503** (modelo não carregado):
```json
{
  "detail": "Modelo não disponível. Verifique os logs de startup."
}
```

---

## 6. Schema de Entrada — ChurnRequest

### Campos obrigatórios (19 no total)

#### Demográficas

| Campo | Tipo | Valores aceitos |
|-------|------|----------------|
| `gender` | string | `"Male"`, `"Female"` |
| `senior_citizen` | string | `"Yes"`, `"No"` |
| `partner` | string | `"Yes"`, `"No"` |
| `dependents` | string | `"Yes"`, `"No"` |

#### Relacionamento com a operadora

| Campo | Tipo | Restrições |
|-------|------|-----------|
| `tenure_months` | int | `[0, 72]` |
| `contract` | string | `"Month-to-month"`, `"One year"`, `"Two year"` |
| `paperless_billing` | string | `"Yes"`, `"No"` |
| `payment_method` | string | `"Electronic check"`, `"Mailed check"`, `"Bank transfer (automatic)"`, `"Credit card (automatic)"` |

#### Cobrança

| Campo | Tipo | Restrições |
|-------|------|-----------|
| `monthly_charges` | float | `[0.0, 200.0]` |
| `total_charges` | float | `[0.0, 10000.0]` |

#### Serviços de telefonia

| Campo | Tipo | Valores aceitos |
|-------|------|----------------|
| `phone_service` | string | `"Yes"`, `"No"` |
| `multiple_lines` | string | `"Yes"`, `"No"`, `"No phone service"` |

#### Serviços de internet

| Campo | Tipo | Valores aceitos |
|-------|------|----------------|
| `internet_service` | string | `"DSL"`, `"Fiber optic"`, `"No"` |
| `online_security` | string | `"Yes"`, `"No"`, `"No internet service"` |
| `online_backup` | string | `"Yes"`, `"No"`, `"No internet service"` |
| `device_protection` | string | `"Yes"`, `"No"`, `"No internet service"` |
| `tech_support` | string | `"Yes"`, `"No"`, `"No internet service"` |
| `streaming_tv` | string | `"Yes"`, `"No"`, `"No internet service"` |
| `streaming_movies` | string | `"Yes"`, `"No"`, `"No internet service"` |

### Alias de chave

O campo `monthly_charges` também é aceito como `"Monthly Charge"` (sem o 's'), que é o nome original da coluna no dataset IBM Telco.

---

## 7. Schema de Saída — ChurnResponse

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `churn_probability` | float `[0.0, 1.0]` | Probabilidade de churn prevista pelo modelo |
| `churn_label` | `"churn"` \| `"no_churn"` | Classificação binária aplicando o threshold |
| `threshold_used` | float | Limiar de decisão utilizado (carregado do Optuna ou winner_model_report) |
| `cost_estimate_brl` | float | R$ 73,52 se churn, R$ 0,00 se no_churn |
| `model_version` | string | Nome do arquivo de pesos carregado (ex: `"best_model_mlp"`) |

---

## 8. Normalização de Entrada

Implementada em `src/api/schemas.py` via `model_validator(mode="before")`.

### Normalização de chaves

Qualquer formato de nome de coluna é aceito:

| Enviado | Normalizado para |
|---------|-----------------|
| `"Tenure Months"` | `tenure_months` |
| `"TenureMonths"` | `tenure_months` |
| `"GENDER"` | `gender` |
| `"Senior Citizen"` | `senior_citizen` |
| `"Monthly Charge"` | `monthly_charges` (via `_KEY_ALIASES`) |
| `"CustomerID"` | `customerid` (descartado pelo Pydantic) |

A função `to_snake_case()` em `config.py` converte qualquer string para snake_case. O dicionário `_KEY_ALIASES` resolve discrepâncias específicas do dataset original.

### Normalização de valores

O dicionário `_CANONICAL` mapeia qualquer variação de casing para o valor canônico aceito pelo Pydantic e pelo preprocessor. A comparação é feita em lowercase após strip de espaços.

---

## 9. Pipeline de Transformação Interna

### 9.1 SemanticNormalizer (`src/data/preprocessing.py`)

Sklearn transformer sem estado (fit é no-op). Aplica duas operações:

**Operação 1 — Redução de cardinalidade:**
```
"No internet service" → "No"
```
Afeta as 6 colunas: `online_security`, `online_backup`, `device_protection`, `tech_support`, `streaming_tv`, `streaming_movies`.

**Operação 2 — Correção de inconsistências lógicas:**
Se `internet_service = "No"`, todos os serviços dependentes são forçados para `"No"`, independente do que foi enviado.

### 9.2 FeatureEngineer (`src/data/features.py`)

Sklearn transformer sem estado. Adiciona 6 colunas ao DataFrame:

| Feature | Tipo | Lógica | Correlação com churn |
|---------|------|--------|----------------------|
| `num_services` | int [0–7] | Contagem de serviços ativos | -0.08 |
| `charges_per_month` | float | `monthly_charges / (tenure_months + 1)` | +0.41 |
| `is_month_to_month` | int {0,1} | `contract == "Month-to-month"` | +0.41 (Cramer's V) |
| `tenure_group` | str | `"novo"` (0–12m), `"medio"` (13–48m), `"longo"` (>48m) | alta não-linearidade |
| `has_security_support` | int {0,1} | `online_security == "Yes" OR tech_support == "Yes"` | -0.18 |
| `is_fiber_optic` | int {0,1} | `internet_service == "Fiber optic"` | +0.31 |

O DataFrame passa de 19 para 25 colunas após esta etapa.

### 9.3 ColumnTransformer — preprocessor.pkl

Aplica encoding e scaling às 25 colunas, gerando um array `float64` com shape `(1, 30)`:

| Pipeline | Colunas | Transformação |
|----------|---------|--------------|
| `num` | tenure, charges, total, num_services, charges_per_month | log1p → StandardScaler |
| `bin` | gender, senior_citizen, partner, dependents, etc. | OrdinalEncoder |
| `ohe` | internet_service, contract, payment_method, tenure_group | OneHotEncoder |
| `pass` | is_month_to_month, has_security_support, is_fiber_optic | passthrough |

> **Nota de compatibilidade:** O `preprocessor.pkl` foi serializado com sklearn 1.8.0. Usar versões anteriores gera `InconsistentVersionWarning`. Execute `uv sync` para garantir a versão correta.

> **Patch de compatibilidade sklearn:** `predictor.py` aplica um monkey-patch em `sklearn.utils._encode._check_unknown` para contornar um bug onde `np.isnan()` falha em arrays de strings (`dtype=object`). O patch faz fallback para `np.setdiff1d` + `np.isin` quando `TypeError` é levantado.

---

## 10. Arquitetura do Modelo

**ChurnMLPv2** com LayerNorm (sem BatchNorm), compatível com `best_model_mlp.pt`:

```
Input (30)
  → Linear(30 → 128) → LayerNorm(128) → ReLU → Dropout(0.15)
  → Linear(128 → 64) → LayerNorm(64)  → ReLU → Dropout(0.15)
  → Linear(64 → 1)
  → [sigmoid aplicado externamente na inferência]
```

**Por LayerNorm em vez de BatchNorm?**  
BatchNorm requer batch size > 1 para calcular média/variância. Na inferência batch size = 1 (uma predição por vez), o que torna BatchNorm instável. LayerNorm opera por amostra, é estável com qualquer batch size.

O modelo salvo em `best_model_mlp.pt` usa LayerNorm (ausência de `running_mean`/`running_var` no state_dict confirma isso).

---

## 11. Middleware e Rastreabilidade

`LatencyMiddleware` em `src/api/middleware.py` intercepta **todas** as requisições antes dos handlers.

**Headers adicionados à resposta:**

| Header | Valor | Uso |
|--------|-------|-----|
| `X-Request-ID` | UUID v4 único por requisição | Correlacionar logs de uma requisição específica |
| `X-Process-Time` | Ex: `"12.45ms"` | Monitoramento de latência |

**Atributo em `request.state`:**
- `request.state.request_id` — o mesmo UUID do header, disponível para exception handlers e endpoints logarem junto.

**Log gerado por requisição:**
```
request | POST /api/v1/predict | 200 | 14.2ms | request_id=3f4a...
```

---

## 12. Tratamento de Erros

| Cenário | Status | Origem |
|---------|--------|--------|
| Campo faltando no payload | 422 | Pydantic — campo obrigatório ausente |
| Valor fora do domínio | 422 | Pydantic — Literal type |
| Valor numérico fora do range | 422 | Pydantic — Field(ge=, le=) |
| JSON malformado | 422 | FastAPI — JSON decode error |
| Artefatos não carregados no startup | 503 | router.py — `predictor is None` |
| Shape inesperado após preprocessamento | 422 | predictor.py — ValueError |
| Erro inesperado no pipeline | 500 | router.py — Exception catch-all |

Todos os erros 422 retornam o mesmo formato estruturado com `request_id` para facilitar diagnóstico:
```json
{
  "error": "validation_error",
  "detail": [{"field": "...", "message": "...", "received": "..."}],
  "request_id": "uuid"
}
```

---

## 13. Cobertura de Testes

Total: **73 testes** — todos passando.

### `tests/conftest.py` — Fixtures

| Fixture | Descrição |
|---------|-----------|
| `valid_payload` | Dict com as 19 features obrigatórias em snake_case |
| `raw_payload_33` | Dict com as 33 colunas do dataset IBM Telco (inclui IDs, leakage, lowercase) |
| `test_client` | `TestClient` FastAPI com `ChurnPredictor` mockado em `app.state.predictor` |

### `tests/test_schema.py` — Validação Pydantic (sem API)

| Teste | O que verifica |
|-------|---------------|
| `test_valid_payload_creates_instance` | Payload válido instancia `ChurnRequest` corretamente |
| `test_rejeita_gender_invalido` | `gender="Other"` → `ValidationError` |
| `test_rejeita_tenure_acima_de_72` | `tenure_months=73` → `ValidationError` |
| `test_rejeita_monthly_charges_negativo` | `monthly_charges=-1.0` → `ValidationError` |
| `test_rejeita_contract_invalido` | `contract="Weekly"` → `ValidationError` |
| `test_rejeita_payment_method_invalido` | `payment_method="Crypto"` → `ValidationError` |
| `test_rejeita_senior_citizen_invalido` | `senior_citizen="Maybe"` → `ValidationError` |
| `test_rejeita_internet_service_invalido` | `internet_service="5G"` → `ValidationError` |
| `test_aceita_no_internet_service_em_servicos` | `online_security="No internet service"` → aceito |
| `test_model_dump_tem_19_campos` | `model_dump()` retorna exatamente 19 campos |
| `test_aceita_tenure_zero` | `tenure_months=0` → aceito |
| `test_aceita_total_charges_zero` | `total_charges=0.0` → aceito |
| `test_response_valida` | `ChurnResponse` válida instancia corretamente |
| `test_rejeita_probabilidade_acima_de_1` | `churn_probability=1.5` → `ValidationError` |
| `test_rejeita_label_invalido` | `churn_label="maybe"` → `ValidationError` |

### `tests/test_api.py` — Endpoints HTTP

#### `TestHealthEndpoint`

| Teste | O que verifica |
|-------|---------------|
| `test_health_200_com_mocks` | `/health` retorna 200 com predictor mockado |
| `test_health_retorna_ok` | `status == "ok"` |
| `test_health_503_sem_predictor` | `predictor=None` → 503 |
| `test_health_model_loaded_false_quando_model_none` | `_model=None` → `model_loaded=False` no corpo |
| `test_health_preprocessor_loaded_false_quando_preprocessor_none` | `_preprocessor=None` → `preprocessor_loaded=False` |

#### `TestPredictEndpoint`

| Teste | O que verifica |
|-------|---------------|
| `test_payload_valido_retorna_200` | Payload válido → 200 |
| `test_schema_de_resposta_correto` | Resposta contém os 5 campos obrigatórios |
| `test_churn_label_valido` | `churn_label ∈ {"churn", "no_churn"}` |
| `test_gender_invalido_retorna_422` | `gender="Other"` → 422 |
| `test_campo_faltando_retorna_422` | `tenure_months` ausente → 422 |
| `test_tenure_acima_de_72_retorna_422` | `tenure_months=99` → 422 |
| `test_resposta_tem_header_request_id` | Header `X-Request-ID` presente na resposta |
| `test_predict_503_sem_predictor` | `predictor=None` → 503 |
| `test_predict_chama_predictor_uma_vez` | `predictor.predict` chamado exatamente 1x |

#### `TestPayloadVariants`

| Teste | O que verifica |
|-------|---------------|
| `test_19_features_retorna_200_e_predicao_valida` | 19 campos → 200 + schema correto |
| `test_33_features_retorna_200_e_predicao_valida` | 33 campos → 200 + schema correto |
| `test_33_features_descarta_colunas_extras` | CustomerID, Churn Label, CLTV ignorados silenciosamente |
| `test_valores_em_lowercase_sao_normalizados` | `"female"`, `"dsl"`, `"month-to-month"` → aceitos |
| `test_valores_em_uppercase_sao_normalizados` | `"MALE"`, `"DSL"`, `"ONE YEAR"` → aceitos |
| `test_chaves_em_title_case_sao_normalizadas` | `"Tenure Months"`, `"Monthly Charge"` → snake_case |
| `test_33_e_19_features_chamam_predictor_uma_vez_cada` | 2 requests → `call_count == 2` |

### `tests/test_smoke.py` — Smoke tests

| Teste | O que verifica |
|-------|---------------|
| `test_app_starts` | TestClient instancia sem erro |
| `test_health_returns_200` | `/health` responde 200 |
| `test_health_schema` | Campos obrigatórios presentes e com tipos corretos |
| `test_health_with_mocks_is_healthy` | `status="ok"`, `model_loaded=true`, `preprocessor_loaded=true` |
| `test_predict_endpoint_exists` | POST `/predict` responde 200 |
| `test_request_id_header_present` | Header `X-Request-ID` presente |
| `test_latency_header_present` | Header `X-Process-Time` presente |

### `tests/test_mlp.py` — Arquitetura ChurnMLP

| Teste | O que verifica |
|-------|---------------|
| `test_mlp_constroi_com_atributos_corretos` | Atributos e camada de saída corretos |
| `test_mlp_conta_parametros_treinaveis` | `count_parameters()` == soma manual |
| `test_forward_retorna_shape_correto` | Output shape `(batch_size, 1)`, dtype `float32` |
| `test_forward_retorna_logits_nao_probabilidades` | Valores fora de [0,1] confirmam ausência de sigmoid |
| `test_backward_produz_gradientes` | Todos os parâmetros têm gradientes finitos |
| `test_construtor_rejeita_argumentos_invalidos` | in_features=0, hidden_dims=[], dropout=1.0, etc. → ValueError |
| `test_factory_e_reprodutivel_com_mesma_seed` | Mesma seed → pesos idênticos |
| `test_train_eval_modes_diferem` | eval() determinístico; train() estocástico (Dropout) |

---

## 14. Configuração e Artefatos

### Variáveis de ambiente

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `MODEL_PATH` | `models/best_model_mlp.pt` | Caminho para os pesos da MLP |
| `PREPROCESSOR_PATH` | `models/preprocessor.pkl` | Caminho para o ColumnTransformer |

### Artefatos obrigatórios

```
models/
├── preprocessor.pkl       # Gerado pelo notebook 2_vab_preprocessing
└── best_model_mlp.pt      # Gerado pelo notebook de treinamento MLP
```

Se qualquer artefato estiver ausente na subida, `app.state.predictor = None` e todos os endpoints retornam 503. A aplicação **não quebra** — ela sobe e fica em estado degradado com logs de erro claros.

### Artefatos opcionais

```
reports/json/
├── winner_model_report.json        # Lido para extrair threshold ótimo
└── optuna_best_params_*.json       # Gerado pelo Optuna em runtime (prioridade máxima)
```

---

## 15. Como Executar

### Desenvolvimento

```bash
# Instalar dependências
uv sync

# Subir a API com hot-reload
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Acessar documentação interativa
# http://localhost:8000/docs
```

### Testes

```bash
# Rodar todos os testes
uv run pytest tests/ -v

# Rodar com relatório de cobertura
uv run pytest --cov=src -v

# Rodar apenas os testes de schema (sem carregar artefatos)
uv run pytest tests/test_schema.py -v
```

### Lint e formatação

```bash
uv run ruff check .
uv run ruff format --check .
```

### Exemplo de requisição — 19 features

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "total_charges": 358.20
  }'
```

### Exemplo de requisição — 33 features (dataset IBM Telco completo)

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CustomerID": "3668-QPYBK",
    "Count": 1,
    "Country": "United States",
    "State": "California",
    "City": "Los Angeles",
    "Zip Code": 90003,
    "Gender": "Male",
    "Senior Citizen": "No",
    "Partner": "No",
    "Dependents": "No",
    "Tenure Months": 2,
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Internet Service": "DSL",
    "Online Security": "Yes",
    "Online Backup": "Yes",
    "Device Protection": "No",
    "Tech Support": "No",
    "Streaming TV": "No",
    "Streaming Movies": "No",
    "Contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Mailed check",
    "Monthly Charge": 53.85,
    "Total Charges": 108.15,
    "Churn Label": "Yes",
    "Churn Value": 1,
    "Churn Score": 86,
    "CLTV": 3239,
    "Churn Reason": "Competitor made better offer"
  }'
```

**Resposta esperada:**
```json
{
  "churn_probability": 0.847321,
  "churn_label": "churn",
  "threshold_used": 0.16,
  "cost_estimate_brl": 73.52,
  "model_version": "best_model_mlp"
}
```
