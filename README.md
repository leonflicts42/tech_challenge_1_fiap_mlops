# Churn Prediction — MLOps End-to-End

**Tech Challenge Fase 1 — Pós-Graduação em MLOps · FIAP**

Pipeline completo de Machine Learning para predição de churn em uma operadora de telecomunicações — do entendimento do negócio ao modelo servido via API, com rastreamento de experimentos, testes automatizados e containerização.

---

## Contexto de Negócio

Uma operadora de telecomunicações perde **26,5% dos seus clientes** anualmente, destruindo ~**US$ 3,79 M** em receita futura contratada e não realizada. A diretoria precisa de um modelo preditivo capaz de identificar, com antecedência, quais clientes têm risco de cancelar — permitindo que a equipe de retenção aja antes que o cancelamento ocorra.

### Assimetria de Custo

O ponto central do problema é a **diferença brutal de custo entre os tipos de erro**:

| Erro | Significado | Custo |
|------|-------------|-------|
| **Falso Negativo (FN)** | Churner real não detectado → cliente perdido | **US$ 2.845** (CLV médio residual) |
| **Falso Positivo (FP)** | Cliente retido sem necessidade → ação desnecessária | **US$ 73,52** (desconto de 10% oferecido) |
| **Razão FN/FP** | Perder um cliente custa 38,7× mais do que uma ação desnecessária | **38,7×** |

Por isso, o threshold de decisão **não foi escolhido pelo F1**, e sim por uma **função de valor de negócio** que otimiza o retorno financeiro real sob a restrição de Recall ≥ 70% (SLO operacional).

### Recuperação Estimada com o Modelo

Com Recall de 70% e taxa de conversão de campanha de 50%:

| Churners detectados/ano | Retidos efetivamente | Receita recuperada | Custo de descontos | **Resultado líquido** |
|---|---|---|---|---|
| 960 / 1.371 | 480 | US$ 1,326 M | US$ 35 K | **US$ 1,291 M** |

---

## Dataset

**IBM Telco Customer Churn** (público) — 7.043 clientes, 33 variáveis originais, **26,5% de taxa de churn**.

| Campo | Valor |
|-------|-------|
| Fonte | [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Registros | 7.043 clientes |
| Features após pré-processamento | 30 |
| Missing values | 11 (0,16%) em `Total Charges` → imputados com mediana |
| Split treino/teste | 80% / 20% estratificado por `churn` |

### Variáveis mais preditivas

| Feature | Tipo | Associação com churn |
|---------|------|----------------------|
| `tenure_months` | Numérica | Maior separabilidade (Cohen's d = −0,89) |
| `contract` | Categórica | Month-to-month: 42,7% de churn (Cramer's V = 0,41) |
| `internet_service` | Categórica | Fiber optic: 41,9% de churn (V = 0,35) |
| `online_security` | Categórica | Ausente: 41,8% de churn |
| `tech_support` | Categórica | Ausente: 41,6% de churn |
| `monthly_charges` | Numérica | Cohen's d = +0,47 |

Features removidas: IDs geográficos, e variáveis pós-evento que causariam data leakage (`Churn Score`, `Churn Reason`, `CLTV`).

---

## Resultados

### Modelo Vencedor — MLP (PyTorch)

| Métrica | Valor |
|---------|-------|
| **ROC-AUC** | 0,850 |
| **PR-AUC** | 0,666 |
| **Recall** | **98,9%** ✅ SLO atendido |
| Precisão | 36,5% |
| F1-score | 0,533 |
| FN (clientes perdidos) | **4** |
| FP (ações desnecessárias) | 640 |
| **Valor de negócio (teste)** | **R$ 1.017.420** |
| Threshold otimizado | 0,16 |

### Comparativo de Modelos (conjunto de teste, threshold otimizado por negócio)

| Modelo | ROC-AUC | PR-AUC | Recall | FN | Valor de Negócio |
|--------|---------|--------|--------|-----|-----------------|
| **MLP (selecionado)** | **0,850** | 0,666 | **98,9%** | **4** | **R$ 1.017.420** |
| GradientBoosting | 0,851 | 0,675 | 99,7% | 1 | R$ 1.010.964 |
| LogisticRegression | 0,847 | 0,660 | 98,9% | 4 | R$ 1.014.773 |
| RandomForest | 0,844 | 0,667 | 97,8% | 8 | R$ 993.778 |
| DummyClassifier | 0,500 | 0,265 | 0,0% | — | −R$ 636.000 |

> O MLP foi selecionado por apresentar o **maior valor de negócio** no conjunto de teste, com o segundo menor número de FN entre todos os modelos.

---

## Cobertura dos Requisitos do Tech Challenge

### Etapa 1 — Entendimento e Preparação

| Requisito | Entregável |
|-----------|-----------|
| ML Canvas (stakeholders, métricas de negócio, SLOs) | [`docs/ml_canvas.md`](docs/ml_canvas.md) |
| EDA completa (volume, qualidade, distribuição, data readiness) | [`notebooks/1_vab_eda.ipynb`](notebooks/1_vab_eda.ipynb) |
| Métrica técnica (AUC-ROC, PR-AUC, F1) e métrica de negócio | [`docs/metricas_tecnicas_negocios.md`](docs/metricas_tecnicas_negocios.md) |
| DummyClassifier + Regressão Logística (Scikit-Learn) | [`notebooks/3_vab_baselines_unificado.ipynb`](notebooks/3_vab_baselines_unificado.ipynb) |
| Experimentos registrados no MLflow | `mlflow.db` + `mlartifacts/` |

### Etapa 2 — Modelagem com Redes Neurais

| Requisito | Entregável |
|-----------|-----------|
| MLP em PyTorch (arquitetura, ativação, loss) | [`src/models/mlp2.py`](src/models/mlp2.py) — [128, 64], ReLU, BCEWithLogitsLoss, LayerNorm |
| Loop de treinamento com early stopping e batching | [`src/models/trainer.py`](src/models/trainer.py) — EarlyStopping(patience=15), batch=128, AdamW |
| Comparação MLP vs baselines (≥ 4 métricas) | ROC-AUC, PR-AUC, F1, Recall, Precision, Specificity |
| Análise de trade-off de custo (FP vs FN) | [`docs/tradeoff custo fp fp.md`](docs/tradeoff%20custo%20fp%20fp.md) — razão 38,7:1 documentada |
| Todos os experimentos no MLflow | [`notebooks/4_vab_mlp_vs_baselines.ipynb`](notebooks/4_vab_mlp_vs_baselines.ipynb) — Optuna + MLflow |
| Modelo vencedor no MLflow Model Registry | `mlflow.register_model("ChurnMLP", alias="champion")` |

### Etapa 3 — Engenharia e API

| Requisito | Entregável |
|-----------|-----------|
| Código refatorado em módulos (`src/`) | [`src/api/`](src/api/), [`src/data/`](src/data/), [`src/models/`](src/models/), [`src/utils/`](src/utils/) |
| Pipeline reprodutível (sklearn + transformadores custom) | `SemanticNormalizer` → `FeatureEngineer` → `ColumnTransformer` |
| Testes (pytest): unitários, schema (pandera), smoke | 12 arquivos em [`tests/`](tests/) |
| API FastAPI: `/predict`, `/health`, validação Pydantic | [`src/api/router.py`](src/api/router.py), [`src/api/schemas.py`](src/api/schemas.py) |
| Logging estruturado + middleware de latência | `get_logger()` em [`src/config.py`](src/config.py), [`src/api/middleware.py`](src/api/middleware.py) |
| `pyproject.toml` + `ruff` + `Makefile` | `make lint`, `make test`, `make pre-commit` |

### Etapa 4 — Documentação e Entrega Final

| Requisito | Entregável |
|-----------|-----------|
| Model Card (performance, limitações, vieses, falhas) | [`docs/model_card.md`](docs/model_card.md) |
| Arquitetura de deploy (batch vs real-time + justificativa) | [`docs/deploy_architecture.md`](docs/deploy_architecture.md) + seção [Arquitetura de Deploy](#arquitetura-de-deploy) |
| Plano de monitoramento (métricas, alertas, playbook) | [`docs/monitoring_plan.md`](docs/monitoring_plan.md) |
| README completo | Este arquivo |

---

## Arquitetura do Projeto

```
.
├── data/
│   ├── raw/              # Dataset original IBM Telco (.xlsx)
│   ├── interim/          # Dados tipados e limpos (.parquet)
│   └── processed/        # train.parquet (80%) · test.parquet (20%)
├── docs/                 # Documentação técnica e analítica
│   ├── model_card.md     # Model Card formal
│   ├── monitoring_plan.md# Plano de monitoramento
│   ├── ml_canvas.md      # ML Canvas
│   └── ...               # EDA, feature engineering, baselines, trade-off
├── models/
│   ├── best_model_mlp.pt # Pesos do MLP (PyTorch state_dict)
│   └── preprocessor.pkl  # Pipeline sklearn serializado
├── notebooks/
│   ├── 1_vab_eda.ipynb               # EDA completa
│   ├── 2_vab_preprocessing.ipynb     # Pipeline de dados
│   ├── 3_vab_baselines_unificado.ipynb # Baselines + MLflow
│   └── 4_vab_mlp_vs_baselines.ipynb  # MLP + Optuna + Model Registry
├── reports/
│   ├── figures/          # Plots EDA, baselines e MLP
│   └── json/             # Métricas, parâmetros Optuna, tabela comparativa
├── src/
│   ├── api/              # FastAPI (router, schemas, predictor, middleware)
│   ├── data/             # SemanticNormalizer, FeatureEngineer
│   ├── models/           # ChurnMLPv2, trainer, evaluation, experiment
│   ├── utils/            # Plots e business logic
│   ├── config.py         # Single source of truth (paths, seeds, SLOs, custos)
│   └── main.py           # Entrypoint da API
├── tests/                # 12 arquivos de teste (pytest)
├── Dockerfile            # Container de produção (Python 3.12-slim + uv)
├── Makefile              # lint · format · test · pre-commit
└── pyproject.toml        # Dependências, ruff, pytest, taskipy
```

---

## Pipeline de Dados e Treinamento

```
data/raw/ (.xlsx)
    │
    ▼ notebooks/2_vab_preprocessing.ipynb
data/interim/ (tipagem + limpeza + feature engineering)
    │
    ▼ SemanticNormalizer → FeatureEngineer → ColumnTransformer
data/processed/ (train.parquet · test.parquet)
    │
    ▼ notebooks/3 (baselines) + notebooks/4 (MLP + Optuna)
models/ (best_model_mlp.pt · preprocessor.pkl)
    │
    ▼ MLflow Model Registry (ChurnMLP@champion)
    │
    ▼ src/ → API FastAPI
```

### Pré-processamento

| Etapa | Transformador | Operação |
|-------|---------------|----------|
| Normalização semântica | `SemanticNormalizer` | "No internet service" → "No" |
| Feature engineering | `FeatureEngineer` | 6 features derivadas |
| Numéricas (3) | `SimpleImputer` + `log1p` + `StandardScaler` | Imputa, normaliza skew, padroniza |
| Binárias (11) | `OrdinalEncoder` | ["No", "Yes"] → [0, 1] |
| Nominais (6) | `OneHotEncoder(drop='first')` | Dummy encoding |

### Features Derivadas

| Feature | Descrição |
|---------|-----------|
| `num_services` | Contagem de serviços ativos (âncora de retenção) |
| `charges_per_month` | `monthly_charges / (tenure + 1)` |
| `is_month_to_month` | Flag do contrato mais associado ao churn |
| `tenure_group` | Buckets novo / médio / longo — captura não-linearidade dos primeiros 12 meses |
| `has_security_support` | Online Security OR Tech Support |
| `is_fiber_optic` | Fibra ótica tem 2× mais churn que DSL |

---

## Instalação e Setup

**Requisitos:** Python 3.12+ · Git · [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/leonflicts42/tech_challenge_1_fiap_mlops.git
cd tech_challenge_1_fiap_mlops
uv sync
```

---

## Executando a API Localmente

```bash
# Com uv
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# Ou com o ambiente ativado
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

A documentação interativa estará em:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### Endpoints disponíveis

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/api/v1/health` | Status da API e do modelo carregado |
| `POST` | `/api/v1/predict` | Predição de churn para um cliente |

### Exemplo de requisição

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "tenure_months": 12,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 70.35,
    "total_charges": 844.20
  }'
```

### Exemplo de resposta

```json
{
  "churn_probability": 0.87,
  "churn_label": "churn",
  "threshold_used": 0.16,
  "cost_estimate_brl": 73.52,
  "model_version": "best_model_mlp.pt"
}
```

---

## Executando com Docker

### Pré-requisito — artefatos do modelo

Antes de buildar, confirme que os artefatos existem localmente:

```bash
ls models/best_model_mlp.pt models/preprocessor.pkl
```

Se não existirem, execute o treinamento primeiro:

```bash
uv run python src/train_mlp.py
```

### Build e execução local

```bash
# Build da imagem
docker build -t churn-api .

# Executar em background (porta 5000)
docker run -d --name churn-api -p 5000:5000 churn-api
```

A API ficará disponível em http://localhost:5000/docs

### Passando os artefatos do modelo como volume

Por padrão, os artefatos (`models/best_model_mlp.pt` e `models/preprocessor.pkl`) são copiados para dentro da imagem no `docker build`. Para atualizar o modelo sem rebuild, monte um volume externo:

```bash
docker run -d --name churn-api \
  -p 5000:5000 \
  -v $(pwd)/models:/app/models:ro \
  churn-api
```

### Verificar se a API está saudável

```bash
curl http://localhost:5000/api/v1/health
```

Resposta esperada:

```json
{
  "status": "ok",
  "model_loaded": true,
  "preprocessor_loaded": true,
  "model_version": "best_model_mlp.pt",
  "threshold": 0.16
}
```

### Testar uma predição

```bash
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "tenure_months": 12,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 70.35,
    "total_charges": 844.20
  }'
```

Resposta esperada:

```json
{
  "churn_probability": 0.87,
  "churn_label": "churn",
  "threshold_used": 0.16,
  "cost_estimate_brl": 73.52,
  "model_version": "best_model_mlp.pt"
}
```

### Parar o container

```bash
docker stop churn-api && docker rm churn-api
```

---

## MLflow — Rastreamento de Experimentos

```bash
# Iniciar a interface web do MLflow
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db

# Com uv
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Acesse em **http://localhost:5000** para visualizar:
- Todos os experimentos: Optuna trials (50 por modelo), runs de validação, run do vencedor
- Métricas por epoch do MLP (loss, AUC)
- Datasets rastreados (train/test com MD5)
- **Model Registry:** `ChurnMLP` com alias `champion`

Para carregar o modelo registrado:

```python
import mlflow.pytorch
model = mlflow.pytorch.load_model("models:/ChurnMLP@champion")
```

---

## Testes

```bash
# Executar todos os testes
make test

# Com relatório de cobertura
pytest --cov=src --cov-report=html
# Abrir htmlcov/index.html no navegador
```

### Suíte de testes (12 arquivos)

| Arquivo | Cobertura |
|---------|-----------|
| `test_smoke.py` | Smoke tests — API inicia e responde |
| `test_schema.py` | Validação Pydantic — campos, tipos, ranges |
| `test_api.py` | Endpoints `/health` e `/predict` — 18+ cenários |
| `test_main.py` | Configuração e inicialização da aplicação |
| `test_preprocessing.py` | `SemanticNormalizer` — transformações e consistência |
| `test_features.py` | `FeatureEngineer` — 6 features, 36+ cenários |
| `test_predictor.py` | Pipeline completo de inferência end-to-end |
| `test_evaluation.py` | Métricas e `CostAnalyzer` |
| `test_mlp.py` | Arquitetura MLP — forward, backward, inicialização |
| `test_mlp2.py` | `ChurnMLPInference` — compatibilidade de checkpoint |
| `test_etapa2.py` | Testes de integração da modelagem |
| `conftest.py` | Fixtures compartilhados (payloads, mock predictor) |

---

## Linting e Qualidade de Código

```bash
make format      # Formata com ruff (auto-fix)
make lint        # Verifica erros (ruff check)
make pre-commit  # format → lint → test (sequencial)
```

O CI/CD (`.github/workflows/CI.yml`) executa automaticamente `ruff check`, `ruff format --check` e `pytest` em todo push e pull request para `main`.

---

## Arquitetura de Deploy

### Modo implementado: Real-time (Online Inference)

A API serve predições **síncronas e individuais** via `POST /api/v1/predict`.

**Justificativa:** O caso de uso exige acionamento imediato da equipe de retenção no momento em que um evento de risco é detectado (ex.: consulta de planos concorrentes, abertura de chamado). A latência de inferência do MLP é < 50 ms por requisição — viável para integração direta com CRM em tempo real. O volume de requisições (poucos milhares de clientes/dia) não justifica processamento batch.

**Alternativa batch (descartada para este escopo):** Adequada para campanhas diárias de e-mail, mas perde a capacidade de reagir a eventos em tempo real. O [`docs/monitoring_plan.md`](docs/monitoring_plan.md) documenta a arquitetura batch para escenários futuros.

### Fluxo de inferência

```
Cliente (CRM)  →  POST /api/v1/predict
                        │
               FastAPI (main.py)
                        │
          LatencyMiddleware (UUID · tempo ms)
                        │
             ChurnPredictor (predictor.py)
              ┌──────────────────────┐
              │                      │
   preprocessor.pkl           best_model_mlp.pt
   ColumnTransformer           ChurnMLPInference
              │                      │
   SemanticNormalizer           forward pass
   FeatureEngineer            [30 → 128 → 64 → 1]
   30 features                  sigmoid → proba
              └──────────────────────┘
                        │
               ChurnResponse (JSON)
          {probability · label · threshold · cost}
```

---

## Boas Práticas Implementadas

| Prática | Implementação |
|---------|---------------|
| **Reprodutibilidade** | `RANDOM_STATE = 42` em numpy, random, sklearn, torch, torch.cuda |
| **Validação cruzada estratificada** | `StratifiedKFold(5)`, OOF collection em todos os modelos |
| **Logging estruturado** | `get_logger()` — console + arquivo, sem `print()` no `src/` |
| **Linting** | `ruff check` + `ruff format` — CI bloqueia PRs com erros |
| **Seeds fixados** | Aplicados em todas as etapas: split, CV, Optuna sampler, DataLoader |
| **Pipeline sklearn** | Fit apenas no treino; `preprocessor.pkl` garante consistência treino→inferência |
| **Testes automatizados** | 12 arquivos, cobertura de smoke, schema, API, unit, integração |
| **Single source of truth** | `src/config.py` centraliza todas as constantes do projeto |

---

## Documentação Adicional

| Documento | Descrição |
|-----------|-----------|
| [`docs/model_card.md`](docs/model_card.md) | Performance, limitações, vieses e cenários de falha do modelo |
| [`docs/deploy_architecture.md`](docs/deploy_architecture.md) | ADR completo: batch vs real-time, fluxo de inferência, containerização, opções de nuvem |
| [`docs/monitoring_plan.md`](docs/monitoring_plan.md) | SLOs, alertas P1–P4, playbooks de incidentes, critérios de retreinamento |
| [`docs/ml_canvas.md`](docs/ml_canvas.md) | ML Canvas — stakeholders, problema, métricas de negócio |
| [`docs/metricas_tecnicas_negocios.md`](docs/metricas_tecnicas_negocios.md) | Hierarquia de métricas e alinhamento com negócio |
| [`docs/tradeoff custo fp fp.md`](docs/tradeoff%20custo%20fp%20fp.md) | Análise completa da assimetria FP/FN com CLV por segmento |
| [`docs/analise_eda.md`](docs/analise_eda.md) | Síntese da análise exploratória |
| [`docs/analise_3_baseline.md`](docs/analise_3_baseline.md) | Resultados detalhados dos modelos baseline |

---

## Dependências Principais

| Biblioteca | Versão | Uso |
|------------|--------|-----|
| `torch` | ≥ 2.11 | Treinamento e inferência da MLP |
| `scikit-learn` | ≥ 1.4 | Pipelines de pré-processamento e baselines |
| `mlflow` | ≥ 3.0 | Rastreamento de experimentos e Model Registry |
| `fastapi` | ≥ 0.135 | API de inferência |
| `optuna` | ≥ 4.8 | Otimização de hiperparâmetros (50 trials/modelo) |
| `imbalanced-learn` | ≥ 0.14 | Tratamento de desbalanceamento de classes |
| `pandera` | ≥ 0.31 | Validação de schema em testes |
| `ruff` | ≥ 0.15 | Linting e formatação |

Todas as dependências com versões exatas em [`pyproject.toml`](pyproject.toml) e travadas em [`uv.lock`](uv.lock).
