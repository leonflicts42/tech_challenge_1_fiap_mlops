# Churn Prediction — MLOps End-to-End

Projeto de predição de churn para uma operadora de telecomunicações, desenvolvido como Tech Challenge da Fase 1 do programa de pós-graduação em MLOps da FIAP.

O modelo central é uma **Rede Neural MLP (PyTorch)** comparada com baselines Scikit-Learn, rastreada com MLflow e servida via **API FastAPI**.

---

## Contexto de Negócio

Uma operadora de telecomunicações enfrenta perda acelerada de clientes. Com taxa de churn de ~26%, o impacto financeiro estimado ultrapassa R$ 5 milhões anuais. O objetivo é classificar clientes com risco de cancelamento para permitir ações proativas de retenção.

**Custo assimétrico:** Perder um cliente (FN) custa ~38,7× mais do que acionar uma campanha de retenção desnecessária (FP). O modelo foi calibrado para maximizar recall dentro de uma função de valor de negócio.

---

## Resultados do Modelo (Conjunto de Teste)

| Métrica | Valor |
|---------|-------|
| ROC-AUC | **0.850** |
| PR-AUC | 0.666 |
| Recall | **98,9%** |
| Precisão | 36,5% |
| Threshold otimizado | 0.16 |
| Valor de negócio estimado | **R$ 1.017.420** |

Veja o [Model Card completo](docs/model_card.md) para performance detalhada, limitações, vieses e cenários de falha.

---

## Estrutura do Repositório

```
.
├── data/
│   ├── raw/          # Dataset original IBM Telco (.xlsx)
│   ├── interim/      # Dados tipados e limpos (.parquet)
│   └── processed/    # Train/test splits (.parquet)
├── docs/             # Documentação técnica e analítica
│   ├── model_card.md         # Model Card formal
│   ├── monitoring_plan.md    # Plano de monitoramento
│   ├── ml_canvas.md          # ML Canvas do projeto
│   └── ...                   # Análises de EDA, features, baseline
├── models/           # Artefatos treinados
│   ├── best_model_mlp.pt     # Pesos do MLP (PyTorch)
│   └── preprocessor.pkl      # Pipeline sklearn serializado
├── notebooks/        # Análise exploratória e treinamento
│   ├── 1_vab_eda.ipynb
│   ├── 2_vab_preprocessing.ipynb
│   ├── 3_vab_baselines_unificado.ipynb
│   └── 4_vab_mlp_vs_baselines.ipynb
├── reports/          # Métricas, figuras e JSONs de resultados
├── src/              # Código-fonte do projeto
│   ├── api/          # FastAPI (router, schemas, predictor, middleware)
│   ├── data/         # Pré-processamento e feature engineering
│   ├── models/       # MLP, trainer, evaluation, experiment
│   ├── utils/        # Plots e business logic
│   ├── config.py     # Single source of truth (paths, seeds, SLOs)
│   └── main.py       # Entrypoint da API
├── tests/            # Testes automatizados (pytest)
├── Dockerfile        # Container de produção
├── Makefile          # Comandos de lint, test e execução
└── pyproject.toml    # Dependências, ruff, pytest
```

---

## Setup e Instalação

**Requisitos:** Python 3.12+

### Instalação com pip

```bash
# 1. Clone o repositório
git clone https://github.com/leonflicts42/tech_challenge_1_fiap_mlops.git
cd tech_challenge_1_fiap_mlops

# 2. Crie e ative o ambiente virtual
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\Activate.ps1       # PowerShell (Windows)

# 3. Instale o projeto
pip install -e .
```

### Instalação com uv (recomendado)

```bash
# Instala todas as dependências de forma reprodutível
uv sync
```

---

## Executando a API

```bash
# Com uvicorn direto
uvicorn main:app --app-dir src --reload --port 8000

# Com uv
uv run uvicorn main:app --app-dir src --reload --port 8000
```

A documentação interativa estará disponível em:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Endpoints

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/api/v1/health` | Verifica status da API e do modelo |
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

## MLflow — Rastreamento de Experimentos

```bash
# Inicia a interface web do MLflow
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Acesse em http://localhost:5000 para ver os experimentos, métricas por epoch e artefatos de todos os modelos treinados (MLP, RandomForest, GradientBoosting, LogisticRegression, DummyClassifier).

---

## Testes

```bash
# Executar todos os testes
make test

# Com cobertura
pytest --cov=src --cov-report=html
```

O projeto possui **12 arquivos de teste** cobrindo smoke tests, schema validation (Pandera), testes de API, pré-processamento, feature engineering, treinamento do MLP e avaliação de métricas.

---

## Linting e Formatação

```bash
make format      # Formata código com ruff
make lint        # Verifica erros de linting
make pre-commit  # Executa format → lint → test em sequência
```

---

## Execução com Docker

```bash
# Build da imagem
docker build -t churn-api .

# Executa o container
docker run -p 8000:8000 churn-api
```

---

## Arquitetura de Deploy

### Modo escolhido: Real-time (Online Inference)

A API foi implementada em modo **real-time (online inference)** via FastAPI, onde cada requisição resulta em uma predição imediata e síncrona.

**Justificativa:**
- O caso de uso exige acionamento rápido da equipe de retenção no momento em que o cliente apresenta sinal de risco (ex.: abertura de chamado, consulta de planos concorrentes).
- A latência de inferência do MLP é < 50 ms por requisição, viável para integração com CRM em tempo real.
- O volume de requisições é baixo (poucos milhares de clientes/dia), sem necessidade de processamento batch.

**Alternativa batch (descartada):** Um pipeline batch diário seria adequado se o objetivo fosse apenas gerar listas de risco para campanhas de e-mail. Nesse cenário, a latência não importa, mas perde-se a capacidade de reagir a eventos em tempo real.

### Componentes da arquitetura

```
Cliente (CRM) → POST /api/v1/predict
                    ↓
              FastAPI (main.py)
                    ↓
         LatencyMiddleware (UUID, tempo)
                    ↓
           ChurnPredictor (predictor.py)
            ↓                    ↓
  preprocessor.pkl         best_model_mlp.pt
  (ColumnTransformer)      (ChurnMLPInference)
            ↓                    ↓
     SemanticNormalizer     Forward pass (30 → 128 → 64 → 1)
     FeatureEngineer        sigmoid → probabilidade
            ↓                    ↓
                   ChurnResponse (JSON)
```

### Pipeline de dados (offline)

```
data/raw/ (xlsx)
    ↓ [notebooks/2_vab_preprocessing.ipynb]
data/interim/ (parquet tipado + limpo)
    ↓ [SemanticNormalizer + FeatureEngineer + ColumnTransformer]
data/processed/ (train.parquet + test.parquet)
    ↓ [notebooks/4_vab_mlp_vs_baselines.ipynb + MLflow]
models/ (best_model_mlp.pt + preprocessor.pkl)
    ↓
API (src/)
```

---

## Documentação Adicional

| Documento | Descrição |
|-----------|-----------|
| [Model Card](docs/model_card.md) | Performance, limitações, vieses e cenários de falha |
| [Plano de Monitoramento](docs/monitoring_plan.md) | Métricas, alertas e playbook de incidentes |
| [ML Canvas](docs/ml_canvas.md) | Formulação do problema e stakeholders |
| [EDA](docs/analise_eda.md) | Análise exploratória dos dados |
| [Trade-off de Custo](docs/tradeoff%20custo%20fp%20fp.md) | Análise FP vs FN com CLV |

---

## Dataset

**IBM Telco Customer Churn** (público) — 7.043 clientes, 33 variáveis, 26,4% de taxa de churn.

Fonte: [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## Dependências Principais

| Biblioteca | Versão | Uso |
|------------|--------|-----|
| PyTorch | ≥ 2.11 | Treinamento e inferência do MLP |
| Scikit-Learn | ≥ 1.4 | Pipelines de pré-processamento e baselines |
| MLflow | ≥ 3.0 | Rastreamento de experimentos |
| FastAPI | ≥ 0.135 | API de inferência |
| Optuna | ≥ 4.8 | Otimização de hiperparâmetros |

Veja todas as dependências em [pyproject.toml](pyproject.toml).
