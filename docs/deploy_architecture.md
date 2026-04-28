# Arquitetura de Deploy — Churn Prediction API

## Decisão: Real-time (Online Inference)

**Padrão adotado:** Inferência síncrona e individual via API REST (`POST /api/v1/predict`).

---

## Contexto da Decisão

O modelo serve equipes de retenção de uma operadora de telecomunicações. O gatilho para uma predição é um **evento de risco em tempo real** — abertura de chamado, consulta de planos concorrentes, atraso de pagamento. Nesse contexto, a janela de oportunidade para acionar uma oferta de retenção é de minutos, não de horas.

---

## Comparativo Batch vs. Real-time

| Critério | Batch | Real-time (adotado) |
|----------|-------|----------------------|
| **Latência de resposta** | Horas (próximo ciclo de processamento) | < 50 ms por requisição |
| **Gatilho de predição** | Calendário fixo (diário/semanal) | Evento em tempo real (CRM, webhook) |
| **Integração com CRM** | Arquivo exportado → importação manual | API chamada diretamente pelo CRM |
| **Infraestrutura** | Job scheduler (Airflow, cron) | Servidor HTTP persistente (Uvicorn) |
| **Escalabilidade** | Alta para grandes volumes periódicos | Horizontal via réplicas do container |
| **Operação** | Simples, sem SLA de latência | Requer monitoramento de disponibilidade |
| **Custo computacional** | Concentrado em janelas curtas | Distribuído continuamente |

---

## Justificativa da Escolha

### 1. Janela de ação é curta

A literatura de retenção de telecomunicações indica que o período ótimo para intervenção é **imediatamente após** o evento de insatisfação. Um batch diário entrega a predição horas depois do evento, reduzindo significativamente a taxa de conversão da campanha de retenção.

### 2. Latência de inferência viável

O MLP (30 → 128 → 64 → 1) realiza inferência em < 50 ms em CPU, dentro do SLO de latência p95 ≤ 500 ms. O pré-processamento (`SemanticNormalizer` → `FeatureEngineer` → `ColumnTransformer`) adiciona < 5 ms. Real-time é tecnicamente viável sem GPU.

### 3. Volume não justifica batch

O dataset base tem 7.043 clientes. Mesmo com crescimento, o volume de requisições diárias é da ordem de milhares — compatível com um único container. Batch traria complexidade operacional (scheduler, armazenamento intermediário, reconciliação) sem ganho real.

### 4. Integração nativa com sistemas existentes

Sistemas de CRM (Salesforce, HubSpot) e plataformas de atendimento expõem webhooks e chamadas HTTP. A API REST é o contrato de integração natural, sem necessidade de pipelines ETL intermediários.

---

## Arquitetura Implementada

```
Cliente / CRM
    │
    │  POST /api/v1/predict  (JSON · 19 campos)
    ▼
┌─────────────────────────────────────────────┐
│              FastAPI (main.py)              │
│                                             │
│  LatencyMiddleware                          │
│  ├─ UUID por requisição                     │
│  └─ Tempo de resposta (ms) no log           │
│                                             │
│  ChurnPredictor (predictor.py)              │
│  ├─ preprocessor.pkl  (sklearn Pipeline)    │
│  │   SemanticNormalizer                     │
│  │   FeatureEngineer  (6 features)          │
│  │   ColumnTransformer (30 features)        │
│  │                                          │
│  └─ best_model_mlp.pt (ChurnMLPInference)   │
│      30 → 128 → 64 → 1  (sigmoid)          │
│                                             │
│  ChurnResponse (JSON)                       │
│  ├─ churn_probability                       │
│  ├─ churn_label                             │
│  ├─ threshold_used  (0.16)                  │
│  └─ cost_estimate_brl                       │
└─────────────────────────────────────────────┘
    │
    │  HTTP 200 JSON  (< 50 ms)
    ▼
Cliente / CRM
```

### Componentes e responsabilidades

| Componente | Arquivo | Responsabilidade |
|-----------|---------|-----------------|
| Aplicação HTTP | `src/main.py` | Lifespan: carrega modelo e preprocessor uma vez na inicialização |
| Middleware | `src/api/middleware.py` | UUID de rastreamento, logging de latência por requisição |
| Endpoints | `src/api/router.py` | `GET /health` (liveness probe), `POST /predict` (inferência) |
| Contrato de dados | `src/api/schemas.py` | Pydantic: validação de tipos, ranges e valores canônicos |
| Pipeline de inferência | `src/api/predictor.py` | Orquestra preprocessor → modelo → threshold → resposta |
| Threshold | `src/config.py` | `DECISION_THRESHOLD = 0.16` (otimizado por valor de negócio) |

---

## Containerização

O `Dockerfile` empacota a aplicação em uma imagem reprodutível:

```dockerfile
# Python 3.12-slim + uv para instalação determinística
FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install uv && uv sync --frozen
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "5000"]
```

```bash
# Build e execução local
docker build -t churn-api .
docker run -p 5000:5000 churn-api

# Verificação
curl http://localhost:5000/api/v1/health
```

---

## Alternativa Batch (Descartada)

Para cenários futuros onde o volume ou o orçamento de infraestrutura mudem, o padrão batch seria adequado:

| Cenário | Padrão Recomendado |
|---------|--------------------|
| Campanha diária de e-mail para toda a base | Batch — Job diário às 06h, CSV entregue ao CRM |
| > 100 k requisições/hora | Batch com fila assíncrona (Celery + Redis) |
| Integração com data warehouse (BigQuery, Redshift) | Batch com Airflow DAG |
| Custo de inferência em tempo real proibitivo | Batch periódico com cache de predições |

O plano de monitoramento ([`docs/monitoring_plan.md`](monitoring_plan.md)) documenta os SLOs e alertas operacionais para o modo real-time em produção.

---

## Estratégia de Deploy em Nuvem (Opcional)

A imagem Docker é portável para qualquer provedor:

| Provedor | Serviço sugerido | Observações |
|----------|-----------------|-------------|
| **AWS** | ECS Fargate + ALB | Sem gerenciamento de servidor, auto-scaling nativo |
| **GCP** | Cloud Run | Deploy direto de imagem Docker, scale-to-zero |
| **Azure** | Container Apps | Integração nativa com Azure Monitor |

O endpoint público seria referenciado no README após o deploy com a URL do load balancer ou do serviço gerenciado.
