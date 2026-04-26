# Plano de Monitoramento — Churn Prediction API

## Contexto

A API de predição de churn serve o modelo `ChurnMLP v2` em tempo real via `POST /api/v1/predict`. Este plano define as métricas a monitorar, os thresholds de alerta e o playbook de resposta a incidentes para garantir que o modelo continue operando dentro dos SLOs definidos em produção.

---

## SLOs de Operação

| SLO | Threshold | Janela |
|-----|-----------|--------|
| **Recall mínimo** | ≥ 0.70 | Mensal |
| **ROC-AUC mínimo** | ≥ 0.80 | Mensal |
| **Latência p95 da API** | ≤ 500 ms | Diária |
| **Disponibilidade da API** | ≥ 99.5% | Mensal |
| **Taxa de erros 5xx** | ≤ 0.5% | Diária |

---

## Métricas a Monitorar

### 1. Métricas de Performance do Modelo

| Métrica | Descrição | Frequência | Alerta |
|---------|-----------|------------|--------|
| **Recall** | % churners corretamente identificados | Mensal | < 0.70 |
| **ROC-AUC** | Capacidade discriminativa geral | Mensal | < 0.80 |
| **PR-AUC** | AUC da curva Precision-Recall | Mensal | < 0.55 |
| **Taxa de predição positiva** | % de clientes classificados como churn | Semanal | Desvio > 10pp do baseline (≈ 73%) |
| **Distribuição de probabilidades** | Histograma de `churn_probability` | Semanal | Mudança na forma da distribuição |

> Nota: O recall é a métrica primária por conta da assimetria de custo (FN custa 38,7× mais que FP).

### 2. Métricas de Data Drift (Qualidade dos Dados de Entrada)

| Feature | Método | Threshold de Alerta |
|---------|--------|---------------------|
| `monthly_charges` | PSI (Population Stability Index) | PSI > 0.20 |
| `total_charges` | PSI | PSI > 0.20 |
| `tenure_months` | PSI | PSI > 0.20 |
| `contract` | Chi-Quadrado | p-value < 0.05 |
| `internet_service` | Chi-Quadrado | p-value < 0.05 |
| Features binárias agregadas | Taxa de missing values | > 5% |

**Cálculo do PSI:**
```
PSI = Σ (% atual - % referência) × ln(% atual / % referência)
PSI < 0.10 → sem drift  
0.10 ≤ PSI < 0.20 → drift moderado (investigar)  
PSI ≥ 0.20 → drift severo (retreinar)
```

### 3. Métricas de Infraestrutura da API

| Métrica | Ferramenta | Threshold de Alerta |
|---------|------------|---------------------|
| **Latência p50** | Logs `X-Process-Time` | > 200 ms |
| **Latência p95** | Logs `X-Process-Time` | > 500 ms |
| **Taxa de erro 4xx** | Logs do middleware | > 5% das requisições |
| **Taxa de erro 5xx** | Logs do middleware | > 0.5% das requisições |
| **Disponibilidade** | `GET /api/v1/health` | status ≠ "ok" |
| **Memória do processo** | Sistema operacional | > 80% do limite do container |

### 4. Métricas de Negócio

| Métrica | Descrição | Frequência |
|---------|-----------|------------|
| **Valor de negócio estimado** | R$ gerado/perdido por predição | Mensal |
| **Taxa de conversão de campanhas** | % churners salvos após intervenção | Trimestral |
| **Custo total de FP** | Total R$ gastos em ações desnecessárias | Mensal |
| **Custo de FN não detectado** | Churners reais não identificados | Mensal |

---

## Coleta de Dados para Monitoramento

### Logs disponíveis
O middleware `LatencyMiddleware` (`src/api/middleware.py`) registra automaticamente:
- `X-Request-ID` — identificador único por requisição
- `X-Process-Time` — latência em ms
- Método HTTP, path, status code

### Ground truth
Para calcular métricas de performance do modelo, é necessário coletar o **resultado real de churn** 30 dias após cada predição. Recomenda-se:
1. Armazenar em banco de dados: `{request_id, customer_id, churn_probability, prediction, timestamp}`
2. Fazer join com dados de cancelamento 30 dias depois
3. Calcular métricas mensais com janela deslizante de 90 dias

### Referência de distribuição (baseline)
Os dados de referência para cálculo de PSI devem ser o conjunto de **treino** do modelo atual:
- `data/processed/train.parquet`
- MD5: `7575994d37201ad71c968369257942a0`

---

## Alertas e Notificações

### Níveis de severidade

| Nível | Cor | Ação |
|-------|-----|------|
| **P1 — Crítico** | 🔴 | Intervenção imediata (< 1h); escalonar para engenheiro de plantão |
| **P2 — Alto** | 🟠 | Investigação no mesmo dia (< 4h) |
| **P3 — Médio** | 🟡 | Investigação na próxima sprint |
| **P4 — Informativo** | 🔵 | Registrar e revisar mensalmente |

### Tabela de alertas

| Condição | Severidade | Canal |
|----------|-----------|-------|
| API retornando 503 consecutivos | P1 🔴 | PagerDuty + Slack |
| Taxa de erro 5xx > 0.5% | P1 🔴 | PagerDuty + Slack |
| Recall < 0.70 no mês | P2 🟠 | Slack + email |
| ROC-AUC < 0.80 no mês | P2 🟠 | Slack + email |
| PSI > 0.20 em qualquer feature numérica | P2 🟠 | Slack + email |
| Latência p95 > 500 ms | P2 🟠 | Slack |
| Taxa de predição positiva desvio > 10pp | P3 🟡 | Email |
| PSI entre 0.10 e 0.20 | P3 🟡 | Email semanal |
| Latência p50 > 200 ms | P4 🔵 | Log diário |

---

## Playbook de Resposta a Incidentes

### Incidente 1: API indisponível (503)

```
1. Verificar health check: GET /api/v1/health
2. Checar logs do container: docker logs <container_id> --tail 100
3. Verificar se modelo foi carregado: campo model_loaded = true no /health
4. Se modelo não carregado: reiniciar container e verificar models/best_model_mlp.pt existe
5. Se modelo carregado mas API não responde: verificar uso de memória e CPU
6. Escalonar se não resolvido em 30min
```

### Incidente 2: Recall abaixo do SLO (< 0.70)

```
1. Confirmar com dados dos últimos 30 dias (mínimo 200 predições)
2. Calcular PSI de todas as features numéricas vs conjunto de treino
3. Se PSI > 0.20 em alguma feature → drift confirmado → iniciar retreinamento
4. Se PSI < 0.10 → problema pode ser de amostragem ou sazonalidade → aguardar 1 mês
5. Se PSI entre 0.10-0.20 → investigar distribuição da feature afetada
6. Comunicar equipe de negócio sobre redução de performance
```

### Incidente 3: Data drift severo (PSI > 0.20)

```
1. Identificar qual(is) feature(s) estão em drift
2. Investigar causa: reajuste de preços, novo plano, campanha de marketing?
3. Coletar dados do período recente (mínimo 2.000 clientes)
4. Re-executar pipeline de pré-processamento e retreinar modelo
5. Validar métricas do novo modelo no conjunto de teste atualizado
6. Deploy do novo modelo se recall >= 0.70 e ROC-AUC >= 0.80
7. Atualizar dataset de referência de PSI para o novo período
```

### Incidente 4: Latência alta (p95 > 500 ms)

```
1. Verificar logs de X-Process-Time dos últimos 100 requests
2. Checar se o problema é no carregamento do modelo (ocorre só no startup?)
3. Verificar carga do servidor (CPU, memória, disco)
4. Se problema de carga: escalar horizontalmente ou otimizar batch de requisições
5. Se problema no modelo: investigar se preprocessor.pkl ou best_model_mlp.pt corrompeu
6. Considerar implementar cache de preprocessor em memória
```

### Incidente 5: Taxa de erro 4xx alta (> 5%)

```
1. Analisar logs de validação: quais campos estão falhando?
2. Verificar se houve mudança no schema de entrada (novos planos, valores fora de range)
3. Atualizar SemanticNormalizer se novos valores de categorias foram introduzidos
4. Atualizar validações Pydantic em schemas.py se ranges mudaram
5. Comunicar equipe de integração sobre mudanças
```

---

## Ciclo de Retreinamento

| Gatilho | Ação | Prazo |
|---------|------|-------|
| PSI > 0.20 em feature numérica | Retreinamento completo | 5 dias úteis |
| Recall < 0.70 por 2 meses consecutivos | Retreinamento completo | 10 dias úteis |
| Novo produto/plano lançado pela operadora | Atualizar SemanticNormalizer + avaliar retreinamento | 3 dias úteis |
| 6 meses sem retreinamento | Retreinamento preventivo | Próxima sprint |

### Critérios de aprovação para novo modelo
- Recall ≥ 0.70 no conjunto de teste
- ROC-AUC ≥ 0.80 no conjunto de teste
- Valor de negócio ≥ R$ 950.000 no conjunto de teste (equivalente a 93% do baseline atual)
- Todos os testes automatizados passando (`pytest`)

---

## Referências

- Artefato atual: `models/best_model_mlp.pt`
- Métricas de referência: `reports/json/winner_model_report.json`
- Dataset de referência (PSI): `data/processed/train.parquet`
- Configurações de custo: `src/config.py` (`COST_CLV`, `COST_RETENTION`, `SLO_RECALL_MIN`)
- Experimentos MLflow: `mlflow.db` (experimento `churn-telecom`)
