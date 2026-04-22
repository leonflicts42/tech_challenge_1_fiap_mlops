mudanças:
cltv não é mantida, é descartada para o treinamento, 
2.3 remover cenario de equilibrio, 
4.1 remover mcc. 
4.2 a metrica operacional primeira é o recall e a prioridade é o auc-roc. 
4.3 tabela comparativa do cenarios com dummy, onde se concede desconto para todos, evita 100% do churn, porem além de evitar a perda de receita pelo churn, ainda gasta com campanha para clientes que ficariam, o ponto otimo é o equilibrio financeiro entre o valor gasto com campanha e o churn retido. 
4.5 explicar mais a relação do valor investido em campanha e o recuperado, 
4.6 remover custo do projeto de $500 reais mensais, pois não sabemos ainda . 
utilizar o 5 e o 5.1 nas proximas etapas, utilzar o 
5.3 na otimização do mlp

# Etapa 1 — Consolidação Final
## Tech Challenge · MLOps · Predição de Churn para Telecomunicações

> **Documento:** Relatório técnico consolidado da Etapa 1 — ML Canvas, métricas, trade-off e plano de deploy
> **Dataset:** Telco Customer Churn (IBM) · 7.043 clientes
> **Destinatários:** Stakeholders de negócio, time técnico e avaliação acadêmica
> **Status:** Fechamento da fundamentação antes do treino dos baselines e da MLP

---

## Sumário Executivo

A empresa perde 26,5% dos clientes ao ano, destruindo aproximadamente **US$ 5,16M em receita futura** contratada mas não realizada. Em cenário de equilíbrio, cerca de **114 clientes saem por mês** sem qualquer ação preventiva.

Construímos a fundamentação para um modelo de predição de churn calibrado pelo custo real dos erros: um **Falso Negativo** (cliente que sai sem o modelo alertar) custa, em mediana, **US$ 2.845** em receita futura perdida — enquanto um **Falso Positivo** (campanha de retenção enviada a quem ficaria) custa apenas **US$ 73,52** no desconto do plano anual.

Essa assimetria de aproximadamente **30 vezes** (precisamente 38,7x) é a âncora das decisões técnicas: o modelo deve maximizar **Recall** mesmo que sacrifique precisão.

Com o modelo atingindo as metas projetadas (Recall ≥ 70%) e taxa de conversão da campanha de 50%, estima-se uma recuperação líquida anual de **US$ 1,29M** — aproximadamente **34% da perda anual** em regime permanente.

---

## 1. ML Canvas Consolidado

### 1.1 Background

**End-User (Usuário Final)**

| Ator | Papel |
|---|---|
| Time de Retenção / CRM | Usuário direto — recebe lista priorizada de clientes em risco e executa campanhas |
| Gerente de Customer Success | Monitora KPIs de churn e eficiência das ações de retenção |
| Analista de Dados / MLOps | Opera, monitora e retreina o modelo via MLflow |
| Diretoria Comercial | Stakeholder estratégico — decisões orçamentárias baseadas em previsões |
| Clientes em risco | Afetados indiretamente — recebem abordagem proativa |

**Restrições éticas:** o modelo é ferramenta interna B2B; o cliente nunca é informado da classificação; grupos demográficos vulneráveis (ex.: `Senior Citizen = Yes`) devem ter tratamento monitorado como viés potencial.

**Value Proposition**

> *"Antecipar quais clientes irão cancelar o serviço nos próximos 30 dias, permitindo que o time de retenção atue preventivamente antes que o cancelamento ocorra."*

| Objetivo | Métrica de Sucesso |
|---|---|
| Reduzir churn voluntário | Redução ≥ 5 p.p. na taxa de churn mensal após 6 meses |
| Priorizar esforço de retenção | Top-decile lift ≥ 2,5x sobre seleção aleatória |
| Maximizar receita recorrente retida | ROI de campanha ≥ 3:1 |
| Recuperar receita futura | Receita líquida anual recuperada ≥ US$ 1M |

### 1.2 Problem Specification

| Item | Descrição |
|---|---|
| Pergunta a predizer | "Este cliente vai cancelar o contrato nos próximos 30 dias?" |
| Input | 20 features tabulares: demografia, contrato, serviços, cobranças |
| Output | `Churn = 0/1` + probabilidade contínua `P(Churn) ∈ [0, 1]` |
| Tipo de problema | Classificação binária supervisionada · desbalanceamento moderado (26,5%) |

### 1.3 Features utilizadas e descartadas

**Descartadas (13 colunas):**

| Categoria | Colunas | Motivo |
|---|---|---|
| Identificação | `CustomerID`, `Count` | Sem valor preditivo, risco de overfitting a IDs específicos |
| Geográficas | `Country`, `State`, `City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude` | **Evitar viés geográfico** — dataset concentrado em um estado; generalização comprometida |
| Oráculos / pós-evento | `Churn Score` | Score gerado pela IBM com acesso ao rótulo — causaria data leakage |
| Pós-evento | `Churn Reason` | Só existe para clientes que já saíram — não disponível em produção |
| Targets | `Churn Label`, `Churn Value` | São a variável a predizer, não features |

**Mantidas (20 features):**

| Grupo | Features |
|---|---|
| Demográficas (4) | `Gender`, `Senior Citizen`, `Partner`, `Dependents` |
| Contrato/Billing (5) | `Contract`, `Paperless Billing`, `Payment Method`, `Monthly Charges`, `Total Charges` |
| Relacionamento (1) | `Tenure Months` |
| Serviços telefônicos (2) | `Phone Service`, `Multiple Lines` |
| Internet e add-ons (7) | `Internet Service`, `Online Security`, `Online Backup`, `Device Protection`, `Tech Support`, `Streaming TV`, `Streaming Movies` |
| Valor do cliente (1) | `CLTV` |

> **Nota sobre CLTV:** mantido como feature por ser uma estimativa prospectiva calculada a partir do perfil do cliente (não do resultado de churn). Será validado via análise de correlação e de importância após o treino — se houver indícios de leakage, será removido.

### 1.4 Data Sources

| Fonte | Uso | Disponibilidade |
|---|---|---|
| CRM Interno | Dados contratuais: tenure, tipo de contrato, serviços ativos | Produção |
| Sistema de Billing | Cobranças mensais, forma de pagamento | Produção |
| Dataset IBM Telco | Desenvolvimento e treinamento inicial (7.043 clientes) | Kaggle / IBM |
| Sistema de Atendimento | NPS, chamados (feature futura) | Produção (futuro) |

---

## 2. Contexto Financeiro da Empresa

### 2.1 Estado atual

| Métrica | Valor |
|---|---|
| Total de clientes | 7.043 |
| Ativos | 5.174 (73,5%) |
| Churned | 1.869 (26,5%) |
| MRR atual | US$ 316.985,75 / mês |
| MRR perdido pelos churned | US$ 139.130,85 / mês |
| Receita histórica total (Σ Total Charges) | US$ 16.056.168,70 |
| **Receita futura destruída pelo churn** | **US$ 5.163.882,40** |
| CLTV total dos churned | US$ 7.755.256,00 |

### 2.2 Por que a receita destruída diverge do CLTV total

Um ponto que pode causar confusão na leitura dos números acima: **a receita futura destruída (US$ 5,16M) é menor que o CLTV total dos churned (US$ 7,75M)**. A diferença não é erro de cálculo — são métricas que medem coisas diferentes.

| Métrica | O que representa | Valor |
|---|---|---|
| **CLTV total dos churned** | Valor vitalício total estimado, do primeiro mês até o último possível | US$ 7.755.256 |
| **Total Charges já pagos** | Receita que esses clientes JÁ pagaram antes de cancelar | US$ 2.862.927 |
| **Receita futura destruída** | Valor que AINDA seria gerado (CLTV − já pago, clip em zero) | US$ 5.163.882 |

```
CLTV total       =  US$ 7.755.256    (valor vitalício estimado, integral)
                 = Total Charges já pagos (US$ 2.862.927)
                 + Receita futura não realizada (US$ 5.163.882)
                 - Ajuste (US$ 271.553, quem já ultrapassou o CLTV)
```

**Em palavras simples:** o CLTV é uma estimativa do valor total que o cliente geraria desde sua entrada. Quando um cliente sai, uma parte desse valor **já foi capturada** (os Total Charges recebidos até a data do cancelamento) — essa receita não pode ser "perdida" porque já está no caixa da empresa.

O que de fato se perde com o churn é a **parte ainda não realizada** do CLTV — a receita futura que o cliente geraria se continuasse. Esse é o número relevante para o modelo de ML: **US$ 5,16M**. É o valor que o projeto tenta recuperar.

### 2.3 Cenário de equilíbrio (regime permanente)

Assumindo plateau da base de clientes (entradas = saídas):

- **Churners/mês:** ~114 clientes (2,21% da base ativa)
- **Churners/ano em regime:** ~1.371 clientes (26,5% da base ativa)
- **Perda anual esperada:** ~US$ 3,79M em receita futura destruída

---

## 3. Trade-off Falso Negativo × Falso Positivo

### 3.1 Custo do Falso Negativo

> **Situação:** modelo classifica como "vai ficar" → cliente sai sem ação preventiva → receita futura perdida.

```
custo_FN = max(CLTV − Total Charges, 0)
```

| Estatística | Valor |
|---|---|
| **Mediana (valor adotado)** | **US$ 2.845,25** |
| Média | US$ 2.762,91 |
| P25 | US$ 1.426,45 |
| P75 | US$ 4.165,65 |
| Churners com custo FN = 0 | 222 (11,9%) — já ultrapassaram o CLTV |

### 3.2 Custo do Falso Positivo

> **Situação:** modelo classifica como "vai sair" → empresa dispara campanha → cliente ficaria de qualquer forma.

A campanha de retenção consiste em oferecer migração para plano anual com **desconto de 10%**:

```
custo_FP = ticket_mensal_ativo × 12 × 10%
         = US$ 61,27 × 12 × 10%
         = US$ 73,52
```

O custo incide **apenas sobre clientes que aceitam** a oferta — não sobre todos os alertas emitidos.

### 3.3 Razão de assimetria

```
Razão FN/FP = US$ 2.845,25 / US$ 73,52 = 38,7x
```

**Errar um Falso Negativo custa aproximadamente 30 vezes mais que errar um Falso Positivo.** Essa é a âncora técnica: o modelo deve priorizar **Recall** sobre Precision.

### 3.4 Variação por faixa de tenure

A razão FN/FP não é uniforme — depende de quanto tempo o cliente ainda ficaria:

| Faixa | N | % churners | CLTV resid. mediana | Razão FN/FP |
|---|---|---|---|---|
| **0 – 6 meses** | 784 | 41,9% | US$ 3.980 | **54,1x** |
| **7 – 12 meses** | 253 | 13,5% | US$ 3.241 | **44,1x** |
| 13 – 24 meses | 294 | 15,7% | US$ 2.537 | 34,5x |
| 25 – 48 meses | 325 | 17,4% | US$ 771 | 10,5x |
| 49 – 72 meses | 213 | 11,4% | US$ 0 | 0,0x¹ |

> ¹ Mediana zero porque 58% desse grupo já ultrapassou o CLTV estimado.

**Insight estratégico:** 55,5% dos churners saem nos primeiros 12 meses — são o alvo principal do modelo, com razão FN/FP entre 44x e 54x. A feature `Tenure Months` deve ter peso relevante na arquitetura.

---

## 4. Métricas Técnicas e Custo de Churn Evitado

### 4.1 Hierarquia de métricas

| Métrica | Papel | SLO mínimo |
|---|---|---|
| **Recall (Churn)** | Operacional primária — cobertura de risco | **≥ 0,70** |
| **PR-AUC** | Técnica primária — desempenho na classe minoritária | ≥ 0,65 |
| **AUC-ROC** | Técnica complementar — ranking | ≥ 0,85 |
| **F1-Score (Churn)** | Comparativa entre modelos — nunca para threshold | ≥ 0,62 |
| **Precision (Churn)** | Controle operacional — evita sobrecarga da campanha | ≥ 0,55 |
| **MCC** | Robustez geral | ≥ 0,44 |

### 4.2 Justificativa técnica

**Por que Recall é a métrica operacional primária:** com razão FN/FP de ~30x, cada ponto percentual de recall a mais detecta um churner adicional (US$ 2.845 salvos) ao custo marginal de alguns alertas extras (US$ 73,52 cada). A conta sempre favorece aumentar recall — até o limite da capacidade operacional da equipe de retenção.

**Por que PR-AUC é priorizada sobre AUC-ROC:** em datasets desbalanceados, a curva ROC pode mascarar fragilidades na classe positiva. PR-AUC mede diretamente o desempenho onde importa. Ambas são relevantes — PR-AUC é a primária por ser mais conservadora.

**Por que F1 não define o threshold:** o F1 trata FP e FN com peso igual, ignorando a assimetria 30:1 dos custos reais. Usar F1 como critério de threshold pode elevar significativamente o custo total.

### 4.3 Benchmarks de referência (literatura IBM Telco)

| Modelo | AUC-ROC esperado | Recall esperado | F1 esperado |
|---|---|---|---|
| DummyClassifier (piso) | 0,50 | 0,00 | 0,00 |
| Regressão Logística | ~0,84 | ~0,55 | ~0,60 |
| **Meta MLP PyTorch** | **≥ 0,85** | **≥ 0,70** | **≥ 0,62** |

### 4.4 Custo de Churn Evitado — quanto a empresa pode recuperar

A métrica de negócio primária quantifica, em dólares, a receita que o modelo permite recuperar em comparação ao cenário sem intervenção.

```
Custo de Churn Evitado (anual) =
    (n_churners × recall × conversão) × (CLTV_residual_médio − custo_desconto)
```

**Parâmetros:**
- Churners esperados por ano (regime permanente): 1.371
- CLTV residual médio por churner: US$ 2.762,91
- Custo do desconto por cliente retido: US$ 73,52
- Receita líquida por retido: US$ 2.689,39

### 4.5 Cenários de recuperação anual

Projeção assumindo diferentes combinações de desempenho do modelo (Recall) e conversão da campanha:

| Recall | Conversão | Retidos/ano | Receita bruta | Custo dos descontos | **Receita líquida anual** |
|---|---|---|---|---|---|
| 50% | 40% | 274 | US$ 758K | US$ 20K | US$ 738K |
| **70%** | **50%** | **480** | **US$ 1.326K** | **US$ 35K** | **US$ 1.291K** |
| 80% | 50% | 548 | US$ 1.514K | US$ 40K | US$ 1.474K |
| 90% | 50% | 617 | US$ 1.704K | US$ 45K | **US$ 1.659K** |

### 4.6 Cenário alvo (SLO de Recall ≥ 70% · Conversão 50%)

| Indicador | Valor |
|---|---|
| Churners esperados/ano | 1.371 |
| Detectados pelo modelo | 960 |
| Efetivamente retidos (após conversão) | 480 |
| Receita bruta recuperada | US$ 1.326.000 |
| Custo dos descontos concedidos | US$ 35.000 |
| **Receita líquida anual recuperada** | **US$ 1.291.000** |
| Equivalente mensal | US$ 107.550 |
| **% da perda anual recuperada** | **34,1%** |

**Comparação com status quo:**

- Sem modelo (hoje): perda anual de ~US$ 3,79M em receita futura
- Com modelo atingindo SLO: recuperação líquida de US$ 1,29M/ano
- Projeto paga custo fixo mensal (~US$ 500) com folga de **200x**

---

## 5. Threshold Operacional

### 5.1 Lógica de seleção

O threshold operacional será determinado após o treinamento do modelo real (Etapa 2), aplicando a lógica de minimização de custo total sobre a curva de probabilidades do MLP:

```python
threshold_otimo = argmin_{t ∈ [0,1]}  Σ custo_FN_individual(t) + n_FP(t) × 73.52

onde:
    custo_FN_individual(i) = max(CLTV_i − Total_Charges_i, 0)
    para cada cliente i classificado como FN no threshold t
```

### 5.2 Referências operacionais

**Threshold a evitar:**

- **t = 0,50 (default):** ignora a assimetria de custo — opera como se FN e FP tivessem mesmo peso
- **t = argmax F1:** otimiza equilíbrio aritmético entre Precision e Recall, mas ignora a razão 30:1 de custo real

**Threshold recomendado:**

- O menor valor que atenda ao SLO de Recall ≥ 0,70 **e** minimize custo total
- Estimativa inicial: entre 0,30 e 0,45 para modelos com AUC-ROC ~0,85

### 5.3 Processo de escolha (Etapa 2)

1. Treinar modelos e obter curva de probabilidades para validação
2. Para cada threshold candidato: calcular `custo_total(t)` usando CLTV residual individual
3. Filtrar candidatos que atendam Recall ≥ 0,70
4. Dentro do subconjunto filtrado, escolher o de menor `custo_total`
5. Validar escolha em teste cego (holdout estratificado)

---

## 6. Model Card

### 6.1 Identificação

| Campo | Valor |
|---|---|
| Nome | `telco_churn_predictor` |
| Versão | 1.0.0 (a ser emitida após Etapa 2) |
| Tipo | Classificador binário supervisionado |
| Arquitetura | MLP (Multilayer Perceptron) com PyTorch — baselines: Dummy, Logistic Regression |
| Responsáveis | Equipe MLOps do Tech Challenge |

### 6.2 Uso pretendido

- **Uso primário:** gerar score de risco diário para clientes ativos, priorizando abordagens de retenção
- **Usuários primários:** time de Retenção / CRM, Customer Success
- **Fora de escopo:**
  - Decisões automatizadas sem revisão humana
  - Comunicação direta ao cliente sobre seu score
  - Segmentação de marketing
  - Predição para outros segmentos (B2B, enterprise)
  - Generalização para outras regiões geográficas (modelo treinado com dados concentrados em um estado)

### 6.3 Performance esperada

| Métrica | Meta mínima | Meta objetivo |
|---|---|---|
| AUC-ROC | 0,80 | 0,85 |
| PR-AUC | 0,60 | 0,70 |
| Recall (Churn) | 0,70 | 0,75 |
| Precision (Churn) | 0,55 | 0,60 |
| F1-Score (Churn) | 0,62 | 0,65 |

### 6.4 Dados de treinamento

- **Fonte:** IBM Telco Customer Churn (7.043 clientes)
- **Balanceamento:** 73,5% não-churn · 26,5% churn
- **Divisão:** 80/20 estratificado (StratifiedShuffleSplit, seed=42)
- **Validação cruzada:** StratifiedKFold 5-fold (shuffle=True, seed=42)
- **Features:** 20 (3 numéricas + 17 categóricas); identificadores e geográficas descartados
- **Tratamento de missing:** 11 valores em `Total Charges` imputados pela mediana

### 6.5 Limitações conhecidas

- **Snapshot estático:** dataset sem série temporal — não captura sazonalidade ou drift evolutivo
- **Concentração regional:** dados de um único estado — generalização para outras regiões **não garantida**; features geográficas foram removidas para não ancorar o modelo em padrões locais
- **Viés de produto:** apenas serviços de telecom residencial — não se aplica a B2B ou enterprise
- **Tenure curto subrepresentado:** clientes com tenure < 2 meses têm poucos dados para aprendizado, apesar de serem o grupo de maior risco (42% dos churners)
- **Feature CLTV:** é uma estimativa prospectiva do próprio dataset — será monitorada quanto a possível leakage

### 6.6 Vieses potenciais monitorados

| Dimensão | Variável | Monitoramento |
|---|---|---|
| Idade | `Senior Citizen` | Recall diferencial entre grupos |
| Gênero | `Gender` | Taxa de FP por grupo |
| Parceiro/dependentes | `Partner`, `Dependents` | Disparidade de tratamento |
| Tipo de internet | `Internet Service` | Clientes Fiber optic têm churn 2x maior — validar que o modelo não amplifica |
| Método de pagamento | `Payment Method` | Electronic check tem churn 3x maior — investigar causalidade vs correlação |

### 6.7 Cenários de falha

**Falha silenciosa — modelo degradado sem alerta:**
- Causa provável: drift nos dados de entrada
- Detecção: PSI > 0,20 em features críticas (`Tenure Months`, `Monthly Charges`, `Contract`)
- Mitigação: retreinamento emergencial; fallback para regra de negócio manual

**Falha de viés — grupo demográfico com performance pior:**
- Causa provável: subamostragem no treino
- Detecção: diferença de recall > 10 p.p. entre grupos sensíveis
- Mitigação: rebalanceamento com pesos amostrais; auditoria de features

**Falha de dependência externa — API de billing indisponível:**
- Causa provável: indisponibilidade de feature essencial (`Monthly Charges`)
- Detecção: schema validation falha no pipeline
- Mitigação: fallback para regra de negócio (`Contract = Month-to-month` + `Tenure < 12`)

**Falha por deslocamento de premissa — custo FP muda:**
- Causa provável: negócio altera a política de desconto
- Detecção: revisão trimestral dos parâmetros
- Mitigação: recalcular threshold ótimo com novo `custo_fp`; versionar o modelo

**Falha de generalização geográfica:**
- Causa provável: aplicação do modelo em região não representada no treino
- Detecção: drift significativo em todas as features simultaneamente
- Mitigação: não aplicar o modelo fora do perfil do treino até coleta de dados representativos

---

## 7. Arquitetura de Deploy

### 7.1 Modalidade escolhida: **Batch diário** como primária, **Real-time** como secundária

**Justificativa:** churn é um fenômeno de semanas a meses — não requer decisão em milissegundos. O time de retenção opera com uma lista priorizada diária, não com alertas instantâneos. Três fatores reforçam a escolha:

- **Volume:** toda a base ativa (~5.000 clientes) processada em janela overnight
- **Consistência:** mesma versão do modelo aplicada a todos no mesmo dia
- **Custo:** infraestrutura mínima (job agendado) vs serviço always-on
- **Auditabilidade:** output diário versionado facilita análise retroativa

O endpoint real-time é mantido para casos específicos: atendimento reativo quando o cliente liga, pré-venda de novo plano, análise pontual de uma conta.

### 7.2 Arquitetura batch

```
┌───────────────┐    02:00 UTC    ┌───────────────┐    ┌──────────────┐
│  Data Lake    │ ───────────────►│  Batch Job    │───►│   MLflow     │
│  (CRM/Bill)   │   daily sync    │  (Docker)     │    │  Registry    │
└───────────────┘                 └───────┬───────┘    └──────────────┘
                                          │
                                          ▼
                                  ┌───────────────┐
                                  │ predictions   │
                                  │ {date}.parquet│
                                  └───────┬───────┘
                                          │
                                          ▼
                                  ┌───────────────┐
                                  │  CRM/Salesforce│
                                  │  (via API)    │
                                  └───────────────┘
```

**SLOs batch:**
- Janela de execução: 02:00 – 04:00 UTC
- SLO de conclusão: 95% das execuções < 2h
- SLO de disponibilidade: 99,5% de execuções bem-sucedidas por mês

### 7.3 Arquitetura real-time

```
┌──────────────┐   HTTPS   ┌──────────────┐    ┌──────────────┐
│   Consumer   │──────────►│   FastAPI    │───►│  Model.pkl   │
│  (CRM/App)   │ /predict  │  (uvicorn)   │    │  (em memória)│
└──────────────┘           └──────┬───────┘    └──────────────┘
                                  │
                                  ▼
                          ┌───────────────┐
                          │  Structured   │
                          │    Logging    │
                          └───────────────┘
```

**SLOs real-time:**
- Latência p95: < 200 ms
- Latência p99: < 500 ms
- Taxa de erro: < 0,1%
- Disponibilidade: 99,9%

### 7.4 Endpoints da API

| Endpoint | Método | Propósito |
|---|---|---|
| `/predict` | POST | Predição individual com validação Pydantic |
| `/predict/batch` | POST | Lote de até 1.000 clientes por chamada |
| `/health` | GET | Verifica se a API e o modelo estão operacionais |
| `/metrics` | GET | Métricas Prometheus (latência, contadores) |
| `/version` | GET | Versão do modelo em produção |

---

## 8. Plano de Monitoramento

### 8.1 SLOs técnicos (modelo)

| Métrica | SLO | Frequência | Ação se violado |
|---|---|---|---|
| AUC-ROC | ≥ 0,80 | Semanal | Alerta + investigação em 48h |
| Recall (Churn) | ≥ 0,65 | Semanal | Revisão de threshold |
| PSI (drift) | < 0,20 | Semanal | Retreinamento se > 0,25 |
| Custo total estimado | dentro de ±15% do baseline | Mensal | Revisão de premissas |

### 8.2 SLOs de infraestrutura

| Métrica | SLO | Frequência | Ação se violado |
|---|---|---|---|
| Latência API (p95) | < 200 ms | Contínuo | Scale-up automático |
| Taxa de erro API | < 0,1% | Contínuo | Alerta imediato |
| Job batch completado | antes das 04:00 UTC | Diário | Investigação no dia seguinte |
| Uso de memória | < 80% | Contínuo | Scale-up |

### 8.3 SLOs de negócio

| Métrica | SLO | Frequência |
|---|---|---|
| Receita líquida recuperada | ≥ US$ 90.000 / mês | Mensal |
| Taxa de conversão da campanha | ≥ 40% | Mensal |
| Custo médio por cliente retido | ≤ US$ 75 | Mensal |
| ROI da campanha de retenção | ≥ 5:1 | Trimestral |

### 8.4 Playbook de resposta

**Alerta: AUC-ROC abaixo de 0,80 por 2 semanas consecutivas**
1. Executar `eval_model.py` com dados dos últimos 60 dias
2. Comparar distribuição das features atuais vs treino (PSI por feature)
3. Se PSI > 0,25 em ≥ 3 features → disparar retreinamento
4. Caso contrário, investigar possível shift no rótulo (conceito de churn mudou?)

**Alerta: Receita recuperada < US$ 60.000 por 2 meses consecutivos**
1. Verificar se a equipe de retenção está operando as listas diárias
2. Auditar taxa de conversão por tipo de ação
3. Revisar threshold — pode estar muito restrito, reduzindo recall
4. Se necessário, retreinar com dados mais recentes

---

## 9. Próximos Passos (Etapas 2–4)

**Etapa 2 — Modelagem com Redes Neurais**
- [ ] Implementar MLP em PyTorch com early stopping e batching
- [ ] Comparar MLP × Logística × Random Forest usando as 6 métricas desta Etapa 1
- [ ] Determinar threshold operacional pelo critério de custo total mínimo
- [ ] Registrar todos os experimentos no MLflow com parâmetros, métricas e artefatos

**Etapa 3 — Engenharia e API**
- [ ] Refatorar em estrutura modular (`src/`)
- [ ] Pipeline sklearn reprodutível
- [ ] FastAPI com `/predict`, `/health`, validação Pydantic
- [ ] Testes automatizados ≥ 3: smoke, schema, API
- [ ] Logging estruturado, linting com ruff

**Etapa 4 — Documentação e Entrega**
- [ ] Atualizar este Model Card com números reais do modelo treinado
- [ ] README completo com instruções de setup e execução
- [ ] Vídeo STAR de 5 minutos
- [ ] (Opcional) Deploy em nuvem com endpoint público

---

## 10. Referências dos Cálculos

Todos os valores deste documento foram calculados com o dataset real `Telco_customer_churn.xlsx` (IBM, n = 7.043), excluindo features geográficas, identificadores e o `Churn Score` oracular.

```python
# Custo FN por cliente churned
custo_fn = max(CLTV - Total_Charges, 0)

# Custo FP por alerta aceito (desconto 10% do anual)
custo_fp = ticket_mensal_ativo * 12 * 0.10   # = US$ 73,52

# Razão de assimetria
razao_fn_fp = mediana(custo_fn) / custo_fp   # = 38,7x

# Custo de Churn Evitado (anual, regime permanente)
custo_evitado_anual = (
    n_churners_ano * recall * conversao
) * (cltv_residual_medio - custo_fp)

# Threshold ótimo (a ser aplicado ao modelo treinado na Etapa 2)
def custo_total(t, y_true, y_score, custo_fn_individual, custo_fp):
    y_pred = (y_score >= t).astype(int)
    idx_fn = (y_true == 1) & (y_pred == 0)
    idx_fp = (y_true == 0) & (y_pred == 1)
    return custo_fn_individual[idx_fn].sum() + idx_fp.sum() * custo_fp

t_otimo = argmin_{t ∈ [0.05, 1.00]} custo_total(t, ...)
```

---

*Documento gerado na Etapa 1 do Tech Challenge — MLOps · Predição de Churn · POSTECH*
*Referência: ML Canvas v0.1 (Louis Dorard) · machinelearningcanvas.com*
*Todos os valores calculados com dataset IBM Telco Customer Churn (n = 7.043)*
*Features geográficas, identificadores e Churn Score excluídos para evitar viés e data leakage*