# ML Canvas — Churn Prediction para Telecomunicações

> Baseado no **Machine Learning Canvas v0.1** de Louis Dorard  
> Referência: [machinelearningcanvas.com](http://www.machinelearningcanvas.com)  
> Projeto: Tech Challenge POSTECH — Fase 1

---

## BACKGROUND

### End-User (Usuário Final)

**Quem usará o sistema preditivo / quem será afetado por ele?**

| Ator | Papel |
|---|---|
| **Time de Retenção / CRM** | Usuário direto — recebe a lista priorizada de clientes em risco e executa campanhas de retenção (ofertas, descontos, upgrade de plano) |
| **Gerente de Customer Success** | Monitora KPIs de churn e acompanha a eficiência das ações de retenção desencadeadas pelo modelo |
| **Analista de Dados / MLOps** | Opera, monitora e retreina o modelo; acompanha métricas técnicas via MLflow |
| **Diretoria Comercial** | Stakeholder estratégico — toma decisões orçamentárias baseadas nas previsões de churn futuro |
| **Clientes em risco** | Afetados indiretamente — recebem abordagem de retenção proativa (oferta, contato comercial) |

**Restrições e considerações éticas:**
- O modelo não é apresentado diretamente ao cliente; é uma ferramenta interna B2B.
- O cliente nunca é informado de que está sendo classificado como "alto risco de churn".
- Clientes de grupos demográficos vulneráveis (idosos, `SeniorCitizen = 1`) podem receber tratamento diferenciado — isso deve ser monitorado como viés potencial.

---

### Value Proposition (Proposta de Valor)

**O que estamos tentando fazer pelos usuários do sistema?**

> **"Antecipar quais clientes irão cancelar o serviço nos próximos 30 dias, permitindo que o time de retenção atue preventivamente antes que o cancelamento ocorra."**

| Objetivo | Métrica de Sucesso |
|---|---|
| Reduzir a taxa de churn voluntário | Redução ≥ 5 p.p. na taxa de churn mensal após implantação |
| Priorizar esforço de retenção | Top-decile lift ≥ 2,5× sobre o modelo sem ML |
| Reduzir custo de aquisição de novos clientes | Custo de retenção proativa < 30% do CAC (Customer Acquisition Cost) |
| Maximizar receita recorrente retida | ROI de campanha de retenção ≥ 3:1 (receita recuperada / custo da oferta) |

**O que o sistema NÃO faz:**
- Não explica o motivo do churn para o cliente.
- Não substitui a análise humana para casos ambíguos.
- Não realiza segmentação de marketing — apenas sinaliza risco.

---

### Data Sources (Fontes de Dados)

**De onde obtemos os dados?**

| Fonte | Descrição | Disponibilidade |
|---|---|---|
| **CRM Interno** | Dados contratuais: tipo de contrato, tenure, serviços ativos | Produção |
| **Sistema de Billing** | Cobranças mensais, forma de pagamento, inadimplência | Produção |
| **Sistema de Atendimento** | Chamados abertos, tempo de resolução, NPS parcial | Produção (futuro) |
| **Dataset Público IBM Telco** | Snapshot histórico de 7 043 clientes — usado para desenvolvimento e treinamento inicial | [Kaggle / IBM Watson](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |

**Dataset de referência para o Tech Challenge:**
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- 7 043 registros × 21 colunas
- Período: snapshot estático (sem série temporal)
- Licença: IBM Sample Data (uso educacional)

---

## ENGINE SPECS

### Problem (Especificação do Problema)

| Item | Descrição |
|---|---|
| **Pergunta a predizer** | "Este cliente vai cancelar o contrato nos próximos 30 dias?" |
| **Input (parâmetros)** | Perfil demográfico, dados contratuais, serviços ativos, cobranças mensais e totais (20 features tabulares) |
| **Outputs possíveis** | `Churn = 1` (vai cancelar) ou `Churn = 0` (permanece); mais probabilidade contínua `P(Churn)` ∈ [0, 1] |
| **Tipo de problema** | Classificação binária supervisionada com desbalanceamento moderado (~26,5% positivos) |

**Baseline alternativo (sem ML):**
> Regra de negócio manual: *"Sinalizar como churn todo cliente com contrato mês-a-mês + MonthlyCharges > R$ 65 + tenure < 12 meses."*  
> Estimativa: Recall ~40%, Precision ~50% — inferior à Regressão Logística (Recall ~56%, Precision ~64%).

**Hierarquia de modelos a explorar:**

```
Etapa 1: DummyClassifier (piso) → Regressão Logística (baseline linear)
Etapa 2: MLP PyTorch (modelo principal) + Random Forest / GBM (comparação)
Etapa 3: Modelo final serializado → servido via FastAPI
```

---

### Performance Evaluation (Avaliação de Performance)

#### Métricas de Negócio (Bottom-line, monitoramento em produção)

| Métrica | Fórmula | Meta | Periodicidade |
|---|---|---|---|
| **Taxa de Churn Evitado** | Churners retidos / Churners previstos pelo modelo | ≥ 40% de retenção efetiva | Mensal |
| **ROI da Campanha** | (Receita Retida − Custo da Oferta) / Custo da Oferta | ≥ 3:1 | Mensal |
| **Custo por Falso Positivo** | Custo da oferta de retenção enviada a não-churners | < R$ 50/cliente | Mensal |
| **NPV (Negative Predictive Value)** | TN / (TN + FN) | ≥ 0.88 | Semanal |
| **Top-Decile Lift** | Concentração de churners no top 10% de risco | ≥ 2,5× | Mensal |

#### Métricas Técnicas (Acurácia Preditiva)

| Métrica | Justificativa | Meta (Produção) |
|---|---|---|
| **AUC-ROC** *(primária)* | Métrica padrão da literatura para ranking de churn; independente do threshold | ≥ 0.860 |
| **PR-AUC** | Mais informativa que AUC-ROC para classes desbalanceadas | ≥ 0.680 |
| **F1-Score (classe Churn)** | Equilíbrio entre Precision e Recall | ≥ 0.620 |
| **Recall (classe Churn)** | Maximiza detecção de verdadeiros churners | ≥ 0.600 |
| **MCC** | Métrica balanceada para desbalanceamento | ≥ 0.440 |

#### Método de Avaliação Offline

```
Divisão treino/teste: 80/20 estratificado (StratifiedShuffleSplit, random_state=42)
Validação cruzada: StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
Aplicada a: DummyClassifier, Regressão Logística, Random Forest, MLP
```

**Análise de threshold:** Para o modelo final, otimizar o threshold via curva Precision-Recall com base no custo relativo de FP vs. FN.

---

### Data Preparation (Preparação dos Dados)

#### Como obtemos os dados de treinamento?

- Dataset histórico estático: 7 043 clientes com label `Churn` já observado.
- Supervisionado: inputs (features do cliente) + output (Churn: 0/1).
- Sem série temporal — snapshot único.

#### Input Features

**Numéricas (3):**
- `tenure` — meses como cliente (forte preditor negativo de churn)
- `MonthlyCharges` — valor mensal cobrado (forte preditor positivo)
- `TotalCharges` — valor total acumulado (correlacionado com tenure)

**Categóricas — Serviços (9):**
- `PhoneService`, `MultipleLines`
- `InternetService` (DSL / Fiber optic / No)
- `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`
- `StreamingTV`, `StreamingMovies`

**Categóricas — Contrato/Billing (4):**
- `Contract` (Month-to-month / One year / Two year) — **maior preditor**
- `PaperlessBilling`, `PaymentMethod`

**Categóricas — Demográficas (4):**
- `gender`, `SeniorCitizen`, `Partner`, `Dependents`

**Coluna excluída:** `customerID` (identificador sem valor preditivo)

**Tratamento de missing:** 11 valores ausentes em `TotalCharges` → imputação pela mediana.

---

## INTEGRATION

### Using Predictions (Uso das Predições)

**Quando fazemos predições e quantas?**

| Modalidade | Frequência | Volume |
|---|---|---|
| **Batch diário** *(modo principal)* | 1× por dia, às 02h00 | ~7 000–50 000 clientes/execução |
| **Real-time via API** *(modo secundário)* | Sob demanda — pré-venda, atendimento | 1 requisição/call |

**Restrição de latência:**
- Batch: sem restrição rígida (janela de 2h disponível)
- Real-time: < 200 ms por requisição (endpoint `/predict` da FastAPI)

**Como usamos predições e valores de confiança?**

```
P(Churn) ≥ 0.70  →  Prioridade ALTA: contato imediato + oferta premium
P(Churn) 0.50–0.69  →  Prioridade MÉDIA: email + oferta padrão
P(Churn) 0.30–0.49  →  Prioridade BAIXA: monitoramento passivo
P(Churn) < 0.30  →  Nenhuma ação (cliente saudável)
```

**Fallback:** Se a API não responder em < 500 ms → regra de negócio manual (tipo de contrato + tenure).

---

### Learning Models (Treinamento e Atualização)

**Quando criamos/atualizamos modelos? Com quais dados?**

| Evento | Ação |
|---|---|
| **Lançamento inicial** | Treinamento completo no dataset histórico (7 043 registros) |
| **Retreinamento periódico** | Mensal, com janela deslizante dos últimos 12 meses de dados reais |
| **Drift detectado** | Retreinamento emergencial disparado por alerta de monitoramento |
| **Novo produto/plano** | Retreinamento manual após validação das novas features |

**Restrição de tempo para criação do modelo:**
- Retreinamento completo: < 4 horas (pipeline automatizado no MLflow)
- Validação e promoção: < 24 horas (aprovação humana obrigatória)

**Critérios para deploy do modelo:**

```
Critério mínimo (MUST):
  ✓ AUC-ROC (teste) ≥ 0.860
  ✓ PR-AUC (teste) ≥ 0.680
  ✓ Recall-Churn (teste) ≥ 0.600
  ✓ Sem data leakage detectado
  ✓ Testes automatizados passando (smoke, schema, API)
  ✓ Linting ruff sem erros

Critério de promoção vs. modelo anterior:
  ✓ AUC-ROC melhora ≥ 0.005 (diferença estatisticamente significativa)
  OU
  ✓ F1-Churn melhora ≥ 0.010

Critério de rejeição automática:
  ✗ AUC-ROC < 0.800 (pior que baseline logístico)
  ✗ Recall-Churn < 0.500 (menos da metade dos churners detectados)
```

---

*ML Canvas v0.1 — Louis Dorard © 2015 | machinelearningcanvas.com*  
*Adaptado para o Tech Challenge POSTECH MLOps — Fase 1*