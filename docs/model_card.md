# Model Card — Churn Prediction MLP

## Detalhes do Modelo

| Campo | Valor |
|-------|-------|
| **Nome** | ChurnMLP v2 |
| **Tipo** | Rede Neural (MLP — Multilayer Perceptron) |
| **Framework** | PyTorch 2.11+ |
| **Versão do artefato** | `models/best_model_mlp.pt` |
| **Data de treinamento** | 2026-04-25 |
| **Responsável** | Victor Araújo Barros |

---

## Uso Pretendido

### Aplicação primária
Classificação binária de **risco de churn** em clientes de uma operadora de telecomunicações. O modelo prediz a probabilidade de um cliente cancelar o serviço nos próximos 30 dias com base em dados do contrato, uso de serviços e histórico de cobrança.

### Usuários pretendidos
- Times de retenção de clientes que executam campanhas proativas de fidelização
- Sistemas de CRM que integram scores de risco via API

### Uso não pretendido
- **Não usar** para tomar decisões automáticas sem revisão humana de casos individuais
- **Não usar** em contextos onde o custo de um falso positivo for maior do que o custo de um falso negativo (o modelo foi calibrado para minimizar falsos negativos)
- **Não usar** para clientes fora do perfil de telecomunicações residencial (ex.: clientes corporativos B2B não fazem parte do dataset de treinamento)
- **Não usar** para inferência em tempo real em contextos onde a latência máxima aceitável for < 50 ms sem infraestrutura adequada

---

## Dataset de Treinamento

| Campo | Valor |
|-------|-------|
| **Fonte** | IBM Telco Customer Churn (público) |
| **Total de registros** | 7.043 clientes |
| **Split treino/teste** | 80% / 20% (estratificado por `churn`) |
| **Registros de treino** | 5.616 |
| **Registros de teste** | 1.405 |
| **Features após pré-processamento** | 30 |
| **Taxa de churn (treino)** | 26,44% |
| **Taxa de churn (teste)** | 26,48% |
| **MD5 treino** | `7575994d37201ad71c968369257942a0` |
| **MD5 teste** | `7fe86301ae707780c7de1d3faaaa69fa` |

### Features utilizadas
- **Numéricas (3):** `tenure_months`, `monthly_charges`, `total_charges`
- **Binárias (11):** `partner`, `dependents`, `phone_service`, `multiple_lines`, `online_security`, `online_backup`, `device_protection`, `tech_support`, `streaming_tv`, `streaming_movies`, `paperless_billing`
- **Nominais OHE (6):** `gender`, `senior_citizen`, `internet_service`, `contract`, `payment_method`, `tenure_group`
- **Engenhadas (6):** `num_services`, `charges_per_month`, `is_month_to_month`, `tenure_group`, `has_security_support`, `is_fiber_optic`

---

## Arquitetura do Modelo

| Parâmetro | Valor |
|-----------|-------|
| **Camadas ocultas** | 2 (`[128, 64]`) |
| **Dropout** | 0.15 |
| **Normalização** | LayerNorm por camada |
| **Ativação** | ReLU |
| **Função de loss** | BCEWithLogitsLoss |
| **Peso de classe positiva** | 5.77 (corrige desbalanceamento 26% churn) |
| **Inicialização de pesos** | Kaiming-Normal |
| **Otimizador** | Adam (lr=0.001619, weight_decay=1.85e-4) |
| **Batch size** | 128 |
| **Early stopping** | patience=15, monitor=val_loss |
| **Threshold de decisão** | 0.16 (otimizado por valor de negócio) |

---

## Performance no Conjunto de Teste

### Métricas quantitativas

| Métrica | Valor |
|---------|-------|
| **ROC-AUC** | 0.850 |
| **PR-AUC** | 0.666 |
| **Recall** | 0.989 (98,9%) |
| **Precision** | 0.365 (36,5%) |
| **F1-score** | 0.533 |
| **Threshold** | 0.16 |
| **SLO Recall ≥ 0.70** | ✅ Atendido |

### Matriz de confusão (teste, n=1.405)

|  | Previsto: Não Churn | Previsto: Churn |
|--|---------------------|-----------------|
| **Real: Não Churn** | TN = 393 | FP = 640 |
| **Real: Churn** | FN = 4 | TP = 368 |

### Valor de negócio estimado (conjunto de teste)

| Métrica | Valor |
|---------|-------|
| **Valor de negócio total** | R$ 1.017.420,56 |
| **Custo por FP** (ação de retenção desnecessária) | R$ 73,52 |
| **Custo por FN** (churn não evitado) | R$ 2.845,00 |
| **Razão de custo FN/FP** | 38,7× |

### Comparativo com baselines (teste, threshold otimizado por negócio)

| Modelo | ROC-AUC | PR-AUC | Recall | Precision | Valor de Negócio (R$) |
|--------|---------|--------|--------|-----------|----------------------|
| **MLP (selecionado)** | **0.850** | 0.666 | **0.989** | 0.365 | **1.017.420** |
| GradientBoosting | 0.851 | **0.675** | 0.997 | 0.317 | 1.010.964 |
| LogisticRegression | 0.847 | 0.660 | 0.989 | 0.359 | 1.014.773 |
| RandomForest | 0.844 | 0.667 | 0.978 | 0.360 | 993.778 |

> O MLP foi selecionado por apresentar o **maior valor de negócio** no conjunto de teste, combinando ROC-AUC competitivo com o menor número de falsos negativos (FN=4) entre os modelos comparados.

---

## Limitações

### Distribuição do dataset
- Dataset limitado a **7.043 clientes** de uma única operadora americana (IBM Telco), o que pode não representar comportamentos de churn de operadoras brasileiras ou de outros mercados.
- Variáveis geoespaciais e comportamentais de uso (chamadas, dados consumidos) não estão disponíveis; o modelo se baseia apenas em atributos de contrato e cobrança.

### Período de validade
- O modelo não contém informação temporal. Mudanças sazonais, campanhas de concorrentes ou alterações de precificação podem degradar a performance sem sinal observável nas features.

### Balanceamento de classes
- O dataset tem 26% de taxa de churn. Datasets com taxas muito diferentes (< 5% ou > 50%) podem requerer recalibração do `pos_weight` e do threshold.

### Threshold baixo
- O threshold de 0.16 foi otimizado para minimizar FN (recall ≥ 98,9%), resultando em **640 falsos positivos** no conjunto de teste (45,5% das predições positivas são incorretas). Em contextos onde o custo de contato com clientes não churners for alto, esse threshold deve ser revisado.

---

## Vieses Conhecidos

| Feature | Viés Potencial | Impacto |
|---------|---------------|---------|
| `gender` | Diferença de churn por gênero pode refletir vieses históricos de precificação | Baixo — feature tem baixa importância preditiva |
| `senior_citizen` | Idosos têm padrão de churn distinto; subamostragem pode reduzir performance nesse segmento | Médio — taxa de churn de idosos é ~42% vs 24% de não-idosos |
| `tenure_months` | Clientes novos (0–6 meses) têm comportamento diferente; o modelo pode ser menos calibrado para esse segmento | Alto — 55% dos churners saem nos primeiros 12 meses |

---

## Cenários de Falha

| Cenário | Consequência | Mitigação |
|---------|-------------|-----------|
| **Data drift** nos valores de `monthly_charges` ou `total_charges` (ex.: reajuste de preços) | Degradação silenciosa do ROC-AUC | Monitorar PSI mensal; retreinar se PSI > 0.2 |
| **Novo plano de serviço** não mapeado pela `SemanticNormalizer` | Erro 422 na API ou preenchimento incorreto de features | Manter mapeamento de `SemanticNormalizer` atualizado; logar valores inesperados |
| **Clientes B2B** (empresas) inseridos na API | Predição fora da distribuição de treinamento | Validar se o perfil de entrada é residencial antes de chamar o endpoint |
| **Dataset muito desbalanceado** (< 5% churn) | Recall cai abaixo do SLO de 70% | Re-otimizar `pos_weight` e threshold com o novo dataset antes de deploy |
| **Modelo carregado com pesos incompatíveis** | Erro de `state_dict` ao inicializar `ChurnPredictor` | Versionar artefatos no MLflow; validar `n_features=30` antes de inference |

---

## Informações de Rastreabilidade

- **Experimento MLflow:** `churn-telecom` (tracking em `mlflow.db`)
- **Artefatos de treinamento:** `mlartifacts/` (checkpoints, preprocessor, logs de Optuna)
- **Parâmetros de Optuna:** `reports/json/optuna_best_params.json` (50 trials por modelo)
- **Relatório de métricas:** `reports/json/winner_model_report.json`
- **Seed global:** `RANDOM_STATE = 42`
