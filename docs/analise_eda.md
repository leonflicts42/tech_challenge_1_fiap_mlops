# Relatório — Análise Exploratória de Dados (EDA)

**Notebooks:** `0.01` · `0.02` · `0.03` · `0.04`
**Dataset:** IBM Telco Customer Churn — California Q3
**Entrada:** `data/raw/raw_telco_customer_churn.xlsx` — shape (7043, 33)
**Saída:** `data/interim/telco_typed.parquet` — shape (7043, 20)
**Status:** concluído com sucesso

---

## Resumo executivo

| Etapa | Notebook | Principal achado |
|---|---|---|
| Data understanding | `0.01` | 11 nulos em `Total Charges`, desbalanceamento 73/27, leakage isolado |
| EDA univariada | `0.02` | Distribuições não normais, `Phone Service` e `Senior Citizen` com alta concentração |
| EDA bivariada | `0.03` | `Tenure` é o maior preditor numérico (Cohen's d = -0.89), `Contract` o maior categórico (V = 0.41) |
| EDA multivariada | `0.04` | Multicolinearidade `Tenure` × `Total Charges` (corr = 0.83, VIF = 8.08) |

---

## 1. Data understanding (0.01)

### Dimensionalidade e estrutura

O dataset IBM Telco contém 7.043 clientes e 33 variáveis brutas. Após categorização semântica, as colunas foram separadas em quatro grupos:

| Grupo | Colunas | Destino |
|---|---|---|
| `COLS_ID` | 9 | removidas — sem poder preditivo |
| `COLS_NUM` | 3 | features do modelo |
| `COLS_CAT` | 16 | features do modelo |
| `COLS_POS` | 3 | removidas — data leakage |

O dataset reduzido ficou com shape `(7043, 20)`, persistido em `data/interim/telco_typed.parquet`.

### Qualidade dos dados

`Total Charges` chegou como `object` no raw e foi convertida para `float64` via `pd.to_numeric(errors='coerce')`, gerando 11 nulos (0.16%). A validação confirmou que todos os 11 registros possuem `Tenure Months = 0` — clientes recém-chegados sem nenhuma cobrança acumulada. A estratégia de imputação (mediana) foi aplicada no `1.01`.

`Churn Reason` apresentou 73.46% de nulos — comportamento esperado, pois o motivo de cancelamento só existe para clientes que efetivamente saíram. Isolada em `COLS_POS`.

Zero duplicatas detectadas nesta etapa (as 22 duplicatas estruturais só foram identificadas no `1.01` após a remoção do `CustomerID`).

### Desbalanceamento de classes

| Classe | Contagem | % |
|---|---|---|
| Não cancelou (0) | 5.174 | 73.46% |
| Cancelou (1) | 1.869 | 26.54% |
| Ratio | 0.36 | — |

O ratio de 0.36 confirma desbalanceamento moderado. Estratégia definida: SMOTE+ENN aplicado exclusivamente no conjunto de treino, nunca no teste ou validação.

### Isolamento de data leakage

As três colunas `COLS_POS` foram identificadas e isoladas antes de qualquer análise:

- `Churn Score` — score calculado pelo IBM SPSS após o evento de churn
- `CLTV` — Customer Lifetime Value calculado com dados do cliente já encerrado
- `Churn Reason` — motivo registrado apenas depois do cancelamento

Usar qualquer dessas colunas como feature causaria vazamento do futuro para o modelo.

---

## 2. EDA univariada (0.02)

### Variáveis numéricas

| Variável | Média | Mediana | Std | Skewness | Curtose | Outliers IQR | Outliers Z | Normal (Shapiro) |
|---|---|---|---|---|---|---|---|---|
| `Tenure Months` | 32.37 | 29.00 | 24.56 | 0.240 | -1.387 | 0 (0%) | 0 (0%) | Não (p≈0) |
| `Monthly Charges` | 64.76 | 70.35 | 30.09 | -0.221 | -1.257 | 0 (0%) | 0 (0%) | Não (p≈0) |
| `Total Charges` | 2.283 | 1.397 | 2.267 | 0.962 | -0.232 | 0 (0%) | 0 (0%) | Não (p≈0) |

**Achados relevantes:**

Nenhuma das três variáveis segue distribuição normal (Shapiro-Wilk p≈0 para todas). `Tenure Months` e `Monthly Charges` são platicúrticas (caudas leves, distribuição achatada). `Total Charges` apresenta assimetria à direita significativa (skewness = 0.962) — a maioria dos clientes tem cobranças totais baixas, enquanto uma minoria acumula valores muito altos ao longo dos anos.

Ausência completa de outliers extremos em todas as numéricas (IQR e Z-score) — não será necessário aplicar clipping ou técnicas de winsorização.

**Decisão técnica:** `log1p` em `Total Charges` no `1.02` para reduzir assimetria. `StandardScaler` para todas as numéricas no `1.03`.

### Variáveis categóricas

Nenhuma das 16 variáveis categóricas apresentou nulos ou categorias raras (< 1%). Dois alertas relevantes:

**Alta concentração — baixa variância:**

- `Phone Service`: 90.3% dos clientes possuem o serviço → baixo poder discriminativo
- `Senior Citizen`: 83.8% não são idosos → variável dominada por uma categoria

**Distribuição estratégica:**

- `Contract Month-to-month`: domina 55% da base — historicamente o maior preditor de churn em datasets Telco

**Redundância semântica identificada:** seis colunas de serviços de internet continham a categoria `"No internet service"` semanticamente equivalente a `"No"`. Normalização aplicada no `1.01` (1.526 registros por coluna).

---

## 3. EDA bivariada (0.03)

### Variáveis numéricas vs target

| Variável | Média churn=0 | Média churn=1 | Cohen's d | Mann-Whitney p | Efeito |
|---|---|---|---|---|---|
| `Tenure Months` | 37.57 | 17.98 | -0.89 | < 0.001 | Grande |
| `Monthly Charges` | 61.27 | 74.44 | +0.47 | < 0.001 | Médio |
| `Total Charges` | 2.555 | 1.532 | -0.48 | < 0.001 | Médio |

**`Tenure Months` — a variável de ouro:**

Cohen's d = -0.89 (efeito grande) — o maior separador numérico entre as classes. Clientes que cancelam têm mediana de apenas 10 meses de casa, contra 38 meses dos que ficam. Isso evidencia uma janela crítica de risco nos primeiros 12 meses do relacionamento. Feature a preservar sem transformações de corte.

**`Monthly Charges`:**

Clientes que saem pagam em média R$ 74.44 contra R$ 61.27 dos que ficam (Cohen's d = 0.47). O churn está concentrado em clientes de ticket mais alto — sensibilidade ao preço ou percepção de valor inadequada em planos premium.

**`Total Charges`:**

Valor total menor no grupo churn porque o tempo de casa é curto (relação matemática com Tenure). O log identificou 5.83% de outliers no grupo churn — reforça a necessidade de `log1p` antes do StandardScaler.

### Variáveis categóricas vs target

| Variável | Cramer's V | Chi² p | Categoria crítica | Churn (%) |
|---|---|---|---|---|
| `Contract` | 0.41 | < 0.001 | Month-to-month | 42.71% |
| `Online Security` | 0.35 | < 0.001 | No | 41.77% |
| `Tech Support` | 0.34 | < 0.001 | No | 41.64% |
| `Internet Service` | 0.32 | < 0.001 | Fiber optic | 41.89% |
| `Payment Method` | 0.30 | < 0.001 | Electronic check | 45.29% |
| `Online Backup` | 0.29 | < 0.001 | No | 39.93% |
| `Device Protection` | 0.28 | < 0.001 | No | 39.13% |
| `Dependents` | 0.25 | < 0.001 | No | 32.55% |
| `Streaming TV` | 0.23 | < 0.001 | No | 33.52% |
| `Streaming Movies` | 0.23 | < 0.001 | No | 33.68% |
| `Paperless Billing` | 0.19 | < 0.001 | Yes | 33.57% |
| `Senior Citizen` | 0.15 | < 0.001 | Yes | 41.68% |
| `Partner` | 0.15 | < 0.001 | No | 32.96% |
| `Multiple Lines` | 0.04 | 0.003 | Yes | 28.61% |
| `Phone Service` | 0.01 | 0.339 | — | p > 0.05 |
| `Gender` | 0.008 | 0.487 | — | p > 0.05 |

**Preditores categóricos mais fortes:**

`Contract` é o maior preditor categórico (V = 0.41). Contratos mensais têm churn de 42.71% contra apenas 2.83% nos contratos de dois anos — a ausência de multa rescisória remove a barreira de saída.

`Online Security` e `Tech Support` (V ~0.34) funcionam como âncoras de retenção — clientes sem esses serviços têm churn acima de 41%.

`Internet Service Fiber Optic` (V = 0.32) — churn alarmante de 41.89%. Hipótese: preço elevado ou qualidade técnica insatisfatória no serviço premium.

`Payment Method Electronic Check` (V = 0.30) — maior taxa de churn absoluta (45.29%). Métodos de pagamento manuais ou instáveis facilitam a saída.

**Candidatos ao descarte:**

`Gender` (p = 0.487, V = 0.008) e `Phone Service` (p = 0.339, V = 0.011) — sem diferença estatisticamente significativa entre grupos. Serão mantidos no primeiro baseline para confirmação empírica via feature importance antes do descarte definitivo.

---

## 4. EDA multivariada (0.04)

### Correlação das numéricas com o target

| Variável | Correlação com Churn |
|---|---|
| `Tenure Months` | -0.3522 |
| `Total Charges` | -0.1995 |
| `Monthly Charges` | +0.1934 |

### Multicolinearidade

**Par crítico detectado:** `Tenure Months` × `Total Charges` — correlação de 0.8259.

Relação matemática estrutural: `Total Charges ≈ Tenure × Monthly Charges`. Manter ambas as variáveis equivale a fornecer a mesma informação duas vezes ao modelo.

**Variance Inflation Factor (VIF):**

| Variável | VIF | Interpretação |
|---|---|---|
| `Total Charges` | 8.08 | Elevado — próximo do limite crítico (10) |
| `Tenure Months` | 6.33 | Atenção — inflado pela ligação com Total |
| `Monthly Charges` | 3.36 | Saudável — informação única |

**Decisão técnica:** manter `Total Charges` com `log1p` e `median imputer` — decisão alinhada com a literatura (Frontiers/2026, Nature/2025), que confirma `Total Charges` como uma das top features em modelos Telco. A multicolinearidade é mitigada pela transformação logarítmica e pelo fato de que a MLP aprende relações não-lineares que um modelo linear não capturaria.

---

## 5. Decisões técnicas consolidadas para o pipeline

### Feature engineering (1.02)

| Feature nova | Fórmula | Justificativa |
|---|---|---|
| `num_services` | soma de serviços ativos | âncora de retenção — custo de saída |
| `charges_per_month` | `monthly_charges / (tenure + 1)` | captura relação valor × tempo |
| `is_month_to_month` | flag `Contract == Month-to-month` | isola o maior preditor categórico |
| `tenure_group` | buckets novo/medio/longo | captura não-linearidade dos primeiros 12m |
| `has_security_support` | flag `online_security OR tech_support` | consolida âncoras de retenção (V ~0.34) |

### Preprocessing (1.03)

| Tipo | Colunas | Transformação |
|---|---|---|
| Numéricas | `tenure_months`, `monthly_charges`, `total_charges_log` | `StandardScaler` |
| Binárias | 11 colunas Yes/No | já encodadas 0/1 no `1.01` |
| Nominais | `internet_service`, `contract`, `payment_method`, `multiple_lines` | `OneHotEncoder(drop="first")` |
| Descarte pendente | `gender`, `phone_service` | confirmar após feature importance dos baselines |

### Desbalanceamento (1.04)

`SMOTE+ENN` aplicado exclusivamente no `X_train`. Jamais no `X_test` ou nos folds de validação.

### Métricas de avaliação

Prioridade: **Recall** e **F1-Score** da classe positiva (Churn=1).

- Falso Negativo: cliente que vai cancelar não identificado → receita perdida permanentemente
- Falso Positivo: campanha de retenção desnecessária → custo controlável

O custo assimétrico entre FN e FP justifica priorizar Recall sem abrir mão de F1 para evitar modelos que simplesmente predizem churn para todos.

---

## 6. Artefatos gerados

| Artefato | Localização |
|---|---|
| `churn_distribution.png` | `reports/figures/` |
| `missing_values.png` | `reports/figures/` |
| `univariate_num_*.png` (3) | `reports/figures/` |
| `univariate_skew_kurt_*.png` (3) | `reports/figures/` |
| `univariate_cat_*.png` (16) | `reports/figures/` |
| `bivariate_num_*_vs_target.png` (3) | `reports/figures/` |
| `bivariate_cat_*.png` (16) | `reports/figures/` |
| `correlation_matrix_numeric.png` | `reports/figures/` |
| `vif_table.csv` | `reports/figures/` |
| `telco_typed.parquet` | `data/interim/` |
| `data_understanding.md` | `references/` |

---

## 7. Próximo passo

**Notebook:** `1.02_feature_engineering`
**Entrada:** `data/interim/telco_cleaned.parquet` — shape (7021, 20)
**Objetivo:** criar as 5 features novas, aplicar `log1p` em `total_charges` e persistir
`data/interim/telco_features.parquet`