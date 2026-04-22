# Relatório Técnico de EDA — Predição de Churn em Telecomunicações

**Projeto:** Tech Challenge Fase 1 — MLOps FIAP  
**Dataset:** IBM Telco Customer Churn  
**Gerado em:** 2025  
**Autor:** Victor A. B.  
**Versão do dado processado:** `data/interim/telco_typed.parquet` — shape `(7043, 20)`

---

## Sumário

1. [Visão Geral do Dataset](#1-visão-geral-do-dataset)
2. [Qualidade dos Dados](#2-qualidade-dos-dados)
3. [Distribuição da Variável Target](#3-distribuição-da-variável-target)
4. [Análise Univariada — Variáveis Numéricas](#4-análise-univariada--variáveis-numéricas)
5. [Análise Univariada — Variáveis Categóricas](#5-análise-univariada--variáveis-categóricas)
6. [Análise Bivariada — Numéricas vs. Target](#6-análise-bivariada--numéricas-vs-target)
7. [Análise Bivariada — Categóricas vs. Target](#7-análise-bivariada--categóricas-vs-target)
8. [Análise Multivariada e Multicolinearidade](#8-análise-multivariada-e-multicolinearidade)
9. [Decisões Técnicas para o Pipeline](#9-decisões-técnicas-para-o-pipeline)
10. [Síntese Estratégica](#10-síntese-estratégica)

---

## 1. Visão Geral do Dataset

O dataset bruto contém **7.043 registros e 33 colunas**, cobrindo o perfil demográfico, os serviços contratados e o histórico financeiro de clientes de uma operadora de telecomunicações.

Após a triagem inicial, **13 colunas foram descartadas** por representarem identificadores, coordenadas geográficas ou vazamento de dados em relação ao target:

```
Descartadas: CustomerID, Count, Country, State, City, Zip Code, Lat Long,
             Latitude, Longitude, Churn Score, CLTV, Churn Reason, Churn Label
```

O dataset de trabalho resultante possui **shape `(7043, 20)`**, distribuído em:

| Tipo              | Quantidade | Colunas                                              |
|-------------------|------------|------------------------------------------------------|
| Numéricas         | 3          | Tenure Months, Monthly Charges, Total Charges        |
| Categóricas       | 16         | Gender, Contract, Internet Service, e outros         |
| Target            | 1          | Churn Value (binário: 0 = ficou, 1 = cancelou)       |

---

## 2. Qualidade dos Dados

### 2.1 Valores Nulos

| Coluna         | Nulos | % do total | Origem e Estratégia de Tratamento                                        |
|----------------|-------|------------|--------------------------------------------------------------------------|
| Total Charges  | 11    | 0,16%      | Clientes com `Tenure Months = 0` (recém-chegados, sem cobrança acumulada). **Ação:** imputar com `0.0` no pipeline — ausência reflete estado real do negócio. |
| Churn Reason   | 5.174 | 73,46%     | Estruturalmente esperado: só clientes que cancelaram possuem motivo registrado. **Ação:** coluna excluída do escopo do modelo. |

> **Nota:** A coluna `Total Charges` chegou com dtype `object` no raw e foi convertida para `float64` via `pd.to_numeric(errors='coerce')`, gerando os 11 nulos identificados.

### 2.2 Duplicatas

```
Linhas duplicadas encontradas: 0
```

O dataset está íntegro. Nenhum tratamento de deduplicação é necessário.

---

## 3. Distribuição da Variável Target

| Classe              | Contagem | Proporção |
|---------------------|----------|-----------|
| Não cancelou (0)    | 5.174    | 73,46%    |
| Cancelou (1)        | 1.869    | 26,54%    |
| **Ratio positivo**  | —        | **0,36**  |

O **ratio de 0,36** confirma desbalanceamento moderado. Com a classe minoritária representando ~27% da base, o modelo ingênuo que classifica tudo como "não cancelou" atingiria 73% de acurácia sem aprender nenhum padrão real — tornando a **acurácia uma métrica inadequada** para este problema.

### Implicações para a Modelagem

O desbalanceamento reflete a realidade operacional: a maioria dos clientes não cancela. Isso torna o **Recall da classe positiva a métrica mais crítica**:

- **Falso Negativo (FN):** cliente que vai cancelar e não é identificado → receita perdida permanentemente.
- **Falso Positivo (FP):** campanha de retenção disparada para cliente que ficaria → custo controlável.

Essa assimetria de custos será formalizada no ML Canvas. As estratégias de mitigação a serem avaliadas são:
1. `class_weight='balanced'` nos modelos Scikit-Learn (Regressão Logística, baseline).
2. `pos_weight` no `BCEWithLogitsLoss` do PyTorch (MLP).
3. SMOTE como alternativa caso as métricas de Recall permaneçam insatisfatórias.

---

## 4. Análise Univariada — Variáveis Numéricas

### 4.1 Estatísticas Descritivas

| Feature          | n     | Média    | Mediana  | Std      | Min   | Max      | Q1      | Q3      | Outliers IQR | Outliers Z |
|------------------|-------|----------|----------|----------|-------|----------|---------|---------|--------------|------------|
| Tenure Months    | 7.043 | 32,37    | 29,00    | 24,56    | 0,00  | 72,00    | 9,00    | 55,00   | 0 (0,00%)    | 0 (0,00%)  |
| Monthly Charges  | 7.043 | 64,76    | 70,35    | 30,09    | 18,25 | 118,75   | 35,50   | 89,85   | 0 (0,00%)    | 0 (0,00%)  |
| Total Charges    | 7.032 | 2.283,30 | 1.397,47 | 2.266,77 | 18,80 | 8.684,80 | 401,45  | 3.794,74| 0 (0,00%)    | 0 (0,00%)  |

**Ponto positivo:** nenhuma das três variáveis apresenta outliers extremos pelos métodos IQR ou Z-Score. Não será necessário aplicar técnicas drásticas de corte (*clipping*) antes da modelagem.

### 4.2 Normalidade, Assimetria e Curtose

| Feature          | Skewness | Classificação          | Kurtosis | Classificação              | Shapiro-Wilk p | Normal? |
|------------------|----------|------------------------|----------|----------------------------|----------------|---------|
| Tenure Months    | 0,240    | Simétrica              | -1,387   | Platicúrtica (caudas leves)| 0,0000         | **Não** |
| Monthly Charges  | -0,221   | Simétrica              | -1,257   | Platicúrtica (caudas leves)| 0,0000         | **Não** |
| Total Charges    | **0,962**| **Assimétrica à direita** | -0,232 | Mesocúrtica                | 0,0000         | **Não** |

Nenhuma variável segue distribuição normal (confirmado pelo teste de Shapiro-Wilk em todas). `Total Charges` merece atenção especial: sua assimetria à direita (skewness ≈ 0,96) reflete que a maioria dos clientes tem cobranças totais baixas (pouco tempo de casa ou planos baratos), enquanto uma minoria acumula valores altos ao longo dos anos.

**Ação obrigatória no pipeline:** aplicar transformação `log1p` ou `PowerTransformer` (Yeo-Johnson), seguida de `StandardScaler`, antes de alimentar a MLP.

---

## 5. Análise Univariada — Variáveis Categóricas

O dataset processou 16 variáveis categóricas sem nulos e sem categorias raras (< 1%), indicando uma base bem estruturada.

### 5.1 Variáveis Binárias

| Feature           | n     | Dominante         | %     | Minoria  | %     | Alta Concentração | Ação Sugerida          |
|-------------------|-------|-------------------|-------|----------|-------|-------------------|------------------------|
| Gender            | 7.043 | Male              | 50,5% | Female   | 49,5% | Não               | Avaliar descarte (p=0,49) |
| Senior Citizen    | 7.043 | No                | 83,8% | Yes      | 16,2% | **Sim**           | Manter (Cramer's V significativo) |
| Partner           | 7.043 | No                | 51,7% | Yes      | 48,3% | Não               | Manter                 |
| Dependents        | 7.043 | No                | 76,9% | Yes      | 23,1% | Não               | Manter                 |
| Phone Service     | 7.043 | Yes               | 90,3% | No       | 9,7%  | **Sim**           | Avaliar descarte (p=0,34) |
| Paperless Billing | 7.043 | Yes               | 59,2% | No       | 40,8% | Não               | Manter                 |

### 5.2 Variáveis Multiclasse

| Feature          | n     | Dominante           | %     | Minoria                | %     | n Categorias |
|------------------|-------|---------------------|-------|------------------------|-------|--------------|
| Multiple Lines   | 7.043 | No                  | 48,1% | No phone service       | 9,7%  | 3            |
| Internet Service | 7.043 | Fiber optic         | 44,0% | No                     | 21,7% | 3            |
| Online Security  | 7.043 | No                  | 49,7% | No internet service    | 21,7% | 3            |
| Online Backup    | 7.043 | No                  | 43,8% | No internet service    | 21,7% | 3            |
| Device Protection| 7.043 | No                  | 43,9% | No internet service    | 21,7% | 3            |
| Tech Support     | 7.043 | No                  | 49,3% | No internet service    | 21,7% | 3            |
| Streaming TV     | 7.043 | No                  | 39,9% | No internet service    | 21,7% | 3            |
| Streaming Movies | 7.043 | No                  | 39,5% | No internet service    | 21,7% | 3            |
| Contract         | 7.043 | Month-to-month      | 55,0% | One year               | 20,9% | 3            |
| Payment Method   | 7.043 | Electronic check    | 33,6% | Credit card (automatic)| 21,6% | 4            |

> **Observação:** a categoria `"No internet service"` presente em 7 features é derivada diretamente de `Internet Service = No`. Isso gera **redundância estrutural** que será endereçada na engenharia de features.

---

## 6. Análise Bivariada — Numéricas vs. Target

### 6.1 Estatísticas por Grupo (Churn vs. Non-Churn)

| Feature          | n (0) | n (1)  | Média (0) | Média (1) | Mediana (0) | Mediana (1) | Cohen's d | Correlação | p-value   |
|------------------|-------|--------|-----------|-----------|-------------|-------------|-----------|------------|-----------|
| Tenure Months    | 5.174 | 1.869  | 37,57     | **17,98** | 38,00       | **10,00**   | **-0,893**| -0,352     | < 0,0001  |
| Monthly Charges  | 5.174 | 1.869  | 61,27     | **74,44** | 64,43       | **79,65**   | **0,470** | 0,193      | < 0,0001  |
| Total Charges    | 5.163 | 1.869  | 2.555,34  | 1.531,80  | 1.683,60    | 703,55      | **-0,483**| -0,200     | < 0,0001  |

### 6.2 Análise Individual

**Tenure Months — Variável de Ouro (Cohen's d = -0,89 | Efeito Grande)**

O separador mais poderoso do dataset. Clientes que cancelam têm mediana de apenas **10 meses** de contrato, contra **38 meses** dos que permanecem. Isso revela um padrão crítico de negócio: existe uma janela de risco nos **primeiros 12 meses** onde a taxa de saída é significativamente maior. A empresa perde clientes antes de fidelizá-los.

**Monthly Charges — Sensibilidade ao Preço (Cohen's d = 0,47 | Efeito Médio)**

Clientes que cancelam pagam em média **R$ 74,44/mês** contra **R$ 61,27** dos que ficam. O churn está concentrado nos clientes de ticket mais alto, sugerindo que a percepção de valor em planos premium está comprometida — ou que o preço não é justificado pela qualidade percebida do serviço.

**Total Charges — Sinal Derivado (Cohen's d = -0,48 | Efeito Médio)**

O valor total menor no grupo Churn é uma consequência direta do menor tempo de casa (Tenure), não um preditor independente. Adicionalmente, o grupo Churn apresenta **5,83% de outliers** por IQR nesta variável, reforçando a necessidade de transformação robusta. Por ser matematicamente redundante com `Tenure × Monthly Charges`, esta variável é **candidata ao descarte** (confirmado pela análise de VIF na Seção 8).

---

## 7. Análise Bivariada — Categóricas vs. Target

### 7.1 Resumo de Impacto e Significância

| Feature           | Chi² p-value | Cramer's V | Impacto    | Recomendação                |
|-------------------|--------------|------------|------------|-----------------------------|
| Contract          | < 0,0001     | **0,4101** | **Forte**  | Manter — principal preditor |
| Online Security   | < 0,0001     | **0,3474** | **Forte**  | Manter                      |
| Tech Support      | < 0,0001     | **0,3429** | **Forte**  | Manter                      |
| Internet Service  | < 0,0001     | **0,3225** | **Forte**  | Manter                      |
| Payment Method    | < 0,0001     | **0,3034** | Moderado   | Manter                      |
| Online Backup     | < 0,0001     | 0,2923     | Moderado   | Manter                      |
| Device Protection | < 0,0001     | 0,2816     | Moderado   | Manter                      |
| Streaming Movies  | < 0,0001     | 0,2310     | Moderado   | Manter                      |
| Streaming TV      | < 0,0001     | 0,2305     | Moderado   | Manter                      |
| Paperless Billing | < 0,0001     | 0,1915     | Fraco/Mod  | Manter                      |
| Dependents        | < 0,0001     | 0,2482     | Moderado   | Manter                      |
| Senior Citizen    | < 0,0001     | 0,1505     | Fraco/Mod  | Manter                      |
| Partner           | < 0,0001     | 0,1501     | Fraco/Mod  | Manter                      |
| Multiple Lines    | 0,003        | 0,0401     | Fraco      | Avaliar                     |
| Phone Service     | 0,339        | 0,0114     | **Nulo**   | **Descartar**               |
| Gender            | 0,487        | 0,0083     | **Nulo**   | **Descartar**               |

### 7.2 Detalhamento por Feature — Taxas de Churn por Categoria

**Contract (Cramer's V = 0,41 — Maior preditor categórico)**

| Categoria       | n     | Churn (n) | Taxa Churn |
|-----------------|-------|-----------|------------|
| Month-to-month  | 3.875 | 1.655     | **42,71%** |
| One year        | 1.473 | 166       | 11,27%     |
| Two year        | 1.695 | 48        | **2,83%**  |

Contratos mensais têm taxa 15x maior que contratos de dois anos. A ausência de barreira de saída (multa rescisória) é o principal mecanismo. Clientes com compromisso de longo prazo desenvolvem inércia de permanência.

**Internet Service (Cramer's V = 0,32)**

| Categoria   | n     | Churn (n) | Taxa Churn |
|-------------|-------|-----------|------------|
| Fiber optic | 3.096 | 1.297     | **41,89%** |
| DSL         | 2.421 | 459       | 18,96%     |
| No          | 1.526 | 113       | 7,40%      |

O serviço de fibra óptica concentra o maior risco. A hipótese central é que o preço elevado e/ou problemas técnicos de qualidade (latência, instabilidade) estão gerando insatisfação neste segmento premium.

**Online Security (Cramer's V = 0,35) e Tech Support (Cramer's V = 0,34)**

| Categoria           | Churn Online Security | Churn Tech Support |
|---------------------|-----------------------|--------------------|
| No                  | **41,77%**            | **41,64%**         |
| No internet service | 7,40%                 | 7,40%              |
| Yes                 | 14,61%                | 15,17%             |

Clientes sem serviços de proteção e suporte apresentam churn ~3x maior que os que possuem. Esses serviços funcionam como **âncoras de retenção**: criam dependência técnica e percepção de valor adicional.

**Payment Method (Cramer's V = 0,30)**

| Método de Pagamento       | n     | Churn (n) | Taxa Churn |
|---------------------------|-------|-----------|------------|
| Electronic check          | 2.365 | 1.071     | **45,29%** |
| Mailed check              | 1.612 | 308       | 19,11%     |
| Bank transfer (automatic) | 1.544 | 258       | 16,71%     |
| Credit card (automatic)   | 1.522 | 232       | 15,24%     |

Pagamentos via cheque eletrônico concentram o maior churn. Uma hipótese é que esse método está associado a clientes com menor comprometimento contratual ou dificuldades financeiras, facilitando o cancelamento impulsivo.

**Paperless Billing (Cramer's V = 0,19)**

| Categoria | n     | Churn (n) | Taxa Churn |
|-----------|-------|-----------|------------|
| No        | 2.872 | 469       | 16,33%     |
| Yes       | 4.171 | 1.400     | **33,57%** |

Fatura digital está associada a maior churn, possivelmente por correlação com clientes tecnicamente menos engajados com a empresa ou com perfis mais jovens e voláteis.

**Senior Citizen (Cramer's V = 0,15)**

| Categoria | n     | Churn (n) | Taxa Churn |
|-----------|-------|-----------|------------|
| No        | 5.901 | 1.393     | 23,61%     |
| Yes       | 1.142 | 476       | **41,68%** |

Idosos têm taxa de churn quase o dobro da dos não-idosos, apesar de representarem apenas 16,2% da base. Merece atenção especial de monitoramento, pois o modelo pode subestimar esse segmento dado o desbalanceamento.

**Gender e Phone Service — Candidatos ao Descarte**

Gender (p = 0,487, Cramer's V = 0,008) e Phone Service (p = 0,339, Cramer's V = 0,011) não apresentam diferença estatisticamente significativa na propensão ao churn. Mantê-los no modelo adiciona complexidade (colunas extras no One-Hot Encoding) sem acrescentar poder preditivo — gerando potencial ruído para a MLP.

---

## 8. Análise Multivariada e Multicolinearidade

### 8.1 Correlação entre Variáveis Numéricas e com o Target

| Par de Features                      | Correlação |
|--------------------------------------|------------|
| Tenure Months ↔ Target               | -0,3522    |
| Total Charges ↔ Target               | -0,1995    |
| Monthly Charges ↔ Target             | +0,1934    |
| **Tenure Months ↔ Total Charges**    | **+0,8259** ⚠️ |
| Monthly Charges ↔ Total Charges      | — (implícita) |

### 8.2 Variance Inflation Factor (VIF)

| Feature          | VIF    | Status                      |
|------------------|--------|-----------------------------|
| Total Charges    | 8,0792 | ⚠️ Alto — próximo do limite crítico (10) |
| Tenure Months    | 6,3324 | ⚠️ Atenção — inflacionado pela correlação |
| Monthly Charges  | 3,3611 | ✅ Saudável — traz informação única |

### 8.3 Diagnóstico

A correlação de **0,826** entre `Tenure Months` e `Total Charges` revela uma relação estrutural óbvia: o total pago é produto do tempo de contrato pelo valor mensal. Fornecer ambas as variáveis ao modelo é fornecer **a mesma informação duas vezes**, em violação ao princípio de parcimônia.

As consequências práticas de ignorar essa multicolinearidade na MLP são:
- Gradientes instáveis durante o treinamento, pois duas features altamente correlatas competem por atualização de pesos similares.
- Coeficientes inflados e interpretação comprometida.
- Generalização degradada: pequenas variações nos dados de entrada geram variações ampliadas na predição.

**Decisão técnica:** `Total Charges` será **removida do feature set final**. Após sua remoção, o VIF de `Tenure Months` cai para níveis aceitáveis, isolando o sinal real de tempo de permanência sem redundância.

---

## 9. Decisões Técnicas para o Pipeline

Com base em todas as evidências levantadas, as seguintes decisões foram tomadas para a construção do pipeline Scikit-Learn + PyTorch:

### 9.1 Feature Selection

| Decisão            | Features                          | Justificativa                                      |
|--------------------|-----------------------------------|----------------------------------------------------|
| **Remover**        | Total Charges                     | VIF 8,08; correlação 0,83 com Tenure (redundância) |
| **Remover**        | Gender                            | Cramer's V = 0,008; p-value = 0,487 (sem sinal)   |
| **Remover**        | Phone Service                     | Cramer's V = 0,011; p-value = 0,339 (sem sinal)   |
| **Manter**         | Tenure Months, Monthly Charges    | Maior Cohen's d; sem multicolinearidade residual   |
| **Manter**         | Contract, Internet Service, Online Security, Tech Support, Payment Method | Cramer's V ≥ 0,30 |
| **Manter**         | Demais categóricas                | Cramer's V significativo após Chi²                |

### 9.2 Engenharia de Features Sugerida

```python
# Feature derivada 1: indicador de serviço de fibra óptica
df['is_fiber_optic'] = (df['Internet Service'] == 'Fiber optic').astype(int)

# Feature derivada 2: flag de segurança consolidada
df['has_security_support'] = (
    (df['Online Security'] == 'Yes') | (df['Tech Support'] == 'Yes')
).astype(int)
```

**Justificativa:** `Internet Service = Fiber optic` apresenta churn de 41,89% (Cramer's V = 0,32) e `Online Security` / `Tech Support` apresentam Cramer's V ~0,34, consolidar em uma feature binária simplifica o espaço de entrada sem perda de informação.

### 9.3 Transformações Numéricas

```python
# Pipeline de pré-processamento numérico
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),  # Total Charges nulos
    ('power_transform', PowerTransformer(method='yeo-johnson')),       # Normaliza assimetria
    ('scaler', StandardScaler()),                                       # Escala para MLP
])
```

### 9.4 Encoding de Categóricas

```python
# Variáveis ordinais com hierarquia natural
ordinal_features = ['Contract']  # Month-to-month < One year < Two year
ordinal_encoder = OrdinalEncoder(categories=[['Month-to-month', 'One year', 'Two year']])

# Variáveis nominais sem hierarquia
nominal_features = ['Internet Service', 'Payment Method', 'Multiple Lines',
                    'Online Security', 'Online Backup', 'Device Protection',
                    'Tech Support', 'Streaming TV', 'Streaming Movies']
onehot_encoder = OneHotEncoder(drop='if_binary', sparse_output=False, handle_unknown='ignore')

# Variáveis binárias (mapeamento simples)
binary_features = ['Senior Citizen', 'Partner', 'Dependents', 'Paperless Billing']
binary_encoder = OrdinalEncoder()  # Yes=1, No=0
```

### 9.5 Estratégia de Validação

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

Validação cruzada estratificada garante que cada fold preserve o ratio original de ~26,5% de churn, evitando que folds com proporções desiguais contaminem a estimativa de performance.

---

## 10. Síntese Estratégica

### Features por Poder Preditivo (Ranking Final)

| Rank | Feature          | Tipo      | Métrica de Relevância  | Valor      |
|------|------------------|-----------|------------------------|------------|
| 1    | Contract         | Categórica| Cramer's V             | 0,41       |
| 2    | Tenure Months    | Numérica  | Cohen's d / Corr.      | -0,89 / -0,35 |
| 3    | Online Security  | Categórica| Cramer's V             | 0,35       |
| 4    | Tech Support     | Categórica| Cramer's V             | 0,34       |
| 5    | Internet Service | Categórica| Cramer's V             | 0,32       |
| 6    | Payment Method   | Categórica| Cramer's V             | 0,30       |
| 7    | Monthly Charges  | Numérica  | Cohen's d / Corr.      | 0,47 / 0,19 |
| 8    | Online Backup    | Categórica| Cramer's V             | 0,29       |
| 9    | Device Protection| Categórica| Cramer's V             | 0,28       |
| 10   | Dependents       | Categórica| Cramer's V             | 0,25       |
| —    | Total Charges    | Numérica  | VIF = 8,08 / Multicolinear | ❌ Remover |
| —    | Gender           | Categórica| Cramer's V = 0,008     | ❌ Remover |
| —    | Phone Service    | Categórica| Cramer's V = 0,011     | ❌ Remover |

### Perfil do Cliente com Alto Risco de Churn

Com base nos dados, o cliente de maior risco apresenta o seguinte perfil:

- Contrato **Month-to-month** (churn de 42,71%).
- Menos de **12 meses** de tempo de casa (mediana do grupo Churn = 10 meses).
- Serviço de **fibra óptica** (churn de 41,89%).
- **Sem** Online Security e Tech Support (churn > 41%).
- Pagamento via **Electronic check** (churn de 45,29%).
- **Monthly Charges** acima da média (R$ 74,44 vs. R$ 61,27).

### Alerta para o Trade-off de Custo (Negócio)

Dado que o Recall da classe positiva é a métrica prioritária, o modelo deve ser calibrado para tolerar mais Falsos Positivos (custo controlável de ações de retenção desnecessárias) em detrimento de reduzir Falsos Negativos (perda de receita irrecuperável). O limiar de decisão (threshold) da MLP deverá ser ajustado via análise da curva PR ao final da Etapa 2.

---

*Este relatório é um artefato da Etapa 1 do Tech Challenge FIAP MLOps e serve como base para o preenchimento do ML Canvas, a definição de métricas de negócio e a construção do pipeline de pré-processamento.*