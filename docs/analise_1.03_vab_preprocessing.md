# Relatório — Preprocessing (1.03)

**Notebook:** `1.03_vab_preprocessing`
**Entrada:** `data/interim/telco_features.parquet` — shape (7021, 26)
**Saídas:**
- `data/processed/train.parquet` — shape (5395, 31)
- `data/processed/test.parquet` — shape (1405, 31)
- `models/preprocessor.pkl` — ColumnTransformer serializado
**Status:** concluído com sucesso

---

## Resumo executivo

| Métrica | Valor |
|---|---|
| Linhas de entrada | 7.021 |
| Features de entrada | 25 (X) + 1 target |
| Features após transformação | 30 |
| Train (antes SMOTE+ENN) | 5.616 (80%) |
| Train (após SMOTE+ENN) | 5.395 |
| Test (original, sem resample) | 1.405 (20%) |
| Churn rate treino após resample | 57.41% |
| Churn rate teste (real) | 26.48% |
| Preprocessor serializado | models/preprocessor.pkl |

---

## O que este notebook faz

O `1.03` é o único ponto do projeto onde encoding e transformações de escala são aplicados. Ele recebe o dataset com strings originais e features de engenharia criadas no `1.02`, e entrega dois artefatos prontos para a modelagem: os datasets transformados e o objeto `preprocessor.pkl` que será reutilizado identicamente pela API FastAPI na Etapa 3.

---

## 1. Validação de pré-condições

O notebook verificou três contratos obrigatórios da saída do `1.02`:

- Features de engenharia presentes: `num_services`, `charges_per_month`, `is_month_to_month`, `tenure_group`, `has_security_support`, `is_fiber_optic` — todas confirmadas.
- `total_charges_log` ausente — correto. A transformação `log1p` é responsabilidade do `ColumnTransformer` do `1.03`, não do `1.02`.
- `total_charges` original presente — necessária para a transformação `log1p` interna do pipeline.
- Strings originais preservadas (`online_security`, `tech_support`, `partner` ainda são `object/category`).

---

## 2. Separação X e y

O target `churn_value` foi separado do dataset. O `X` resultante ficou com 25 colunas (26 originais menos o target), com taxa de churn de 26.45% — confirmando que o desbalanceamento original foi preservado intacto antes do split.

---

## 3. Grupos de transformação

O `ColumnTransformer` foi construído com quatro grupos cobrindo todas as 25 colunas de `X`:

| Grupo | Transformer | Colunas | Saída |
|---|---|---|---|
| `num` (5) | `SimpleImputer → log1p → StandardScaler` | tenure_months, monthly_charges, total_charges, num_services, charges_per_month | 5 features escaladas |
| `bin` (12) | `OrdinalEncoder(["No","Yes"])` | partner, dependents, phone_service, multiple_lines, online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies, paperless_billing, senior_citizen | 12 features 0/1 |
| `ohe` (5) | `OneHotEncoder(drop="first")` | internet_service, contract, payment_method, gender, tenure_group | 8 features dummies |
| `pass` (3) | `passthrough` | is_month_to_month, has_security_support, is_fiber_optic | 3 features int |

Total: 5 + 12 + 8 + 3 = 28... mas o log reportou 30. A diferença se explica pela expansão do OHE: `internet_service` (3 categorias → 2 dummies), `contract` (3 → 2), `payment_method` (4 → 3), `gender` (2 → 1), `tenure_group` (3 → 2) — total 10 dummies, não 5. O `drop="first"` elimina uma categoria por variável.

**Cobertura validada:** todas as 25 colunas de `X` foram cobertas por algum grupo. Nenhuma coluna foi perdida silenciosamente.

---

## 4. Split estratificado

```
train=5616 (80.0%) | test=1405 (20.0%)
Churn rate | train=26.44% | test=26.48% | delta=0.035%
```

O `stratify=y` funcionou perfeitamente: o delta de apenas 0.035% entre as taxas de churn de treino e teste confirma que a proporção de classes foi preservada em ambos os conjuntos. Isso é essencial para que as métricas de avaliação no test set reflitam a distribuição real do problema.

---

## 5. Transformação

```
X_train=(5616, 30) | X_test=(1405, 30) | dtype=float64
```

O `fit_transform` foi aplicado **apenas no X_train** e o `transform` no `X_test` — garantindo que nenhuma informação do conjunto de teste influenciou os parâmetros do `StandardScaler` ou do `OrdinalEncoder`. Isso elimina o risco de data leakage na etapa de pré-processamento.

O dtype `float64` confirma que a conversão explícita do array funcionou corretamente — o `ColumnTransformer` com `passthrough` retorna `object` por padrão quando há mistura de tipos, mas a conversão forçou `float64` em todo o array antes da criação dos DataFrames.

**30 features geradas:**

```
Numéricas (5):
  tenure_months, monthly_charges, total_charges,
  num_services, charges_per_month

Binárias OrdinalEncoded (12):
  partner, dependents, phone_service, multiple_lines,
  online_security, online_backup, device_protection,
  tech_support, streaming_tv, streaming_movies,
  paperless_billing, senior_citizen

OHE nominais (10):
  internet_service_Fiber optic, internet_service_No,
  contract_One year, contract_Two year,
  payment_method_Credit card (automatic),
  payment_method_Electronic check,
  payment_method_Mailed check,
  gender_Male,
  tenure_group_medio, tenure_group_novo

Passthrough int (3):
  is_month_to_month, has_security_support, is_fiber_optic
```

**Nota sobre as categorias de referência do OHE** (drop="first"):
- `internet_service`: referência = DSL
- `contract`: referência = Month-to-month
- `payment_method`: referência = Bank transfer (automatic)
- `gender`: referência = Female
- `tenure_group`: referência = longo

---

## 6. SMOTE+ENN

```
antes: n=5616 | churn=26.44%
depois: n=5395 | churn=57.41%
delta: -221 amostras
```

O SMOTE+ENN combinou duas operações: oversampling da classe minoritária (Churn=1) via SMOTE e limpeza de fronteira via ENN (Edited Nearest Neighbors), que remove amostras ambíguas próximas à fronteira de decisão. O resultado foi uma **redução líquida de 221 amostras** — o ENN removeu mais amostras ruidosas do que o SMOTE adicionou, o que é o comportamento esperado e desejável.

A taxa de churn subiu de 26.44% para 57.41% no conjunto de treino — praticamente balanceado. O conjunto de teste permaneceu intacto com 26.48%, refletindo a distribuição real do problema.

**Ponto de atenção para a interpretação das métricas:** os modelos serão treinados com churn=57% mas avaliados com churn=26%. Isso é correto — o modelo aprende fronteiras mais claras com dados balanceados, mas o threshold de decisão deve ser calibrado no conjunto de teste com a distribuição real.

---

## 7. Persistência

| Artefato | Shape / Tamanho | Churn rate |
|---|---|---|
| `train.parquet` | (5395, 31) | 57.41% (resampled) |
| `test.parquet` | (1405, 31) | 26.48% (original) |
| `preprocessor.pkl` | — | — |

A validação de round-trip do `preprocessor.pkl` confirmou que o objeto recarregado produz resultados idênticos ao original (`np.allclose` = True).

---

## 8. Impacto na API FastAPI (Etapa 3)

O `preprocessor.pkl` é o artefato central que conecta o pipeline de treino à API de inferência. O fluxo na API será:

```python
# Carrega uma única vez no startup
preprocessor = joblib.load("models/preprocessor.pkl")

# Por requisição — replica exatamente o pipeline de treino
dados = to_snake_case(request.dict())      # snake_case nos nomes
dados = criar_features(dados)              # lógica do 1.02
X = pd.DataFrame([dados])
X_transformado = preprocessor.transform(X) # mesmo ColumnTransformer
predicao = modelo.predict(X_transformado)
```

O SMOTE+ENN **não é aplicado em inferência** — ele foi exclusivo do treino para balancear as classes. A API recebe dados reais com distribuição natural de churn (~26%).

---

## 9. Próximo passo

**Notebook:** `3.01_baseline_dummy` + `3.02_baseline_logistic` + `3.03_baseline_tree`
**Entrada:** `data/processed/train.parquet` — shape (5395, 31)
**Avaliação:** `data/processed/test.parquet` — shape (1405, 31)
**Registro:** MLflow experiment `churn-telecom`
**Métricas foco:** Recall + F1-Score da classe positiva (Churn=1)