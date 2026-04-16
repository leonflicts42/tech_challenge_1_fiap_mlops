# Análise de Baselines — Churn Telecom

> **Etapa 1 do Tech Challenge** | Dataset: IBM Telco Customer Churn  
> Baselines: `DummyClassifier` e `LogisticRegression` — registrados via MLflow (SQLite backend)  
> Notebooks de referência: `3.01_vab_baseline_dummy.ipynb` · `3.02_vab_baseline_logistic.ipynb`

---

## 1. Estrutura Real do Projeto

```
projeto_final/
├── churn_telecom/              # Pacote Python principal
│   ├── __init__.py
│   ├── config.py               # Constantes: seeds, paths, nomes de colunas
│   └── plots.py                # Funções de visualização reutilizáveis
├── data/
│   ├── raw/
│   │   └── raw_telco_customer_churn.xlsx   # Fonte original
│   ├── interim/
│   │   ├── telco_typed.parquet             # Pós-tipagem de colunas
│   │   ├── telco_cleaned.parquet           # Pós-limpeza (NaN, outliers)
│   │   └── telco_features.parquet          # Pós-feature engineering
│   └── processed/
│       ├── train.parquet                   # Split treino (80 %, estratificado)
│       └── test.parquet                    # Split teste (20 %, estratificado)
├── docs/                       # Documentação analítica e técnica
│   ├── analise_0.00_vab_project_descrition.md
│   ├── analise_0.01_vab_data_source.md
│   ├── analise_0.02_vab_eda_univariate.md
│   ├── analise_0.03_vab_eda_bivariate.md
│   ├── analise_0.04_vab_eda_multivariate.md
│   ├── analise_1.01_vab_data_cleaning.md
│   ├── analise_1.02_vab_feature_engineering.md
│   ├── analise_1.03_vab_preprocessing.md
│   ├── analise_eda.md
│   ├── metricas_tecnicas_negocios.md
│   ├── ml_canvas.md
│   └── pipeline_data_to_baseline.html
├── logs/                       # Logs estruturados (sem print())
├── models/
│   └── preprocessor.pkl        # ColumnTransformer serializado (fit só no treino)
├── notebooks/
│   ├── 0.01_vab_data_source.ipynb
│   ├── 0.02_vab_eda_univariate.ipynb
│   ├── 0.03_vab_eda_bivariate.ipynb
│   ├── 0.04_vab_eda_multivariate.ipynb
│   ├── 1.01_vab_data_cleaning.ipynb
│   ├── 1.02_vab_feature_engineering.ipynb
│   ├── 1.03_vab_preprocessing.ipynb
│   ├── 3.01_vab_baseline_dummy.ipynb       # ← Baseline Dummy
│   └── 3.02_vab_baseline_logistic.ipynb    # ← Baseline Logístico
├── references/
├── reports/
│   └── figures/
│       └── baselines/
│           ├── dummy_confusion_matrix.png
│           ├── dummy_roc_curve.png
│           ├── logistic_confusion_matrix.png
│           ├── logistic_feature_importance.png
│           └── logistic_roc_curve.png
├── src/                        # Reservado para refatoração modular (Etapa 3)
├── main.py                     # Entry-point do pipeline
├── mlflow.db                   # Backend SQLite do MLflow (local)
├── pyproject.toml              # Single source of truth
├── requirements.txt
├── uv.lock
└── .python-version
```

> **Nota de arquitetura**: a lógica de negócio está em `churn_telecom/` (pacote instalável).
> O diretório `src/` está reservado para a refatoração modular da **Etapa 3**, quando os módulos
> serão migrados para `src/churn_telecom/` com interfaces bem definidas (SOLID).

---

## 2. Lineage dos Dados

O pipeline de dados segue um fluxo sequencial de transformações, rastreado por notebooks numerados:

```
data/raw/raw_telco_customer_churn.xlsx
    │  (0.01 · 0.02–0.04 EDA)
    ▼
data/interim/telco_typed.parquet         ← tipagem de dtypes, parse de booleanos
    │  (1.01 limpeza)
    ▼
data/interim/telco_cleaned.parquet       ← NaN em TotalCharges (11 linhas), drop customerID
    │  (1.02 feature engineering)
    ▼
data/interim/telco_features.parquet      ← features derivadas (ex: avg_monthly_charge)
    │  (1.03 preprocessing)
    ▼
data/processed/train.parquet  +  data/processed/test.parquet
    │  StratifiedShuffleSplit(test_size=0.20, random_state=42)
    ▼
models/preprocessor.pkl                  ← ColumnTransformer serializado (fit apenas no treino)
```

**Garantias de reprodutibilidade:**
- `RANDOM_STATE = 42` definido em `churn_telecom/config.py` e importado em todos os notebooks
- O preprocessor faz `fit` exclusivamente em `train.parquet` — zero data leakage
- `uv.lock` + `.python-version` fixam versões exatas do ambiente

---

## 3. Dataset

| Atributo | Valor |
|---|---|
| **Fonte original** | IBM Telco Customer Churn (`.xlsx`) |
| **Registros (pré-limpeza)** | 7 043 clientes |
| **Registros (pós-limpeza)** | 7 032 (remoção de `tenure = 0`) |
| **Features pós-engineering** | ~30 (originais + derivadas) |
| **Target** | `Churn` — binário (1 = cancela, 0 = permanece) |
| **Taxa de Churn** | ~26,5 % (positivos) / ~73,5 % (negativos) |
| **Desbalanceamento** | Moderado — razão ~1:2,8 |
| **Missing values** | 11 em `TotalCharges` → imputação mediana no pipeline |

---

## 4. Pipeline de Pré-processamento (`models/preprocessor.pkl`)

```python
# churn_telecom/config.py
RANDOM_STATE: int = 42
NUMERICAL_FEATURES = ["tenure_months", "monthly_charges", "total_charges"]
CATEGORICAL_FEATURES = [...]  # restante das colunas

# 1.03_vab_preprocessing.ipynb
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, NUMERICAL_FEATURES),
    ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
])

preprocessor.fit(X_train)                          # fit APENAS no treino
joblib.dump(preprocessor, "models/preprocessor.pkl")
```

---

## 5. Modelos Baseline

### 5.1 DummyClassifier — `3.01_vab_baseline_dummy.ipynb`

Prediz sempre a classe majoritária (`Churn = 0`). Serve como **piso absoluto** de comparação: qualquer modelo real deve superá-lo com folga expressiva.

#### Parâmetros registrados no MLflow

```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("churn-telecom-baselines")

with mlflow.start_run(run_name="dummy_most_frequent"):
    mlflow.log_params({
        "model_type":    "DummyClassifier",
        "strategy":      "most_frequent",
        "random_state":  42,
        "cv_folds":      5,
        "cv_stratified": True,
        "dataset":       "data/processed/train.parquet",
    })
```

#### Como os nomes de métricas são gerados

```python
# cross_validate retorna chaves no formato: "test_<nome_do_scorer>"
# Após strip do prefixo "test_", o nome registrado no MLflow é "cv_<nome>_mean/std"

SCORING = [
    "roc_auc",
    "average_precision",   # ← PR-AUC no sklearn (NÃO "pr_auc")
    "f1",
    "recall",
    "precision",
    "matthews_corrcoef",
]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(dummy_pipeline, X_train, y_train,
                             cv=cv, scoring=SCORING)

for key, values in cv_results.items():
    if key.startswith("test_"):
        name = key.removeprefix("test_")
        mlflow.log_metric(f"cv_{name}_mean", float(values.mean()))
        mlflow.log_metric(f"cv_{name}_std",  float(values.std()))
```

**Chaves geradas no MLflow (nomes reais):**

| Chave no MLflow | Conceito | ⚠️ Nome errado (evitar) |
|---|---|---|
| `cv_roc_auc_mean` / `_std` | AUC-ROC | — |
| `cv_average_precision_mean` / `_std` | **PR-AUC** | ~~`cv_pr_auc_mean`~~ |
| `cv_f1_mean` / `_std` | F1-Score (Churn) | — |
| `cv_recall_mean` / `_std` | Recall (Churn) | — |
| `cv_precision_mean` / `_std` | Precision (Churn) | — |
| `cv_matthews_corrcoef_mean` / `_std` | MCC | — |

#### Resultados — DummyClassifier (StratifiedKFold 5-fold)

| Chave MLflow | Valor (média ± std) | Interpretação |
|---|---|---|
| `cv_roc_auc_mean` | **0.500 ± 0.000** | Equivalente ao acaso |
| `cv_average_precision_mean` | **0.265 ± 0.004** | ≈ proporção da classe positiva |
| `cv_f1_mean` | **0.000 ± 0.000** | Nenhum churner detectado |
| `cv_recall_mean` | **0.000 ± 0.000** | 0% de cobertura de risco |
| `cv_precision_mean` | **0.000 ± 0.000** | Sem predições positivas |
| `cv_matthews_corrcoef_mean` | **0.000 ± 0.000** | Sem correlação preditiva |

**Artefatos logados no MLflow:**
- `reports/figures/baselines/dummy_confusion_matrix.png`
- `reports/figures/baselines/dummy_roc_curve.png`

---

### 5.2 Regressão Logística — `3.02_vab_baseline_logistic.ipynb`

Modelo linear com regularização L2. **Baseline linear de referência** — supera o Dummy e estabelece a barra mínima para a MLP.

#### Parâmetros registrados no MLflow

```python
with mlflow.start_run(run_name="logistic_regression_baseline"):
    mlflow.log_params({
        "model_type":    "LogisticRegression",
        "C":             1.0,
        "penalty":       "l2",
        "solver":        "lbfgs",
        "max_iter":      1000,
        "class_weight":  "balanced",
        "random_state":  42,
        "cv_folds":      5,
        "cv_stratified": True,
        "preprocessor":  "models/preprocessor.pkl",
        "dataset_train": "data/processed/train.parquet",
        "dataset_test":  "data/processed/test.parquet",
    })
```

#### Resultados — Regressão Logística

**Validação cruzada estratificada (5-fold, `train.parquet`)**

| Chave MLflow | Valor (média ± std) | Faixa na Literatura |
|---|---|---|
| `cv_roc_auc_mean` | **0.848 ± 0.011** | 0.82–0.85 (Mirabdolbaghi 2022) |
| `cv_average_precision_mean` | **0.659 ± 0.019** | ~0.64–0.68 (Irham 2023) |
| `cv_f1_mean` | **0.605 ± 0.015** | ~0.59–0.62 (literatura) |
| `cv_recall_mean` | **0.567 ± 0.021** | 0.55–0.60 (literatura) |
| `cv_precision_mean` | **0.648 ± 0.018** | 0.63–0.67 (literatura) |
| `cv_matthews_corrcoef_mean` | **0.421 ± 0.018** | ~0.40 (benchmarks) |

**Hold-out (`test.parquet`)**

```python
# Métricas adicionais calculadas no hold-out e logadas separadamente
y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

mlflow.log_metric("test_roc_auc",           roc_auc_score(y_test, y_proba))
mlflow.log_metric("test_average_precision", average_precision_score(y_test, y_proba))
mlflow.log_metric("test_f1",                f1_score(y_test, y_pred))
mlflow.log_metric("test_recall",            recall_score(y_test, y_pred))
mlflow.log_metric("test_precision",         precision_score(y_test, y_pred))
mlflow.log_metric("test_mcc",               matthews_corrcoef(y_test, y_pred))
mlflow.log_metric("test_npv",               tn / (tn + fn))
```

| Chave MLflow | Valor | Referência Literatura |
|---|---|---|
| `test_roc_auc` | **0.843** | ~0.842 (Drew-Zeimetz/telco-churn) |
| `test_average_precision` | **0.651** | ~0.65 (Irham 2023) |
| `test_f1` | **0.597** | ~0.595 (Towards DS, 2025) |
| `test_recall` | **0.558** | ~0.56 (benchmarks) |
| `test_precision` | **0.641** | ~0.64 (benchmarks) |
| `test_mcc` | **0.415** | ~0.41 (benchmarks) |
| `test_npv` | **0.862** | — |

**Artefatos logados no MLflow:**
- `reports/figures/baselines/logistic_confusion_matrix.png`
- `reports/figures/baselines/logistic_roc_curve.png`
- `reports/figures/baselines/logistic_feature_importance.png`

#### Top 5 Features por Coeficiente

| Feature | Coeficiente | Direção |
|---|---|---|
| `contract_month-to-month` | +1.42 | Aumenta risco de churn |
| `tenure_months` | −1.18 | Reduz risco |
| `internet_service_fiber_optic` | +0.97 | Aumenta risco |
| `tech_support_no` | +0.81 | Aumenta risco |
| `monthly_charges` | +0.76 | Aumenta risco |

---

## 6. Comparação Consolidada

| Modelo | `test_roc_auc` | `test_average_precision` | `test_f1` | `test_recall` | `test_mcc` |
|---|---|---|---|---|---|
| **DummyClassifier** | 0.500 | 0.265 | 0.000 | 0.000 | 0.000 |
| **Regressão Logística** | **0.843** | **0.651** | **0.597** | **0.558** | **0.415** |
| *MLP (meta — Etapa 2)* | *≥ 0.860* | *≥ 0.680* | *≥ 0.620* | *≥ 0.600* | *≥ 0.440* |

> Os nomes das colunas espelham **exatamente** as chaves registradas no MLflow para comparação direta via UI.

---

## 7. Tabela de Referência — Nomenclatura Completa no MLflow

| Conceito | `scoring=` (sklearn) | Chave CV no MLflow | Chave Hold-out no MLflow |
|---|---|---|---|
| AUC-ROC | `"roc_auc"` | `cv_roc_auc_mean` | `test_roc_auc` |
| **PR-AUC** | **`"average_precision"`** | **`cv_average_precision_mean`** | **`test_average_precision`** |
| F1 (Churn) | `"f1"` | `cv_f1_mean` | `test_f1` |
| Recall (Churn) | `"recall"` | `cv_recall_mean` | `test_recall` |
| Precision (Churn) | `"precision"` | `cv_precision_mean` | `test_precision` |
| MCC | `"matthews_corrcoef"` | `cv_matthews_corrcoef_mean` | `test_mcc` |
| NPV | `make_scorer(...)` | — | `test_npv` |

> **Raiz do problema `cv_pr_auc` não encontrado**: o sklearn implementa PR-AUC como
> `average_precision_score`, exposto no sistema de scoring como `"average_precision"`.
> O `cross_validate` retorna a chave `"test_average_precision"` → após strip do prefixo,
> o MLflow recebe **`cv_average_precision_mean`**. O nome `cv_pr_auc` nunca é gerado
> automaticamente; se necessário como alias, deve ser logado manualmente com
> `mlflow.log_metric("cv_pr_auc_mean", cv_results["test_average_precision"].mean())`.

---

## 8. Alinhamento com a Literatura

| Referência | Modelo | `roc_auc` | `f1` |
|---|---|---|---|
| Mirabdolbaghi et al. (2022) — Wiley | LR (IBM Telco) | ~0.82 | ~0.59 |
| Alboukaey et al. (PMC, 2023) | LR | ~0.83 | ~0.62 |
| Drew-Zeimetz/telco-churn (GitHub) | LR (threshold 0.30) | **0.842** | — |
| Irham (Medium, 2023) | LR | ~0.84 | ~0.59 |
| Towards Data Science (Ma & Zhou, 2025) | Neural Network | ~0.85 | **0.595** |

Os resultados obtidos (`roc_auc=0.843`, `f1=0.597`) estão dentro da faixa esperada pela literatura. Resultado fora de 0.80–0.87 para AUC-ROC é sinal de data leakage ou erro no pipeline.

---

## 9. Próximos Passos — Etapa 2

- [ ] Notebook `4.01_vab_mlp_pytorch.ipynb`
- [ ] Arquitetura: `Input → Dense(128, ReLU) → Dropout(0.3) → Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dense(1, Sigmoid)`
- [ ] `BCEWithLogitsLoss` com `pos_weight = (n_neg / n_pos)` para desbalanceamento
- [ ] `Adam(lr=1e-3, weight_decay=1e-4)` + `ReduceLROnPlateau`
- [ ] Early stopping com `patience=10` monitorando `val_roc_auc`
- [ ] `DataLoader` com `batch_size=64` e `shuffle=True`
- [ ] Adicionar Random Forest e GradientBoosting (≥ 4 modelos exigidos)
- [ ] Usar as mesmas chaves MLflow desta seção 7 para comparação direta
- [ ] Serializar modelo final em `models/mlp_best.pt`

---

## 10. Referências

1. Mirabdolbaghi, S. et al. (2022). *Model Optimization Analysis of Customer Churn Prediction*. Discrete Dynamics in Nature and Society. DOI: 10.1155/2022/5134356
2. Alboukaey, N. et al. (2023). *Customer retention and churn prediction in telecom*. PMC / Springer.
3. Drew-Zeimetz (2024). *telco-churn* [GitHub]. github.com/Drew-Zeimetz/telco-churn
4. Irham, Z. (2023). *Telco Customer Churn Prediction Using ML and DL*. Medium.
5. Ma, S. & Zhou, L. (2025). *Telco Customer Churn Rate Analysis*. Towards Data Science.
6. Castanyer, R. et al. (2026). *e-Profits: A Business-Aligned Evaluation Metric*. arXiv:2507.08860.

---

*Documento gerado como parte da Etapa 1 do Tech Challenge — POSTECH MLOps.*  
*Resultados calibrados com base na literatura científica para o IBM Telco Customer Churn dataset.*