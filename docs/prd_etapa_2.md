# PRD — Etapa 2: Modelagem com Redes Neurais
**Tech Challenge POSTECH MLOps · Fase 1**

> Foco: construção, treinamento e avaliação de MLP com PyTorch, comparação com
> baselines lineares e árvores, análise de custo de negócio e registro completo
> no MLflow.

---

## Contexto e pré-requisitos

### O que a Etapa 1 entregou

| Artefato | Localização | Uso na Etapa 2 |
|---|---|---|
| `train.parquet` | `data/processed/` | Entrada direta para `DataLoader` |
| `test.parquet` | `data/processed/` | Avaliação final de todos os modelos |
| `preprocessor.pkl` | `models/` | Referência de arquitetura de features |
| `mlflow.db` | raiz | Mesmo experimento — runs comparáveis |
| Runs: `dummy`, `baseline-logistic` | MLflow | Baselines de referência para comparação |
| `pos_weight` | `mlflow.db` params | `BCEWithLogitsLoss` na MLP |
| `N_FEATURES = 30` | `mlflow.db` params | `input_dim` da MLP |

### Restrições técnicas obrigatórias

- Seeds fixadas: `RANDOM_STATE = 42`, `torch.manual_seed(42)`, `np.random.seed(42)`
- Logging estruturado via `get_logger()` — zero `print()`
- Linting `ruff` sem erros antes do commit
- Todas as figuras salvas em `reports/figures/` **antes** de serem logadas no MLflow
- Mesmo experimento MLflow (`churn-telecom`) para todos os runs — comparáveis na UI

---

## Tarefas

---

### TASK-2.01 — MLP em PyTorch: arquitetura, ativação e loss function

**Arquivo:** `notebooks/4.01_vab_mlp_architecture.ipynb`
**Referência:** Fundamentos, Aula 04

#### Descrição

Definir e documentar a arquitetura da rede MLP em PyTorch como classe reutilizável
em `churn_telecom/models.py`. A arquitetura deve ser configurável por parâmetro
para facilitar experimentação e logging no MLflow.

#### Decisões técnicas a implementar

**Arquitetura:**
```
Input(30) → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
          → Linear(64)  → BatchNorm → ReLU → Dropout(0.3)
          → Linear(32)  → BatchNorm → ReLU → Dropout(0.2)
          → Linear(1)   → (sem ativação — BCEWithLogitsLoss)
```

**Justificativas a documentar:**
- `BatchNorm1d` após cada camada linear: estabiliza gradientes, acelera convergência
- `ReLU`: padrão para classificação tabular, evita vanishing gradient
- Saída sem ativação + `BCEWithLogitsLoss`: numericamente estável vs `Sigmoid + BCELoss`
- `pos_weight = n_neg / n_pos` (≈ 2.77 do test set): compensa desbalanceamento residual

**Classe a criar em `churn_telecom/models.py`:**
```python
class ChurnMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout_rates: list[float],
    ) -> None: ...
```

#### Critérios de aceite

- `churn_telecom/models.py` criado com `ChurnMLP` seguindo PEP 8 e `ruff` sem erros
- Notebook documenta arquitetura com `torchinfo.summary()` ou equivalente
- `forward()` retorna logits (não probabilidades) — compatível com `BCEWithLogitsLoss`
- Parâmetros de arquitetura logados no MLflow: `hidden_dims`, `dropout_rates`, `activation`

---

### TASK-2.02 — Loop de treinamento com early stopping e batching

**Arquivo:** `notebooks/4.01_vab_mlp_architecture.ipynb` (mesma célula principal)
**Referência:** Fundamentos, Aula 04

#### Descrição

Implementar loop de treinamento completo com early stopping baseado em
`val_roc_auc`, `DataLoader` com batching e separação de validação interna
(do conjunto de treino — sem tocar no test set).

#### Especificações técnicas

**DataLoader:**
```python
# Divisão interna: 80% treino / 20% validação — do train.parquet
# StratifiedShuffleSplit para manter proporção de churn (~57%)
DataLoader(train_dataset, batch_size=64, shuffle=True,
           generator=torch.Generator().manual_seed(42))
DataLoader(val_dataset,   batch_size=256, shuffle=False)
```

**Otimizador e scheduler:**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5
)
```

**Early stopping:**
```python
# Parâmetros a logar no MLflow
patience      = 15    # épocas sem melhora no val_roc_auc
min_delta     = 0.001 # melhora mínima considerada significativa
max_epochs    = 200
```

**Métricas logadas por época no MLflow:**
```python
mlflow.log_metric("train_loss",    train_loss,    step=epoch)
mlflow.log_metric("val_loss",      val_loss,      step=epoch)
mlflow.log_metric("val_roc_auc",   val_roc_auc,   step=epoch)
mlflow.log_metric("val_f1",        val_f1,        step=epoch)
mlflow.log_metric("learning_rate", current_lr,    step=epoch)
```

**Artefatos a salvar:**
- `reports/figures/mlp/training_curves.png` — loss e AUC por época (treino vs val)
- `models/mlp_best.pt` — checkpoint do melhor modelo (menor val_loss com melhor val_roc_auc)

#### Critérios de aceite

- Early stopping encerra o treinamento antes de `max_epochs` se `patience` esgotar
- Melhor checkpoint restaurado automaticamente ao final do treinamento
- `mlp_best.pt` salvo e logado como artefato no MLflow
- Curvas de aprendizado salvas em `reports/figures/mlp/`
- Log estruturado a cada época: `epoch | train_loss | val_loss | val_auc | lr`

---

### TASK-2.03 — Modelos de árvore: Random Forest e Gradient Boosting

**Arquivo:** `notebooks/3.03_vab_baseline_trees.ipynb`
**Referência:** Fundamentos, Aula 05

#### Descrição

Treinar Random Forest e Gradient Boosting como representantes da família de árvores,
necessários para a comparação com ≥ 4 modelos exigida pelo Tech Challenge.
Parâmetros padrão sem tuning — mesma filosofia dos baselines lineares da Etapa 1.

#### Especificações técnicas

```python
# Random Forest — parâmetros padrão + seed
RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced",
)

# Gradient Boosting — parâmetros padrão + seed
GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=RANDOM_STATE,
)
```

**Métricas a registrar no MLflow:**
- `test_auc`, `test_pr_auc`, `test_f1`, `test_recall`, `test_precision`
- `cv_auc_mean`, `cv_auc_std` (StratifiedKFold, k=5)

**Artefatos a salvar em `reports/figures/baselines/`:**
- `rf_confusion_matrix.png`, `rf_roc_curve.png`, `rf_feature_importance.png`
- `gb_confusion_matrix.png`, `gb_roc_curve.png`, `gb_feature_importance.png`

**MLflow tags:**
```python
{"notebook": "3.03", "fase": "baseline", "modelo": "random_forest", "framework": "sklearn"}
{"notebook": "3.03", "fase": "baseline", "modelo": "gradient_boosting", "framework": "sklearn"}
```

#### Critérios de aceite

- Dois runs MLflow separados no experimento `churn-telecom`
- Feature importance via `feature_importances_` (Gini) logada como JSON e como figura
- Mesmo padrão de registro que os notebooks 3.01 e 3.02 (`setup_mlflow`, `mlflow_run`, `log_dataset_to_mlflow`)
- Modelos registrados no Model Registry: `"baseline-random-forest"`, `"baseline-gradient-boosting"`

---

### TASK-2.04 — Tabela comparativa de modelos (≥ 4 métricas)

**Arquivo:** `notebooks/4.02_vab_model_comparison.ipynb`
**Referência:** Fundamentos, Aula 05

#### Descrição

Carregar todos os runs do MLflow e construir a tabela comparativa final com todos
os modelos treinados nas Etapas 1 e 2. Salvar como artefato CSV e figura.

#### Modelos incluídos na comparação

| Run MLflow | Modelo | Família |
|---|---|---|
| `dummy_most_frequent` | DummyClassifier | piso |
| `3.02_baseline_logistic` | LogisticRegression | linear |
| `3.03_baseline_rf` | RandomForest | árvore |
| `3.03_baseline_gb` | GradientBoosting | árvore |
| `4.01_mlp_pytorch` | MLP PyTorch | rede neural |

#### Métricas da tabela (≥ 4 obrigatórias)

```
test_auc       — AUC-ROC          (métrica primária de ranking)
test_pr_auc    — PR-AUC           (métrica para classe desbalanceada)
test_f1        — F1-Score (churn) (métrica operacional)
test_recall    — Recall (churn)   (cobertura de risco)
test_precision — Precision        (eficiência da abordagem)
test_mcc       — MCC              (robustez ao desbalanceamento)
```

#### Artefatos a salvar

- `reports/figures/comparison/model_comparison_bar.png` — barplot por métrica
- `reports/figures/comparison/model_comparison_roc.png` — curvas ROC sobrepostas
- `reports/figures/comparison/model_comparison_table.csv` — tabela completa

#### Critérios de aceite

- Tabela gerada programaticamente via `mlflow.search_runs()` — sem hardcode de valores
- Figura de barplot salva antes de ser logada no MLflow
- Run MLflow dedicado: `"4.02_model_comparison"` com a tabela como artefato
- Melhor modelo identificado por `test_auc` e apontado em `logger.info()`

---

### TASK-2.05 — Análise de trade-off de custo (FP vs FN)

**Arquivo:** `notebooks/4.02_vab_model_comparison.ipynb` (seção dedicada)
**Referência:** Fundamentos, Aula 05

#### Descrição

Quantificar o impacto financeiro de cada tipo de erro para o modelo MLP e para o
melhor modelo de árvore, usando o modelo de custo definido em
`docs/metrica_tecnica_negocio.md`.

#### Modelo de custo (conforme `metrica_tecnica_negocio.md`)

```
CLV   = R$ 2.400  (receita média por cliente retido)
C_ret = R$ 50     (custo da oferta de retenção)
r_ret = 40%       (taxa de sucesso da retenção quando abordado)

Impacto por célula da matriz de confusão:
  TP → benefício = CLV × r_ret − C_ret = +R$ 910
  TN → impacto   = R$ 0
  FP → custo     = −C_ret = −R$ 50
  FN → perda     = −CLV   = −R$ 2.400  (48× mais caro que FP)

Net_Savings = TP × 910 − FP × 50 − FN × 2.400
```

#### Análise a implementar

Para cada modelo da tabela comparativa, calcular e registrar no MLflow:
```python
mlflow.log_metric("net_savings",           net_savings)
mlflow.log_metric("cost_fn_total",         fn * 2400)
mlflow.log_metric("cost_fp_total",         fp * 50)
mlflow.log_metric("savings_vs_no_model",   net_savings - baseline_no_model)
```

**Artefato a salvar:**
- `reports/figures/comparison/cost_analysis.png`
  — barplot de `net_savings` por modelo
  — anotação do custo do Dummy (pior caso = sem modelo)

**Análise de threshold para MLP:**
```python
# Varre thresholds de 0.20 a 0.80 e plota Net_Savings × threshold
# Identifica threshold ótimo para maximizar Net_Savings
# Salva em: reports/figures/mlp/threshold_analysis.png
```

#### Critérios de aceite

- `net_savings` calculado e logado para todos os modelos comparados
- Threshold ótimo da MLP identificado e logado: `mlflow.log_param("optimal_threshold", ...)`
- Figura de análise de custo salva em `reports/figures/comparison/`
- Conclusão documentada no log: qual modelo maximiza Net_Savings no test set

---

### TASK-2.06 — Registro completo no MLflow

**Aplica-se a:** TASK-2.01, TASK-2.02, TASK-2.03, TASK-2.04, TASK-2.05
**Referência:** Ciclo de Vida, Aula 02

#### Padrão obrigatório para todos os runs da Etapa 2

**Tags mínimas:**
```python
mlflow.set_tags({
    "notebook":    "4.01",              # número do notebook
    "fase":        "modelagem",         # "baseline" ou "modelagem"
    "modelo":      "mlp_pytorch",       # nome do modelo
    "framework":   "pytorch",           # "sklearn" ou "pytorch"
    "task":        "classification",
    "etapa":       "2",                 # novo na Etapa 2
})
```

**Dataset versionado (obrigatório):**
```python
log_dataset_to_mlflow(X_train, y_train, split="train", source_path=TRAIN_PATH)
log_dataset_to_mlflow(X_test,  y_test,  split="test",  source_path=TEST_PATH)
```

**Parâmetros obrigatórios para a MLP:**
```python
mlflow.log_params({
    "input_dim":        30,
    "hidden_dims":      "[128, 64, 32]",
    "dropout_rates":    "[0.3, 0.3, 0.2]",
    "activation":       "relu",
    "loss_fn":          "BCEWithLogitsLoss",
    "optimizer":        "Adam",
    "lr":               1e-3,
    "weight_decay":     1e-4,
    "batch_size":       64,
    "max_epochs":       200,
    "early_stopping_patience": 15,
    "pos_weight":       2.77,
    "best_epoch":       ...,    # preenchido após treinamento
    "optimal_threshold": ...,   # preenchido após análise
})
```

**Artefatos obrigatórios logados no MLflow:**
```
figures/training_curves.png        (MLP)
figures/confusion_matrix.png       (todos)
figures/roc_curve.png              (todos)
figures/feature_importance.png     (árvores e MLP via permutation)
comparison/model_comparison_bar.png
comparison/model_comparison_roc.png
comparison/cost_analysis.png
```

#### Critérios de aceite

- Todos os runs visíveis no mesmo experimento `churn-telecom` via `mlflow ui`
- Nenhum run com status `FAILED` no MLflow
- Todos os artefatos acessíveis via MLflow UI em `mlartifacts/`
- Parâmetro `etapa=2` permite filtrar runs da Etapa 2 no MLflow UI

---

## Entregável da Etapa 2

Conforme o Tech Challenge:

> **Tabela comparativa de modelos + MLP treinado + artefatos no MLflow**

### Checklist de entrega

- [ ] `churn_telecom/models.py` com classe `ChurnMLP`
- [ ] `notebooks/3.03_vab_baseline_trees.ipynb` executado (RF + GBM)
- [ ] `notebooks/4.01_vab_mlp_architecture.ipynb` executado (MLP treinada)
- [ ] `notebooks/4.02_vab_model_comparison.ipynb` executado (tabela + custo)
- [ ] `models/mlp_best.pt` salvo
- [ ] `reports/figures/mlp/` com `training_curves.png`
- [ ] `reports/figures/comparison/` com `model_comparison_bar.png`, `model_comparison_roc.png`, `cost_analysis.png`
- [ ] MLflow UI com ≥ 5 runs comparáveis no experimento `churn-telecom`
- [ ] `README.md` atualizado com resultados da Etapa 2

---

## Ordem de execução

```
1. Criar churn_telecom/models.py          (TASK-2.01)
2. Executar 3.03_vab_baseline_trees       (TASK-2.03)
3. Executar 4.01_vab_mlp_architecture     (TASK-2.01 + 2.02)
4. Executar 4.02_vab_model_comparison     (TASK-2.04 + 2.05)
5. Verificar MLflow UI                    (TASK-2.06)
6. Atualizar README.md
```

---

## Estrutura de arquivos ao final da Etapa 2

```
projeto_final/
├── churn_telecom/
│   ├── config.py
│   ├── models.py          ← NOVO (TASK-2.01)
│   └── plots.py
├── models/
│   ├── preprocessor.pkl
│   └── mlp_best.pt        ← NOVO (TASK-2.02)
├── notebooks/
│   ├── 3.03_vab_baseline_trees.ipynb     ← NOVO (TASK-2.03)
│   ├── 4.01_vab_mlp_architecture.ipynb   ← NOVO (TASK-2.01 + 2.02)
│   └── 4.02_vab_model_comparison.ipynb   ← NOVO (TASK-2.04 + 2.05)
└── reports/figures/
    ├── baselines/          ← Etapa 1 (existente)
    ├── mlp/               ← NOVO
    │   ├── training_curves.png
    │   └── threshold_analysis.png
    └── comparison/        ← NOVO
        ├── model_comparison_bar.png
        ├── model_comparison_roc.png
        ├── model_comparison_table.csv
        └── cost_analysis.png
```

---

*PRD gerado como parte do Tech Challenge POSTECH MLOps — Fase 1.*
*Baseia-se nos requisitos da Etapa 2 e nos artefatos produzidos na Etapa 1.*