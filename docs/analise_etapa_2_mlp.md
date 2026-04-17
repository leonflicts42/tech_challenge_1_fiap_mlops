# Relatório Técnico — Etapa 2: Modelagem com Redes Neurais

**Projeto:** Previsão de Churn — Telecomunicações  
**Fase:** Tech Challenge Fase 1 · FIAP MLOps  
**Etapa:** 2 de 4 — Construção, Treinamento e Avaliação da MLP  
**Status:** Artefatos implementados · Execução pendente de dados processados  

---

## Sumário

1. [Visão geral da etapa](#1-visão-geral-da-etapa)
2. [Arquitetura dos artefatos](#2-arquitetura-dos-artefatos)
3. [Como cada artefato funciona](#3-como-cada-artefato-funciona)
4. [Correspondência com os requisitos do Tech Challenge](#4-correspondência-com-os-requisitos-do-tech-challenge)
5. [O que falta para executar de fato](#5-o-que-falta-para-executar-de-fato)
6. [Notebook ou script Python — qual usar?](#6-notebook-ou-script-python--qual-usar)
7. [Sequência de execução recomendada](#7-sequência-de-execução-recomendada)

---

## 1. Visão geral da etapa

A Etapa 2 tem como objetivo central **construir e validar uma rede neural MLP em PyTorch** para prever churn no setor de telecomunicações, comparar seu desempenho contra os baselines da Etapa 1, analisar o custo de negócio de cada tipo de erro e registrar tudo no MLflow.

O foco de otimização definido para o projeto é a **minimização de falsos negativos** — clientes que iriam cancelar mas não foram identificados pelo modelo. Em telecomunicações, cada falso negativo representa a perda de um cliente cujo valor de ciclo de vida (CLV) estimado é de R$ 1.200,00/ano, enquanto o custo de uma ação de retenção equivocada (falso positivo) é de apenas R$ 80,00. Essa assimetria de custo justifica a priorização de **Recall** e **PR-AUC** como métricas principais, sem abrir mão do ROC-AUC como referência de separabilidade geral.

---

## 2. Arquitetura dos artefatos

```
churn_telecom/
└── models/
    ├── __init__.py        ← expõe todos os símbolos públicos do pacote
    ├── mlp.py             ← Etapa 1: ChurnMLP + build_mlp (arquitetura)
    ├── trainer.py         ← Etapa 2: ChurnTrainer + EarlyStopping (treino)
    ├── evaluation.py      ← Etapa 2: métricas, custo, tabela comparativa
    └── experiment.py      ← Etapa 2: runner MLflow (orquestra tudo)

tests/
├── test_mlp.py            ← Etapa 1: 12 testes da arquitetura MLP
└── test_etapa2.py         ← Etapa 2: 18 testes do trainer e avaliação

data/
└── processed/             ← ⚠️ AINDA NÃO EXISTE — gerado pelo notebook de EDA
    ├── features.npy
    └── target.npy

models/
└── checkpoints/           ← criado automaticamente pelo trainer
    └── best_mlp.pt        ← salvo após cada fit()
```

### Fluxo de dependências entre módulos

```
mlp.py
  └── build_mlp() → ChurnMLP
        ↓
trainer.py
  └── ChurnTrainer.fit(X_train, y_train, X_val, y_val)
        ├── usa: EarlyStopping
        ├── usa: BCEWithLogitsLoss + Adam + ReduceLROnPlateau
        ├── salva: models/checkpoints/best_mlp.pt
        └── loga: MLflow (métricas por época)
              ↓
evaluation.py
  └── MetricsCalculator.compute(name, y_true, y_proba) → ModelMetrics
        ↓
      CostAnalyzer.annotate(metrics) → ModelMetrics com custo
        ↓
      ModelComparator.summary() → DataFrame ordenado por ROC-AUC
              ↓
experiment.py
  └── run_experiment()
        ├── carrega: data/processed/features.npy
        ├── treina: Dummy, LR, RF, GBM, MLP
        ├── avalia: todos com MetricsCalculator + CostAnalyzer
        ├── salva: models/model_comparison.csv
        └── loga: MLflow (runs filhos + run pai)
```

---

## 3. Como cada artefato funciona

### 3.1 `mlp.py` — Arquitetura da rede neural (Etapa 1)

A classe `ChurnMLP` define uma **MLP totalmente parametrizável** com a seguinte estrutura por camada oculta:

```
Linear(in, out) → BatchNorm1d → ReLU → Dropout
```

A saída é um **logit puro** (sem Sigmoid), o que é a prática correta quando se usa `BCEWithLogitsLoss`, pois evita instabilidade numérica no cálculo do gradiente. A inicialização dos pesos segue **Kaiming Normal**, adequada para ativações ReLU.

A factory `build_mlp(n_features, hidden_dims, dropout, device)` recebe os hiperparâmetros, fixa a seed antes da inicialização e retorna o modelo já movido para o device correto.

**Exemplo com a configuração padrão do projeto:**

```python
model = build_mlp(
    n_features=19,        # features do dataset Telco após pré-processamento
    hidden_dims=[128, 64, 32],
    dropout=0.3,
    device="cpu"
)
# → ChurnMLP com 3 camadas ocultas: 128→64→32→1
```

### 3.2 `trainer.py` — Loop de treinamento

O `ChurnTrainer` é responsável por **uma única coisa**: executar o loop de treinamento com todas as boas práticas de engenharia. Suas responsabilidades são:

#### Early Stopping

A classe `EarlyStopping` monitora a `val_loss` a cada época. Se não houver melhora superior a `min_delta=1e-4` por `patience=10` épocas consecutivas, o treino é interrompido e os pesos do melhor checkpoint são restaurados automaticamente via `restore_best()`.

```
Época 1:  val_loss=0.612 → melhora → counter=0, salva best_state
Época 2:  val_loss=0.608 → melhora → counter=0, salva best_state
...
Época 47: val_loss=0.591 → sem melhora → counter=1
Época 48: val_loss=0.593 → sem melhora → counter=2
...
Época 57: val_loss=0.595 → counter=10 → STOP → restaura pesos da época 47
```

#### Batching e otimização

O `DataLoader` do PyTorch divide os dados em mini-batches de `batch_size=256` amostras. A cada batch:

1. Forward pass — logits gerados pelo modelo
2. `BCEWithLogitsLoss` calculada (com `pos_weight` para balancear classes)
3. Backward pass — gradientes calculados
4. `clip_grad_norm_(max_norm=1.0)` — evita explosão de gradientes
5. `Adam.step()` — pesos atualizados

O scheduler `ReduceLROnPlateau` reduz a taxa de aprendizado à metade se a `val_loss` não melhorar por 5 épocas, refinando a convergência no final do treino.

#### Balanceamento de classes

O `pos_weight` é calculado automaticamente como `n_negativos / n_positivos`. No dataset Telco, onde ~26% dos clientes fazem churn, isso resulta em `pos_weight ≈ 2.85` — o modelo penaliza 2,85× mais os erros na classe positiva (churn=1), o que é diretamente alinhado ao objetivo de minimizar falsos negativos.

#### Registro no MLflow

A cada época, o trainer loga automaticamente:

| Métrica logada | O que representa |
|---|---|
| `train_loss` | BCE loss no conjunto de treino |
| `val_loss` | BCE loss no conjunto de validação |
| `train_auc` | ROC-AUC no treino (tendência de overfitting) |
| `val_auc` | ROC-AUC na validação (performance real) |
| `lr` | taxa de aprendizado atual (monitorar ReduceLROnPlateau) |

### 3.3 `evaluation.py` — Métricas e análise de negócio

Três classes com responsabilidades únicas:

#### `MetricsCalculator`

Recebe `(y_true, y_proba)` e retorna um `ModelMetrics` com:

- **ROC-AUC** — separabilidade geral entre classes
- **PR-AUC** (Average Precision) — mais informativo que ROC-AUC em datasets desbalanceados
- **F1** — média harmônica entre precisão e recall
- **Recall** (Sensibilidade) — taxa de verdadeiros positivos = TP/(TP+FN)
- **Specificity** — taxa de verdadeiros negativos = TN/(TN+FP)
- **Matriz de confusão completa** — TN, FP, FN, TP

A escolha de incluir **Specificity** explicitamente é estratégica: ela mede a capacidade do modelo de identificar quem *não vai sair*, que é precisamente o objetivo declarado no Tech Challenge — "identificação de verdadeiros negativos para reter clientes que de fato iriam sair".

#### `CostAnalyzer`

Anota cada `ModelMetrics` com as métricas financeiras do problema:

```
custo_total    = FP × R$80  + FN × R$1.200
churn_avoided  = TP × (R$1.200 - R$80)  =  TP × R$1.120
net_value      = churn_avoided - custo_total
ROI            = net_value / custo_total
```

Isso permite comparar modelos não apenas por métricas técnicas, mas pelo **impacto financeiro real** de cada política de decisão.

#### `ModelComparator`

Agrega múltiplos `ModelMetrics` e gera um `DataFrame` ordenado por ROC-AUC decrescente, com todas as métricas lado a lado — o entregável final da Etapa 2.

### 3.4 `experiment.py` — Runner MLflow

Orquestra o treinamento e avaliação de todos os modelos em um único script, criando a estrutura de runs no MLflow:

```
Run pai: "etapa2_comparacao"
├── Run filho: DummyClassifier   (baseline de referência)
├── Run filho: LogisticRegression (baseline linear)
├── Run filho: RandomForest       (ensemble de árvores)
├── Run filho: GradientBoosting   (ensemble boosted)
└── Run filho: ChurnMLP           (modelo principal)
```

Ao final, salva dois artefatos CSV:
- `models/model_comparison.csv` — tabela com ≥ 4 métricas técnicas de todos os modelos
- `models/cost_tradeoff.csv` — análise de trade-off FP/FN com net_value e ROI

---

## 4. Correspondência com os requisitos do Tech Challenge

| Requisito da Etapa 2 | Status | Como é atendido |
|---|---|---|
| Construir MLP em PyTorch com arquitetura definida | ✅ Implementado | `ChurnMLP` em `mlp.py` — `Linear→BN→ReLU→Dropout` parametrizável |
| Implementar loop com early stopping | ✅ Implementado | `EarlyStopping` em `trainer.py` — restaura melhor checkpoint |
| Implementar batching | ✅ Implementado | `DataLoader` com `batch_size=256` em `ChurnTrainer._make_loader()` |
| Comparar MLP vs baselines com ≥ 4 métricas | ✅ Implementado | `ModelComparator` — ROC-AUC, PR-AUC, F1, Recall, Specificity |
| Analisar trade-off FP vs FN | ✅ Implementado | `CostAnalyzer` — custo_total, churn_avoided, net_value, ROI |
| Registrar todos os experimentos no MLflow | ✅ Implementado | `experiment.py` — runs filhos + métricas por época |
| Seeds fixadas para reprodutibilidade | ✅ Implementado | `_set_seed(42)` em `trainer.py` + `random_state=42` nos sklearn |
| Validação cruzada estratificada | ✅ Implementado | `_cv_evaluate()` em `experiment.py` — `StratifiedKFold(n_splits=5)` |
| Logging estruturado sem `print()` | ✅ Implementado | `logging.getLogger(__name__)` em todos os módulos |
| Testes automatizados | ✅ Implementado | `test_etapa2.py` — 18 testes cobrindo trainer, métricas e custo |
| Linting com ruff | ✅ Validado | `ruff check` — All checks passed |

---

## 5. O que falta para executar de fato

A execução do `experiment.py` depende de **um pré-requisito crítico** que ainda não existe: os dados processados gerados pelo notebook de EDA da Etapa 1.

### 5.1 Pré-requisito: dados processados

O `experiment.py` espera dois arquivos em `data/processed/`:

```python
# experiment.py, função _load_data()
X = np.load("data/processed/features.npy")   # ← não existe ainda
y = np.load("data/processed/target.npy")     # ← não existe ainda
```

Para gerá-los, o notebook de EDA precisa exportar os arrays ao final do pré-processamento:

```python
# Adicionar ao final do notebook de EDA (Etapa 1)
import numpy as np
from pathlib import Path

Path("data/processed").mkdir(parents=True, exist_ok=True)
np.save("data/processed/features.npy", X_processed)  # array float32 já escalado
np.save("data/processed/target.npy", y.values.astype(np.float32))
```

> **Atenção:** `X_processed` deve ser o array já pré-processado pelo pipeline sklearn da Etapa 1 — com encoding de variáveis categóricas e normalização aplicados. O `experiment.py` consome features prontas, sem pré-processamento interno.

### 5.2 Pré-requisito: MLflow iniciado

O MLflow precisa estar rodando localmente para receber os logs:

```bash
# Terminal separado — manter aberto durante a execução
mlflow ui --port 5000
# Acessar: http://localhost:5000
```

Alternativamente, basta garantir que o `MLFLOW_TRACKING_URI` não está apontando para um servidor remoto indisponível. Se não configurado, o MLflow salva os runs localmente em `./mlruns/`.

### 5.3 Pré-requisito: pacote instalado em modo editável

```bash
pip install -e .
```

Sem isso, os imports `from churn_telecom.models import ...` falham.

### 5.4 Resumo: o que falta

| Item | Ação necessária | Onde fazer |
|---|---|---|
| `data/processed/features.npy` | Exportar do notebook de EDA | `notebooks/01_eda.ipynb` |
| `data/processed/target.npy` | Exportar do notebook de EDA | `notebooks/01_eda.ipynb` |
| MLflow rodando | `mlflow ui` no terminal | Terminal separado |
| Pacote instalado | `pip install -e .` | Terminal do projeto |

---

## 6. Notebook ou script Python — qual usar?

A resposta depende do que se quer fazer. Os dois formatos têm papéis distintos e **complementares** neste projeto.

### Use notebook para:

- **EDA e exploração** — análise visual, distribuições, correlações, geração dos `.npy`
- **Análise de resultados** — carregar as tabelas CSV geradas pelo `experiment.py` e criar visualizações para o relatório e o vídeo STAR
- **Debug iterativo** — testar o `ChurnTrainer` manualmente com um subconjunto dos dados antes de rodar o experimento completo

```python
# Exemplo de uso do trainer diretamente no notebook
from churn_telecom.models import build_mlp, ChurnTrainer, TrainerConfig
import mlflow

mlflow.set_experiment("debug_notebook")

with mlflow.start_run(run_name="teste_manual"):
    model = build_mlp(X_train.shape[1], [128, 64, 32], 0.3, "cpu")
    cfg = TrainerConfig(epochs=20, patience=5)
    trainer = ChurnTrainer(model, cfg)
    history = trainer.fit(X_train, y_train, X_val, y_val)
```

### Use script Python (`experiment.py`) para:

- **Execução reprodutível do experimento completo** — todos os modelos, métricas e artefatos em um único comando
- **CI/CD e automação** — pode ser chamado por `Makefile`, GitHub Actions ou qualquer orquestrador
- **Commit limpo no repositório** — scripts versionados são rastreáveis; notebooks com outputs têm diffs poluídos

```bash
# Execução completa do experimento
python -m churn_telecom.models.experiment
```

### Recomendação para este projeto

| Atividade | Formato |
|---|---|
| EDA + geração dos `.npy` | Notebook (`01_eda.ipynb`) |
| Debug/testes do trainer | Notebook (`02_debug_trainer.ipynb`) |
| Experimento oficial registrado no MLflow | Script (`experiment.py`) |
| Análise dos resultados + tabela comparativa | Notebook (`03_resultados.ipynb`) |
| Geração dos gráficos para o vídeo STAR | Notebook (`03_resultados.ipynb`) |

Esta separação é a prática padrão em projetos MLOps: **notebooks para exploração, scripts para execução**.

---

## 7. Sequência de execução recomendada

```bash
# 1. Garantir que o pacote está instalado
pip install -e .

# 2. Rodar o notebook de EDA e gerar os dados processados
#    → abre o Jupyter e executa notebooks/01_eda.ipynb até o final
#    → confirma que data/processed/features.npy existe
ls data/processed/

# 3. Subir o MLflow (terminal separado)
mlflow ui --port 5000

# 4. Rodar os testes para confirmar que tudo está ok
pytest tests/test_mlp.py tests/test_etapa2.py -v
# esperado: 29 passed

# 5. Executar o experimento completo
python -m churn_telecom.models.experiment

# 6. Verificar os resultados
#    → http://localhost:5000 (MLflow UI)
#    → cat models/model_comparison.csv
#    → cat models/cost_tradeoff.csv

# 7. Rodar linting final
ruff check churn_telecom/ tests/
```

### Resultado esperado no MLflow após a execução

```
Experiment: churn_telecom_etapa2
└── Run: etapa2_comparacao
    ├── Parâmetros: seed=42, threshold=0.5, cost_fn=1200, cost_fp=80
    ├── Métricas: best_roc_auc, best_pr_auc, best_f1, best_model
    ├── Artifacts/reports/model_comparison.csv
    ├── Artifacts/reports/cost_tradeoff.csv
    ├── Artifacts/checkpoints/best_mlp.pt
    └── Runs filhos:
        ├── DummyClassifier   → val_roc_auc, val_pr_auc
        ├── LogisticRegression → val_roc_auc, val_pr_auc
        ├── RandomForest      → val_roc_auc, val_pr_auc
        ├── GradientBoosting  → val_roc_auc, val_pr_auc
        └── ChurnMLP          → train_loss/val_loss/train_auc/val_auc por época
                                best_epoch, stopped_early
```

---

## Apêndice — Decisões técnicas e justificativas

### Por que `BCEWithLogitsLoss` e não `BCELoss`?

A `BCEWithLogitsLoss` funde internamente a Sigmoid com a BCE em uma única operação numericamente estável usando o truque `log-sum-exp`. Usar `BCELoss` exigiria aplicar `torch.sigmoid()` antes, o que introduz instabilidade para logits muito grandes ou muito negativos (saturação do gradiente).

### Por que `BatchNorm1d` antes de `ReLU`?

A BatchNorm normaliza as ativações pré-ReLU, garantindo que a distribuição das entradas de cada camada permaneça estável ao longo do treino. Isso acelera a convergência, reduz a sensibilidade à taxa de aprendizado e atua como regularizador implícito — especialmente importante em datasets tabulares pequenos como o Telco (~7k linhas).

### Por que `pos_weight` no critério de perda?

Com ~26% de churn no dataset Telco, a classe positiva é minoritária. Sem correção, o modelo aprende a prever "não churn" para a maioria dos casos e obtém ~74% de acurácia sem aprender nada útil. O `pos_weight = n_neg/n_pos ≈ 2.85` faz com que cada erro na classe positiva pese 2,85× mais, forçando o modelo a aprender as características dos clientes que cancelam.

### Por que PR-AUC como métrica principal além de ROC-AUC?

Em problemas desbalanceados, o ROC-AUC pode ser otimistamente alto mesmo para modelos fracos, porque leva em conta os verdadeiros negativos (que são abundantes). A PR-AUC (Precision-Recall AUC) foca exclusivamente no desempenho na classe positiva, sendo mais informativa para o objetivo de negócio: encontrar clientes em risco de churn em uma população majoritariamente estável.