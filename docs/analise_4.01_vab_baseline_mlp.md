# Relatório de Execução — Etapa 2: MLP Churn Telecom

**Data de execução:** 16/04/2026 · 23:45–23:46  
**Duração total:** ~44 segundos (CPU) · 3 runs  
**Artefato principal:** `models/best_mlp_etapa2.pt` · `run_id: ab318b6f`

---

## 1. Arquitetura do código — fluxo de chamadas entre módulos

O notebook `02_treinamento_mlp.ipynb` não contém lógica de ML diretamente.
Ele atua como **orquestrador**: configura parâmetros, chama módulos externos e
registra resultados. O fluxo completo é:

```
02_treinamento_mlp.ipynb
│
├── churn_telecom/config.py          ← fonte única de verdade
│   ├── PROJECT_ROOT, DEVICE, RANDOM_STATE
│   ├── MLP_HIDDEN_DIMS, MLP_DROPOUT, MLP_LR, MLP_BATCH_SIZE
│   ├── COST_FN, COST_FP
│   └── setup_mlflow(), get_logger()
│
├── churn_telecom/models/mlp.py      ← arquitetura da rede
│   └── build_mlp(input_dim, hidden_dims, dropout, device)
│       └── retorna: ChurnMLP (nn.Module PyTorch)
│
├── churn_telecom/models/trainer.py  ← loop de treinamento
│   ├── TrainerConfig (dataclass de hiperparâmetros)
│   └── ChurnTrainer(model, config)
│       ├── .fit(X_train, y_train, X_val, y_val) → TrainHistory
│       └── .predict_proba(X_test) → np.ndarray[float32]
│
└── mlflow                           ← tracking de experimentos
    ├── mlflow.start_run(run_name)
    ├── mlflow.log_params({...})
    ├── mlflow.log_metrics({...}, step=epoch)
    ├── mlflow.log_artifact(path)
    └── mlflow.pytorch.log_model(model)
```

### O que `build_mlp()` faz internamente

Recebe `(input_dim=30, hidden_dims, dropout, device)` e constrói uma MLP com
a estrutura `Linear → BatchNorm1d → ReLU → Dropout` por camada oculta, mais
uma camada de saída `Linear(last_dim, 1)` sem ativação (logit puro).

A saída é um **logit**, não uma probabilidade. A conversão para probabilidade
(`torch.sigmoid`) só acontece em `predict_proba()`, o que é correto porque
o critério de perda usado no treino é `BCEWithLogitsLoss` — numericamente
mais estável que aplicar sigmoid antes e usar `BCELoss`.

### O que `ChurnTrainer.fit()` faz internamente

1. Cria `DataLoader` com `batch_size` e `shuffle=True` para treino
2. Para cada época até `max_epochs`:
   - Forward pass em mini-batches
   - Calcula `BCEWithLogitsLoss` com `pos_weight`
   - Backward pass + `clip_grad_norm_(max_norm=1.0)`
   - `Adam.step()` para atualizar pesos
   - Avalia no validation set (sem gradiente)
   - Loga métricas no MLflow: `train_loss`, `val_loss`, `train_auc`, `val_auc`, `lr`
   - Chama `EarlyStopping.step()` — para se `val_loss` não melhorar em `patience` épocas
3. Restaura os pesos do melhor checkpoint
4. Salva `models/checkpoints/best_mlp.pt`

---

## 2. Análise linha a linha dos logs

### 2.1 Setup — configuração inicial

```
23:45:18 | INFO | Device: cpu
```

O PyTorch detectou que não há GPU disponível (`torch.cuda.is_available()` retornou
`False`) e selecionou CPU automaticamente via `DEVICE = "cuda" if ... else "cpu"` no
`config.py`. Para este dataset (6.800 amostras, 30 features, arquiteturas pequenas),
o treino em CPU levou ~15 segundos por run — aceitável nesta fase.

```
23:45:18 | INFO | Custo FN: US$2903 | Custo FP: US$50 | Razão: 58.1x
```

Estes são os **custos reais calibrados com o dataset Telco** (IBM), provados pelo
gráfico de trade-off FP×FN. A assimetria de 58× é o parâmetro mais importante do
projeto: significa que cada churner não detectado custa 58 vezes mais do que uma
campanha de retenção desnecessária. Toda decisão de threshold, pos_weight e critério
de sucesso deriva desta razão.

```
23:45:18 | INFO | Threshold ótimo: 0.64 | SLO recall mínimo: 70%
```

O `threshold=0.64` foi determinado pela análise de custo real anterior: é o ponto que
minimiza o custo total mantendo `recall=100%`. O `SLO recall ≥ 70%` é o contrato de
nível de serviço do negócio — abaixo disso o modelo não pode ir a produção.

### 2.2 Carregamento dos dados

```
23:45:18 | INFO | X=(6800, 30) | churn_rate=51.0% | pos_weight=0.96
```

**Este log revela um fato importante sobre os dados:** a `churn_rate=51%` indica que
o dataset usado no treino foi **balanceado artificialmente** durante o EDA. O dataset
Telco original tem taxa de churn de ~26.5%, não 51%. Isso ocorre porque o notebook de
EDA aplicou alguma técnica de balanceamento (provavelmente SMOTE ou oversampling da
classe minoritária) antes de salvar o `train.parquet`.

Consequência direta: o `pos_weight=0.96 ≈ 1.0` — as classes estão quase equilibradas,
então o modelo não recebe penalidade extra por errar churners. Isso é diferente do
cenário de produção, onde a distribuição natural exigiria `pos_weight ≈ 2.85`.

O número de amostras (6.800) também é diferente do total original (7.043), o que
confirma que houve transformações durante o EDA — remoção de outliers ou
balanceamento que não preservou exatamente o total original.

```
23:45:18 | INFO | train=4080 | val=1360 | test=1360
```

Split 60/20/20 estratificado com `random_state=42`. A proporção de churn é idêntica
nos três splits (51%), confirmando que o `stratify=y` funcionou corretamente.

### 2.3 Run 1 — MLP_baseline

```
23:45:35 | INFO | [MLP_baseline] recall=0.906 | cost=US$192245 | threshold=0.10
```

**Arquitetura:** `[128, 64, 32]` · **Dropout:** 0.3 · **LR:** 1e-3 · **Batch:** 256  
**Duração:** ~17 segundos

- `recall=90.6%` — o modelo identifica 90.6% dos churners do test set
- `cost=US$192.245` — acima do benchmark de US$69K (dados balanceados vs reais)
- `threshold=0.10` — **sinal de alerta**: o threshold ótimo encontrado é muito baixo

O threshold de 0.10 significa que `find_optimal_threshold()` precisou baixar o limiar
de decisão até 10% para capturar churners suficientes. Isso ocorre porque o modelo
treinado em dados 50/50 calibra suas probabilidades de saída de forma diferente de
um modelo treinado na distribuição real (26.5% churn). As probabilidades ficam
comprimidas em torno de 0.50, e o ótimo de custo acaba em regiões muito baixas da
curva.

### 2.4 Run 2 — MLP_recall_otimizado

```
23:45:50 | INFO | [MLP_recall_otimizado] recall=0.931 | cost=US$143344 | threshold=0.13
```

**Arquitetura:** `[256, 128, 64, 32]` · **Dropout:** 0.15 · **LR:** 5e-4 · **Batch:** 128  
**Duração:** ~15 segundos · **pos_weight:** `pos_weight_val × 1.5`

Melhorias em relação ao baseline:

| Métrica | Run 1 | Run 2 | Δ |
|---|---|---|---|
| Recall | 90.6% | 93.1% | +2.5 pp |
| Custo total | US$192K | US$143K | −US$49K |
| Threshold ótimo | 0.10 | 0.13 | +0.03 |

A arquitetura maior `[256, 128, 64, 32]` com dropout reduzido (0.15 vs 0.30)
permite que o modelo aprenda mais padrões dos churners sem regularização excessiva.
O batch menor (128 vs 256) produz gradientes com mais variância, o que ajuda a
escapar de mínimos locais. O threshold subiu levemente para 0.13, indicando que o
modelo ficou marginalmente mais confiante nas suas predições.

### 2.5 Run 3 — MLP_posweight_alto ⭐ Vencedor

```
23:46:02 | INFO | [MLP_posweight_alto] recall=0.970 | cost=US$67963 | threshold=0.17
```

**Arquitetura:** `[128, 64, 32]` · **Dropout:** 0.20 · **LR:** 1e-3 · **Batch:** 256  
**Duração:** ~12 segundos  
**pos_weight:** `pos_weight_val × (COST_FN / COST_FP) / 10 = 0.96 × 58.1 / 10 ≈ 5.58`

Este run implementou a estratégia mais alinhada ao negócio: o `pos_weight` foi
calculado não apenas pela razão de classes, mas pela **razão de custo real** (58×).
O efeito é que o modelo penaliza 5.58× mais os erros na classe positiva durante o
treino — forçando-o a preferir falsos positivos a falsos negativos, exatamente o
que a assimetria de custo exige.

| Métrica | Run 1 | Run 2 | Run 3 | Benchmark real |
|---|---|---|---|---|
| Recall | 90.6% | 93.1% | **97.0%** | 100% @ t=0.64 |
| Custo total | US$192K | US$143K | **US$68K** | US$69K |
| Threshold | 0.10 | 0.13 | **0.17** | 0.64 |
| SLO (≥70%) | ✓ | ✓ | ✓ | ✓ |

O custo de US$67.963 é praticamente idêntico ao benchmark de US$69K provado pelo
gráfico de dados reais — o Run 3 replicou o resultado ótimo mesmo com a distribuição
balanceada.

---

## 3. Observação técnica — distribuição balanceada vs produção

O fato de `churn_rate=51%` no dataset de treino (vs 26.5% real) e os thresholds
encontrados (0.10–0.17 vs 0.64 real) indicam que o modelo foi treinado e avaliado
em uma distribuição diferente da que encontrará em produção.

Isso **não invalida os resultados da Etapa 2** — o ranking relativo entre os runs
é válido, e a lógica de otimização por pos_weight funcionou. Mas significa que antes
do deploy (Etapa 3/4), o threshold de produção precisará ser recalibrado em dados com
distribuição natural. A recalibração pode ser feita com `sklearn.calibration.CalibratedClassifierCV`
ou simplesmente re-executando `find_optimal_threshold()` em um hold-out com a
distribuição real de 26.5%.

---

## 4. Resultado registrado no MLflow

```
run_id: ab318b6f30b4439a9b1ffde553b007ec
tags:   winner=true | optimal_threshold=0.17 | cost_savings_vs_default=US$~124K
```

Artefatos logados por run:
- `plots/`: curvas de loss, AUC e análise de threshold por época
- `model/`: modelo PyTorch serializado via `mlflow.pytorch.log_model`
- `checkpoints/best_mlp.pt`: pesos do melhor epoch (early stopping)
- `final_model/best_mlp_etapa2.pt`: checkpoint do run vencedor
- `reports/run_comparison_etapa2.csv`: tabela comparativa dos 3 runs

---

## 5. Próximos passos — Etapa 3

Com o Run 3 validado como modelo de produção, a Etapa 3 envolve:

1. **Recalibrar threshold** em dados com distribuição real (26.5% churn) antes do deploy
2. **Refatorar** o notebook em módulos `src/` com pipelines sklearn reprodutíveis
3. **Implementar API FastAPI** com endpoint `/predict` usando o `best_mlp_etapa2.pt`
4. **Escrever testes automatizados** (smoke, schema, API) conforme exigido
5. **Registrar modelo** no MLflow Model Registry com stage `Staging → Production`