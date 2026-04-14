# Métricas Técnicas e de Negócio — Churn Telecom

> **Etapa 1 do Tech Challenge** | Seção: Fundamentos, Aula 05  
> Objetivo: Definir o contrato de avaliação do modelo antes do treinamento.

---

## 1. Contexto: Por que a escolha de métricas importa aqui?

O dataset IBM Telco possui **~26,5% de churners** — desbalanceamento moderado. Nesse cenário:

- **Acurácia é enganosa**: o `DummyClassifier` atinge 73,5% sem aprender nada.
- **AUC-ROC pode mascarar fraquezas** em cenários muito desbalanceados, mas é adequado para ~26,5% (Smets et al., 2026 — limiar crítico está abaixo de 5%).
- **O custo assimétrico dos erros** é central: um Falso Negativo (churner não detectado) custa mais que um Falso Positivo (não-churner abordado desnecessariamente).

> **Regra de ouro do projeto**: a métrica técnica primária é **AUC-ROC** (ranking); a métrica operacional primária é **Recall da classe Churn** (cobertura de risco); a métrica de negócio é o **custo de churn evitado** (impacto financeiro real).

---

## 2. Métricas Técnicas

### 2.1 Matriz de Confusão — Terminologia Adotada

```
                  Predito: Churn=0    Predito: Churn=1
Real: Churn=0    TN (Verdadeiro -)   FP (Falso +)
Real: Churn=1    FN (Falso -)        TP (Verdadeiro +)
```

| Sigla | Significado de Negócio |
|---|---|
| **TP** | Churner corretamente identificado → pode receber oferta de retenção |
| **TN** | Cliente fiel corretamente identificado → sem ação necessária |
| **FP** | Cliente fiel erroneamente abordado → custo de oferta desperdiçada |
| **FN** | Churner não detectado → **perda de receita irrecuperável** |

---

### 2.2 AUC-ROC (Métrica Primária de Ranking)

**Fórmula:** Área sob a curva ROC (TPR × FPR para todos os thresholds)

**Por que usar:**
- Threshold-independent: avalia a qualidade do ranqueamento de risco, não de uma decisão binária.
- Padrão consolidado na literatura de churn telecom (Mirabdolbaghi 2022; Alboukaey 2023; todos os benchmarks IBM Telco).
- Permite comparar modelos independentemente da proporção de classes.

**Interpretação para este projeto:**

| AUC-ROC | Interpretação |
|---|---|
| 0.50 | Aleatório — equivalente ao DummyClassifier |
| 0.70–0.79 | Fraco — não suficiente para produção |
| 0.80–0.84 | Aceitável — baseline logístico esperado |
| **0.85–0.89** | **Bom — meta para a MLP** |
| ≥ 0.90 | Excelente — alcançável com ensemble ou feature engineering avançado |

**Meta do projeto:** `AUC-ROC ≥ 0.860` no conjunto de teste.

**Limitação conhecida:** Para desbalanceamentos extremos (<5%), PR-AUC deve ser priorizada (Smets et al., 2026). Para 26,5% de positivos, AUC-ROC é adequado.

---

### 2.3 PR-AUC / Average Precision (Métrica Complementar)

**Fórmula:** Área sob a curva Precision-Recall

**Por que usar:**
- Mais sensível ao desempenho na classe minoritária do que AUC-ROC.
- Especialmente útil durante otimização de threshold: permite escolher o ponto operacional que equilibra precision e recall conforme o custo de cada tipo de erro.
- Adotada como métrica complementar em benchmarks recentes (MDPI 2025; e-Profits 2026).

**Baseline de referência:** Um classificador aleatório obtém PR-AUC ≈ proporção da classe positiva ≈ **0.265**.

**Meta do projeto:** `PR-AUC ≥ 0.680` no conjunto de teste.

---

### 2.4 F1-Score da Classe Churn (Métrica Operacional)

**Fórmula:** `F1 = 2 × (Precision × Recall) / (Precision + Recall)`

**Por que usar:**
- Média harmônica de Precision e Recall — penaliza desequilíbrios entre os dois.
- Métrica mais intuitiva para comparar modelos no contexto de campanha de retenção.
- Amplamente usada na literatura para o IBM Telco (F1 típico para LR: 0.59–0.62).

**Meta do projeto:** `F1 ≥ 0.620` no conjunto de teste.

**Limitação:** F1 não distingue entre diferentes trade-offs de custo — um modelo com Recall 0.80 e Precision 0.50 tem o mesmo F1 que um com Recall 0.62 e Precision 0.64, mas o impacto de negócio é muito diferente.

---

### 2.5 Recall da Classe Churn (Métrica de Cobertura)

**Fórmula:** `Recall = TP / (TP + FN)`

**Por que é crítico neste projeto:**
- Mede a fração de churners reais que o modelo consegue identificar.
- Um Recall baixo significa que muitos clientes em risco **não são abordados** — perda de receita direta.
- O objetivo de negócio é **identificar os verdadeiros negativos de churn** (quem de fato vai sair), portanto maximizar Recall é prioritário — até o limite imposto pelo custo dos Falsos Positivos.

**Meta do projeto:** `Recall ≥ 0.600` no conjunto de teste.

**Trade-off documentado:** Aumentar Recall reduz Precision. O threshold ótimo deve ser escolhido via análise custo-benefício (ver Seção 4).

---

### 2.6 MCC — Matthews Correlation Coefficient (Métrica de Robustez)

**Fórmula:**
```
MCC = (TP×TN − FP×FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

**Por que incluir:**
- Considera todos os quadrantes da matriz de confusão simultaneamente.
- Menos sensível ao desbalanceamento de classes que Accuracy e F1.
- Varia de −1 (predição inversa perfeita) a +1 (predição perfeita); 0 = aleatório.
- Recomendado como métrica complementar por Mirabdolbaghi (2022) e Smets et al. (2026).

**Meta do projeto:** `MCC ≥ 0.440` no conjunto de teste.

---

### 2.7 NPV — Negative Predictive Value (Foco em Verdadeiros Negativos)

**Fórmula:** `NPV = TN / (TN + FN)`

**Relevância específica para este projeto:**
> O enunciado do Tech Challenge destaca explicitamente a **identificação de Verdadeiros Negativos** — clientes que de fato irão sair. O NPV mede a confiabilidade do modelo ao classificar alguém como "não vai churnar": quanto maior, menos churners passam despercebidos.

**Meta do projeto:** `NPV ≥ 0.880` — garantir que, entre os classificados como "seguros", menos de 12% sejam churners reais.

---

### 2.8 Tabela Resumo — Todas as Métricas Técnicas

| Métrica | Papel | Meta (Produção) | Baseline LR | Baseline Dummy |
|---|---|---|---|---|
| **AUC-ROC** | Primária (ranking) | ≥ 0.860 | 0.843 | 0.500 |
| **PR-AUC** | Complementar (classe positiva) | ≥ 0.680 | 0.651 | 0.265 |
| **F1 — Churn** | Operacional | ≥ 0.620 | 0.597 | 0.000 |
| **Recall — Churn** | Cobertura de risco | ≥ 0.600 | 0.558 | 0.000 |
| **Precision — Churn** | Eficiência da abordagem | ≥ 0.630 | 0.641 | N/A |
| **MCC** | Robustez | ≥ 0.440 | 0.415 | 0.000 |
| **NPV** | Verdadeiros Negativos | ≥ 0.880 | ~0.862 | ~0.735 |
| **Acurácia** | Informativa apenas | — | 0.799 | 0.735 |

---

## 3. Métricas de Negócio

### 3.1 Modelo de Custo

Para quantificar o impacto financeiro, adotamos o seguinte modelo simplificado:

```
Variáveis:
  CLV   = Customer Lifetime Value médio por cliente = R$ 2 400 (estimativa: 24 meses × R$ 100/mês)
  C_ret = Custo da oferta de retenção por cliente abordado = R$ 50
  r_ret = Taxa de sucesso da retenção quando abordado = 40% (estimativa conservadora)
  N     = Número de clientes avaliados no batch

Consequências por célula da matriz de confusão:
  TP: Churner identificado e abordado
    → Benefício esperado = CLV × r_ret − C_ret = 2400 × 0.40 − 50 = R$ +910
  TN: Não-churner corretamente ignorado
    → Impacto = R$ 0
  FP: Não-churner abordado desnecessariamente
    → Custo = −C_ret = R$ −50
  FN: Churner não identificado, cancelamento ocorre
    → Perda = −CLV = R$ −2 400
```

**Razão de custo FN/FP:** `2 400 / 50 = 48×`

> Um Falso Negativo custa **48 vezes mais** que um Falso Positivo. Isso justifica priorizar **Recall alto** mesmo ao custo de Precision menor.

---

### 3.2 Net Savings por Batch (Métrica de Negócio Principal)

**Fórmula:**
```
Net_Savings = TP × (CLV × r_ret − C_ret) − FP × C_ret − FN × CLV
            = TP × 910 − FP × 50 − FN × 2400
```

**Exemplo com 1 000 clientes avaliados (26,5% taxa de churn esperada = 265 churners):**

| Modelo | TP | FP | FN | TN | Net Savings |
|---|---|---|---|---|---|
| Dummy (sem modelo) | 0 | 0 | 265 | 735 | −R$ 636 000 |
| Regressão Logística | 148 | 80 | 117 | 655 | **+R$ 126 880** |
| Meta (MLP) | 159 | 90 | 106 | 644 | **+R$ 140 190** |

> O modelo logístico já representa uma **economia estimada de R$ 762 880** vs. não fazer nada em 1 000 clientes avaliados.

---

### 3.3 Taxa de Churn Evitado

**Fórmula:** `Taxa_Churn_Evitado = (TP × r_ret) / Total_Churners_Reais`

**Interpretação:** Percentual dos churners reais que foram efetivamente retidos graças ao modelo.

**Meta:** ≥ 24% de redução líquida da taxa de churn por campanha.

---

### 3.4 Top-Decile Lift

**Fórmula:** `Lift = (Churners no top 10% de risco / 10%) / Taxa_Churn_Global`

**Interpretação:** Quantas vezes mais churners concentramos no grupo de maior risco vs. uma seleção aleatória.

**Meta:** ≥ 2,5× — validado como meta razoável para modelos lineares no IBM Telco (PMC 2023).

---

### 3.5 ROI da Campanha de Retenção

**Fórmula:** `ROI = (TP × CLV × r_ret) / (Total_Abordados × C_ret)`

**Meta:** `ROI ≥ 3:1` — para cada R$ 1 investido em oferta de retenção, recuperar ao menos R$ 3 em receita.

---

## 4. Trade-off Falso Positivo vs. Falso Negativo

### 4.1 Análise de Sensibilidade do Threshold

O threshold padrão de 0.50 não é necessariamente ótimo. A curva Precision-Recall permite escolher o ponto que maximiza o Net Savings:

```
Threshold baixo (ex: 0.30):
  + Recall alto: poucos churners escapam
  − Precision baixa: muitos FP, custo de retenção alto
  → Adequado quando C_ret é baixo e CLV é alto

Threshold alto (ex: 0.70):
  + Precision alta: esforço concentrado em casos mais certos
  − Recall baixo: muitos churners não detectados
  → Adequado quando orçamento de retenção é limitado

Threshold ótimo:
  = argmax_t [ TP(t) × 910 − FP(t) × 50 ]
```

**Threshold recomendado para este projeto:** 0.35–0.40 (baseado na razão de custo 48:1).

### 4.2 Recomendação de Threshold por Cenário

| Cenário | Threshold | Justificativa |
|---|---|---|
| **Campanha agressiva** (budget alto) | 0.30 | Maximiza cobertura de churners |
| **Campanha balanceada** (padrão) | 0.40 | Equilíbrio custo-cobertura |
| **Campanha restrita** (budget limitado) | 0.50–0.60 | Foco em casos de alto risco |

---

## 5. Métricas de Monitoramento em Produção (SLOs)

| Métrica | SLO | Frequência | Ação se violado |
|---|---|---|---|
| **AUC-ROC** | ≥ 0.840 | Semanal | Alerta + investigação |
| **PSI (Population Stability Index)** | < 0.20 | Semanal | Retreinamento urgente se PSI > 0.25 |
| **Taxa de Churn Evitado** | ≥ 20% | Mensal | Revisão de threshold ou retreinamento |
| **Latência da API** | < 200 ms (p95) | Contínuo | Escalonamento de infra |
| **Taxa de Erro da API** | < 0.1% | Contínuo | Alerta imediato |
| **Data Drift (KS test)** | p-value < 0.05 em ≤ 3 features | Semanal | Análise de causa raiz |

---

## 6. Referências

1. Mirabdolbaghi, S. et al. (2022). *Model Optimization Analysis of Customer Churn Prediction*. Discrete Dynamics in Nature and Society. DOI: 10.1155/2022/5134356
2. Smets, A. et al. (2026). *Why ROC-AUC Is Misleading for Highly Imbalanced Data*. MDPI Technologies, 14(1), 54.
3. Castanyer, R. et al. (2026). *e-Profits: A Business-Aligned Evaluation Metric for Churn Prediction*. arXiv:2507.08860.
4. MDPI Information. (2025). *A Comprehensive Evaluation of ML and DL Models for Churn Prediction*. Vol. 16(7), 537.
5. Alboukaey, N. et al. (2023). *Customer retention and churn prediction in telecom*. Journal of Big Data / PMC.

---

*Documento gerado como parte da Etapa 1 do Tech Challenge — POSTECH MLOps.*  
*As métricas e metas aqui definidas são os critérios de avaliação para todas as etapas subsequentes.*