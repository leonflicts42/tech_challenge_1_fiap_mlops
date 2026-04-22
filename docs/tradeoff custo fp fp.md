# Trade-off Falso Negativo × Falso Positivo
## Fundamentação para o ML Canvas — Predição de Churn · Telco

> **Dataset:** Telco Customer Churn (IBM) · 7.043 clientes · 1.869 churners (26,5%)
> **Escopo:** análise restrita aos clientes que saíram — são esses que o modelo precisa aprender a identificar.
> **Todos os valores calculados com dados individuais reais**, sem premissas externas.

---

## 1. Definição dos Custos

### 1.1 Custo do Falso Negativo (FN)

> **Situação:** modelo classifica o cliente como "vai ficar" → empresa não age → cliente cancela.

O custo de um FN é a **receita futura perdida** por não ter agido a tempo.
Medido pelo **CLTV residual** de cada cliente churned:

```
custo_FN = max(CLTV − Total Charges, 0)
```

O `clip(0)` garante que clientes que já ultrapassaram o CLTV estimado não gerem custo negativo — sua receita futura é tratada como zero (eles já extraíram o valor esperado).

| Estatística | Valor |
|---|---|
| Média | US$ 2.762,91 |
| **Mediana (valor adotado)** | **US$ 2.845,25** |
| P25 | US$ 1.426,45 |
| P75 | US$ 4.165,65 |
| P10 | US$ 0,00 |
| P90 | US$ 5.029,66 |
| Churners com custo FN = 0 | 222 (11,9%) — já ultrapassaram o CLTV |

A mediana é adotada como referência por ser robusta a outliers. O P25 e P75 delimitam o intervalo de confiança do argumento.

---

### 1.2 Custo do Falso Positivo (FP)

> **Situação:** modelo classifica o cliente como "vai sair" → empresa dispara campanha de retenção → cliente ficaria de qualquer forma.

O custo de um FP é o **valor do desconto oferecido** ao cliente que não precisaria dele.
A campanha de retenção consiste em oferecer migração para plano anual com desconto de 10%:

```
custo_FP = ticket_mensal_ativo × 12 × 10%
         = US$ 61,27 × 12 × 10%
         = US$ 73,52
```

A base é o **ticket médio dos clientes ativos** (US$ 61,27/mês) — são eles que recebem a oferta, não os churners já identificados.

| Parâmetro | Valor |
|---|---|
| Ticket mensal médio — ativos | US$ 61,27 |
| Ticket anual médio — ativos | US$ 735,18 |
| **Custo FP — desconto 10% do plano anual** | **US$ 73,52** |
| Incidência | Apenas sobre clientes que **aceitam** a oferta |

> **Nota:** o custo FP não incide sobre todos os alertas emitidos pelo modelo — apenas sobre os clientes que efetivamente aderem à campanha. Alertas ignorados têm custo operacional próximo de zero.

---

## 2. Razão de Custo FN / FP

```
Razão = custo_FN (mediana) / custo_FP
      = US$ 2.845,25 / US$ 73,52
      = 38,7x
```

**Errar um Falso Negativo custa em média 38,7 vezes mais do que errar um Falso Positivo.**

| Referência | Razão FN/FP |
|---|---|
| Mediana geral (valor adotado) | **38,7x** |
| Média geral | 37,6x |
| P25 (caso otimista) | 19,4x |
| P75 (caso pessimista) | 56,7x |

Esta assimetria é a âncora de todas as decisões de threshold e SLO: o modelo deve ser configurado para **minimizar Falsos Negativos**, mesmo que isso aumente o número de Falsos Positivos.

---

## 3. Análise por Faixa de Tenure

A razão FN/FP **não é uniforme** — ela depende de quanto tempo o cliente ainda ficaria. Clientes que saem cedo são os mais caros de perder.

| Faixa de tenure | N | % dos churners | Ticket médio | CLTV resid. mediana | Razão FN/FP | Churners com FN=0 |
|---|---|---|---|---|---|---|
| **0 – 6 meses** | 784 | 41,9% | US$ 63,64 | US$ 3.980,35 | **54,1x** | 0% |
| **7 – 12 meses** | 253 | 13,5% | US$ 75,33 | US$ 3.240,75 | **44,1x** | 0% |
| 13 – 24 meses | 294 | 15,7% | US$ 78,51 | US$ 2.536,85 | 34,5x | 1% |
| 25 – 48 meses | 325 | 17,4% | US$ 84,50 | US$ 771,15 | 10,5x | 30% |
| 49 – 72 meses | 213 | 11,4% | US$ 92,17 | US$ 0,00 | 0,0x¹ | 58% |

> ¹ Mediana zero porque 58% desse grupo já ultrapassou o CLTV. Média ainda é US$ 572,86 → razão de 7,8x.

### Insight estratégico

**55,5% dos churners saíram nos primeiros 12 meses** (grupos 0–6m e 7–12m), com razão FN/FP entre 44x e 54x. São o segmento mais impactante financeiramente e, ao mesmo tempo, o mais desafiador para o modelo — poucos meses de histórico disponível para aprendizado.

Os clientes que ficam mais de 48 meses antes de sair têm razão próxima de zero: já extraíram quase todo o CLTV estimado. Errar a predição desse grupo tem impacto financeiro desprezível.

**Implicação para o modelo:** o sinal mais valioso é a detecção de churn precoce. A feature `Tenure Months` deve ter peso relevante na arquitetura, e o threshold de decisão deve ser especialmente sensível a clientes com baixo tenure e Churn Score elevado.

---

## 4. Impacto Financeiro por Threshold de Decisão

Calculado com o **Churn Score real do dataset** (AUC-ROC = 0,9417 | PR-AUC = 0,8737) como proxy de probabilidade. O custo FN de cada cliente é seu CLTV residual individual — não uma média.

> **Atenção:** o Churn Score é um oráculo gerado pela IBM com acesso ao rótulo real. Representa o **teto teórico** de desempenho. Um modelo treinado em produção operará com métricas menores — mas a mesma lógica de minimização de custo total se aplica.

| Ponto de decisão | Threshold | Recall | Precision | FN | FP | Custo FN | Custo FP | **Custo Total** |
|---|---|---|---|---|---|---|---|---|
| **Ótimo financeiro** | **0,64** | **100,0%** | 57,4% | **0** | 1.385 | **US$ 0** | US$ 101.823 | **US$ 101.823** |
| SLO Recall ≥ 70% | 0,64 | 100,0% | 57,4% | 0 | 1.385 | US$ 0 | US$ 101.823 | US$ 101.823 |
| Máximo F1 | 0,71 | 81,0% | 66,6% | 356 | 758 | US$ 990.362 | US$ 55.727 | **US$ 1.046.089** |
| Default (t = 0,50) | 0,50 | 100,0% | 42,0% | 0 | 2.578 | US$ 0 | US$ 189.530 | US$ 189.530 |

**Economia ao usar threshold ótimo vs default:** US$ 87.707
**Custo de usar Máximo F1 vs ótimo:** US$ 944.266 a mais — 10x pior

### Por que maximizar F1 é incorreto para este problema

O threshold de máximo F1 (t = 0,71) eleva o custo total em quase **US$ 1 milhão** em relação ao ótimo financeiro. Isso acontece porque o F1 trata FP e FN com peso igual — ignora a assimetria de 38,7:1 nos custos reais. O F1 é válido para **comparar modelos entre si**, mas não deve determinar o threshold operacional de negócio.

---

## 5. Resumo para o ML Canvas

### Custos a registrar

| Campo do ML Canvas | Valor |
|---|---|
| **Custo unitário do Falso Negativo** | US$ 2.845,25 (mediana do CLTV residual dos churners) |
| **Custo unitário do Falso Positivo** | US$ 73,52 (10% do ticket anual do cliente ativo) |
| **Razão FN / FP** | **38,7x** (mediana) · intervalo: 19,4x – 56,7x |
| Razão para churners precoces (0–12m) | **44x – 54x** (55,5% dos casos) |

### Decisão de threshold

| Campo do ML Canvas | Valor |
|---|---|
| Threshold operacional recomendado | A ser determinado com o modelo real treinado |
| Critério de seleção do threshold | Minimizar `Σ custo_FN_individual + n_FP × US$ 73,52` |
| Threshold a evitar | Máximo F1 — ignora assimetria de custos |
| Referência teórica (Churn Score IBM) | t = 0,64 → recall 100%, custo total US$ 101.823 |

### Métricas técnicas derivadas

| Métrica | SLO | Justificativa |
|---|---|---|
| **Recall** | **≥ 0,70** | Razão 38,7x favorece minimizar FN; recall baixo destrói valor |
| **AUC-ROC** | ≥ 0,75 | Referência de separabilidade mínima para ranking confiável |
| **PR-AUC** | ≥ 0,55 | Honesto com desbalanceamento de 26,5% de positivos |
| **Precision** | ≥ 0,55 | Evita sobrecarga operacional da equipe de retenção |
| **F1-Score** | ≥ 0,62 | Somente para comparação entre modelos — não para threshold |

---

## 6. Fórmulas de Referência

```python
# Custo FN por cliente churned
custo_fn = max(CLTV - Total_Charges, 0)

# Custo FP por alerta disparado (aceito)
custo_fp = ticket_mensal_ativo * 12 * 0.10  # = US$ 73.52

# Razão de assimetria
razao_fn_fp = custo_fn_mediana / custo_fp  # = 38.7x

# Custo total para um dado threshold t
custo_total(t) = sum(custo_fn[i] for i in FN(t)) + count(FP(t)) * custo_fp

# Threshold ótimo = argmin custo_total(t)
t_otimo = argmin_{t ∈ [0,1]} custo_total(t)
```

---

## 7. Nota Metodológica

O **CLTV residual** foi adotado como métrica de custo FN por ser a estimativa mais abrangente de valor futuro disponível no dataset. Ele captura tanto o potencial de receita quanto a antiguidade do relacionamento. O `clip(0)` é conservador — clientes que já geraram mais receita que o CLTV estimado têm custo FN tratado como zero, sem assumir geração de receita especulativa.

O **custo FP de US$ 73,52** (10% do ticket anual dos ativos) representa uma campanha de retenção razoável e auditável. Ações mais baratas (melhoria de atendimento, US$ 15–20) reduzem o custo FP e aumentam a razão para ~142x–190x, reforçando ainda mais a prioridade de recall. Ações mais caras (desconto + brinde acima de 20% do anual) aumentam o custo FP e reduzem a razão para ~19x–25x.

---

*Documento gerado na Etapa 1 do Tech Challenge — MLOps · Predição de Churn · POSTECH*
*Calculado com dataset real: Telco_customer_churn.xlsx (IBM, n = 7.043)*