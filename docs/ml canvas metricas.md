# ML Canvas — Métricas Técnicas e de Negócio
## Predição de Churn · Telco Customer Dataset

> **Documento:** Fundamentação quantitativa para preenchimento do ML Canvas  
> **Dataset:** Telco Customer Churn (IBM) · 7.043 clientes · 33 variáveis  
> **Proxy de probabilidade utilizado:** `Churn Score / 100` (score real do dataset, correlação 0.66 com churn)  
> **Data de referência:** Análise exploratória — Etapa 1 do Tech Challenge

---

## 1. Contexto de Negócio

### 1.1 Situação atual da empresa

| Métrica | Valor |
|---|---|
| Total de clientes na base | 7.043 |
| Clientes ativos | 5.174 (73,5%) |
| Clientes que saíram (churned) | 1.869 (26,5%) |
| MRR atual (clientes ativos) | US$ 316.985,75 / mês |
| MRR perdido (churned) | US$ 139.130,85 / mês |
| Receita histórica total (Σ Total Charges) | US$ 16.056.168,70 |
| Receita futura destruída pelo churn | US$ 5.426.335,00 |
| CLTV total destruído | US$ 7.755.256,00 |

### 1.2 Cenário de equilíbrio (plateau)

A análise parte de um cenário em que a base de clientes atingiu equilíbrio: o número de clientes que entram mensalmente é igual ao número que sai.

- **Churners/mês em equilíbrio:** ~114 clientes (2,21% da base ativa)
- **MRR em risco todo mês:** US$ 6.984,22
- **Objetivo do modelo:** identificar, dentro desses 114 clientes em risco, quais são convencíveis — especialmente aqueles na faixa de probabilidade 50%–75% — e direcionar ações de retenção antes que saiam.

---

## 2. Definição dos Erros de Predição e seus Custos Reais

A análise de trade-off foi calculada **com os dados individuais de cada cliente**, usando `receita_futura = CLTV − Total Charges` (com rolling 12 meses para clientes que já ultrapassaram o CLTV estimado).

### 2.1 Falso Negativo (FN) — "modelo disse que ficaria, mas o cliente saiu"

```
Situação:  modelo prediz classe 0 (não-churn)
Realidade: cliente cancela o contrato

Consequência: empresa NÃO dispara ação de retenção
              cliente sai levando toda a receita futura estimada
```

| Estatística | Valor |
|---|---|
| Custo médio por FN | **US$ 2.903,34** |
| Custo mediano por FN | US$ 2.845,25 |
| Custo total (todos os 1.869 FN) | US$ 5.426.335,00 |
| Base de cálculo | `CLTV − Total Charges` por cliente churned |

### 2.2 Falso Positivo (FP) — "modelo disse que sairia, mas o cliente ficou"

```
Situação:  modelo prediz classe 1 (churn)
Realidade: cliente permanece ativo

Consequência: empresa dispara ação de retenção desnecessariamente
              gasta recurso em cliente que já ficaria
```

| Estatística | Valor |
|---|---|
| Custo por FP | **US$ 50,00** |
| Base de cálculo | Custo da ação de retenção (desconto 10%, brinde, melhoria de atendimento) |
| Observação | Custo incide apenas sobre clientes que **aceitam** a oferta, não sobre todos os alertas emitidos |

### 2.3 Razão de Custo FN / FP

```
Razão = US$ 2.903,34 / US$ 50,00 = 58,1x
```

**Errar um Falso Negativo custa 58 vezes mais do que errar um Falso Positivo.**

Esta razão é a âncora de toda a definição de SLO: o modelo deve ser calibrado para **minimizar FN mesmo que isso aumente FP**, pois o custo assimétrico justifica uma postura conservadora no threshold de decisão.

---

## 3. Métricas Técnicas do Modelo

### 3.1 Desempenho do proxy real (Churn Score)

O `Churn Score` do dataset (0–100) foi normalizado como probabilidade e usado para gerar as curvas ROC e PR com os rótulos reais de churn.

| Métrica | Valor | Interpretação |
|---|---|---|
| **AUC-ROC** | **0,9417** | Excelente separabilidade — o modelo ranking está muito acima do aleatório (0,50) |
| **PR-AUC** | **0,8737** | Alta precisão mesmo com desbalanceamento (26,5% de positivos) — muito acima do baseline aleatório (0,265) |

> **Nota sobre PR-AUC:** Em datasets desbalanceados, a curva PR é mais informativa que a ROC. Um PR-AUC de 0,87 com 26,5% de positivos indica modelo com forte poder discriminativo real.

### 3.2 Análise por threshold — dados reais

A tabela abaixo foi calculada iterando sobre todos os thresholds de 0,05 a 1,00, aplicando o custo **individual** de FN de cada cliente churned.

| Ponto de Decisão | Threshold | Recall | Precision | F1 | FN | FP | Custo FN | Custo FP | **Custo Total** |
|---|---|---|---|---|---|---|---|---|---|
| **Ótimo financeiro** | **0,64** | **100,0%** | **57,4%** | 0,730 | **0** | 1.385 | **US$ 0** | US$ 69.250 | **US$ 69.250** |
| **SLO Recall ≥ 70%** | **0,64** | **100,0%** | **57,4%** | 0,730 | 0 | 1.385 | US$ 0 | US$ 69.250 | **US$ 69.250** |
| Máximo F1 | 0,71 | 81,0% | 66,6% | **0,731** | 349 | 525 | US$ 1.033.257 | US$ 37.900 | US$ 1.071.157 |
| Default (t = 0,50) | 0,50 | 100,0% | 42,0% | 0,592 | 0 | 2.578 | US$ 0 | US$ 128.900 | US$ 128.900 |

**Economia ao usar threshold ótimo vs default:** US$ 59.650  
**Custo de usar Máximo F1 vs ótimo:** US$ 1.001.907 a mais — 15x pior

### 3.3 Por que maximizar F1 é o pior critério para este negócio

O threshold de máximo F1 (t = 0,71) eleva o custo total em **US$ 1,0M** em relação ao ótimo financeiro. Isso ocorre porque o F1 trata FP e FN com peso igual, ignorando a assimetria de 58:1 nos custos reais. **O F1 é uma métrica técnica válida para comparação entre modelos, mas não deve ser o critério de escolha do threshold de decisão operacional.**

---

## 4. Métricas de Negócio

Apêndice A — Derivação do Threshold Ótimo e do Break-even

Inserir entre a seção 4 e a seção 4.1 do relatório principal.


A.1 O threshold t = 0,64 pertence ao Churn Score do dataset, não a um modelo treinado
O threshold ótimo identificado na análise (t = 0,64) foi calculado sobre o Churn Score fornecido pelo dataset IBM, não sobre as predições de um modelo de ML treinado por nós.
O Churn Score é um oráculo — foi gerado pela IBM com acesso ao rótulo real de churn (Churn Label). Isso explica por que ele consegue separação perfeita:
GrupoRange do Churn ScoreClientes que saíram (churned)65 – 100Clientes que ficaram (ativos)5 – 80
Nenhum churner real tem score abaixo de 65. Logo, qualquer threshold entre 0,64 e 0,65 captura 100% dos churners com zero FN — algo que nenhum modelo treinado em produção conseguirá replicar, pois ele não tem acesso ao futuro.
O que o t = 0,64 representa neste relatório:
É o teto teórico de desempenho — o melhor resultado possível com esse dataset. Serve como referência para avaliar o quão próximo o modelo MLP treinado consegue chegar. Na prática, um modelo real operará com recall menor e algum número de FN, e seu threshold ótimo será encontrado pela mesma lógica: minimizar custo_fn_total + custo_fp_total sobre a curva de probabilidades do modelo.
python# Lógica que se aplica tanto ao Churn Score quanto ao modelo real:
for t in thresholds:
    y_pred         = (y_score >= t).astype(int)
    idx_fn         = (y_true == 1) & (y_pred == 0)
    custo_fn_total = receita_futura[idx_fn].sum()   # soma individual, não média
    custo_fp_total = fp * 50.0
    custo_total    = custo_fn_total + custo_fp_total
# threshold ótimo = t com menor custo_total

A.2 Derivação do break-even: de onde vem "≥ 45 clientes/mês"
O break-even responde à pergunta: quantos clientes precisam ser retidos por mês para que o projeto cubra seus próprios custos?
Passo 1 — margem líquida por cliente retido
Ticket médio dos clientes ativos  =  US$ 61,27 / mês
Custo da ação de retenção         − US$ 50,00
                                    ──────────
Margem líquida por cliente retido =  US$ 11,27 / mês
O custo de US$ 50,00 representa a ação disparada (desconto, brinde, melhoria de atendimento) e incide somente sobre o cliente que aceita a oferta — não sobre todos os alertas emitidos pelo modelo.
Passo 2 — custo fixo mensal do projeto
Infraestrutura + manutenção do ML  =  US$ 500,00 / mês
Passo 3 — equação de break-even
lucro_mensal = retidos × margem − custo_fixo = 0

retidos_mín  = custo_fixo / margem
             = 500,00 / 11,27
             = 44,4 clientes/mês
Verificação com números inteiros:
Clientes retidosCálculoResultado4444 × 11,27 − 500,00− US$ 4,33 ✗ prejuízo4545 × 11,27 − 500,00+ US$ 6,93 ✓ lucro4646 × 11,27 − 500,00+ US$ 18,20 ✓ lucro
O break-even real é 45 clientes/mês. O valor "≥ 44" no relatório principal deve ser lido como o limiar inferior — na prática o primeiro inteiro com lucro positivo é 45.
Sensibilidade do break-even ao custo da ação:
Custo da açãoMargem por retidoBreak-evenUS$ 30 (só atendimento)US$ 31,2716 clientes/mêsUS$ 50 (base)US$ 11,2745 clientes/mêsUS$ 75 (desconto + brinde)− US$ 13,73inviável¹

¹ Se o custo da ação ultrapassar o ticket médio mensal (US$ 61,27), nenhum volume de retenção cobre o custo fixo — o projeto só é viável com ações mais baratas ou ticket maior.


Apêndice gerado na Etapa 1 do Tech Challenge — MLOps · Predição de Churn · POSTECH

### 4.1 Custo de Churn Evitado (métrica principal)

```
Custo de Churn Evitado (por cliente) =
    receita_futura_cliente − custo_ação_retenção

onde:
    receita_futura = CLTV − Total Charges  (gap ≥ 0)
                   = Monthly Charges × 12  (gap < 0, rolling 12m)
    custo_ação    = US$ 50,00 por cliente que aceita a oferta
```

| Estatística | Valor |
|---|---|
| Margem líquida por cliente retido | US$ 14,54 / mês (US$ 774,48 − US$ 50 custos, mensalizado) |
| Break-even mensal do projeto | reter ≥ 44 clientes/mês |
| Com Recall = 70% + Conversão = 50% | ~40 retidos → lucro US$ 80/mês |
| Com Recall = 80% + Conversão = 50% | ~46 retidos → lucro US$ 163/mês |

### 4.2 Ações de retenção e ROI por tipo

| Ação | Custo estimado | Receita salva (12m) | ROI |
|---|---|---|---|
| Desconto de 10% por 12 meses | US$ 77,45 | US$ 774,48 | 900% |
| Serviço extra de brinde (3 meses) | US$ 29,04 | US$ 774,48 | 2.567% |
| Melhoria de plano (6 meses) | US$ 77,45 | US$ 774,48 | 900% |
| Melhoria de atendimento | US$ 15,00 | US$ 774,48 | 5.063% |

> **Premissa:** ROI calculado sobre 12 meses de retenção com ticket médio de US$ 64,54. Conversão não incluída — fator operacional externo ao modelo.

### 4.3 MRR Salvo (métrica de monitoramento)

```
MRR Salvo = clientes_retidos_no_mês × ticket_médio_do_cliente
```

Métrica acompanhada mensalmente para medir o impacto real do modelo em produção. Alerta de degradação: queda de mais de 15% no MRR Salvo em dois meses consecutivos indica drift do modelo.

---

## 5. SLOs — Service Level Objectives

Os SLOs foram derivados matematicamente a partir do break-even financeiro, não arbitrariamente.

### 5.1 SLOs técnicos (modelo)

| Métrica | SLO Mínimo | Justificativa |
|---|---|---|
| **AUC-ROC** | ≥ 0,75 | Separabilidade mínima para ranking confiável de risco |
| **PR-AUC** | ≥ 0,55 | Honesto com desbalanceamento de 26,5% de positivos |
| **Recall** | **≥ 0,70** | Break-even exige reter ≥ 44 clientes/mês; recall baixo inviabiliza o projeto |
| **Precision** | ≥ 0,55 | Evita sobrecarga operacional da equipe de retenção |
| **F1-Score** | ≥ 0,62 | Referência de comparação entre modelos (não para threshold operacional) |

### 5.2 SLO operacional (threshold de decisão)

| Parâmetro | Valor | Justificativa |
|---|---|---|
| **Threshold recomendado** | **0,64** | Minimiza custo total financeiro com os dados reais |
| Threshold default (nunca usar) | 0,50 | Gera US$ 59.650 a mais em custos desnecessários |
| Threshold de máximo F1 (evitar) | 0,71 | Custo US$ 1,0M maior que o ótimo — ignora assimetria FN/FP |

### 5.3 SLO de negócio (monitoramento em produção)

| Métrica | SLO | Frequência |
|---|---|---|
| MRR Salvo | ≥ US$ 3.000 / mês | Mensal |
| Taxa de conversão da campanha | ≥ 40% | Mensal |
| Custo por cliente retido | ≤ US$ 75,00 | Mensal |
| Degradação do AUC-ROC | < 5% em relação ao baseline | Quinzenal |

---

## 6. Segmentação de Risco (faixas do Churn Score)

| Faixa | N clientes | Churn real | Churn rate | Estratégia |
|---|---|---|---|---|
| Score < 25% | 506 (7,2%) | 0 | 0,0% | Nenhuma ação — clientes seguros |
| Score 25–50% | 2.090 (29,7%) | 0 | 0,0% | Monitoramento passivo |
| **Score 50–75%** | **2.696 (38,3%)** | **556** | **20,6%** | **Alvo principal — ações de retenção proativas** |
| Score > 75% | 1.751 (24,9%) | 1.313 | 75,0% | Ação imediata — alto risco |

O segmento **50–75%** é o mais estratégico: concentra 38% da base, tem churn real de 20,6%, mas ainda tem 79,4% de probabilidade de ficar — são os clientes mais convencíveis. O MRR deste segmento é de **US$ 174.000/mês**.

---

## 7. Resumo Executivo para o ML Canvas

### Problema
Empresa de telecomunicações perde 26,5% dos clientes ao ano, destruindo US$ 5,4M em receita futura e US$ 7,7M em CLTV. Em cenário de equilíbrio (plateau), ~114 clientes saem por mês sem que a empresa possa agir preventivamente.

### Solução
Modelo de predição de churn que gera score de risco individual para cada cliente ativo, permitindo direcionar ações de retenção (desconto, melhoria de plano, atendimento especial) para os clientes com probabilidade entre 50–75% — os convencíveis — antes que cancelem.

### Métrica técnica primária
**Recall ≥ 0,70** — derivado do break-even financeiro. Errar um FN custa 58x mais que errar um FP.

### Métrica de negócio primária
**Custo de Churn Evitado** = receita futura retida − custo da ação de retenção. Break-even em 44 clientes retidos/mês.

### Threshold operacional
**0,64** — minimiza o custo total financeiro calculado com os dados reais do dataset. Não usar threshold padrão de 0,50 nem maximizar F1.

### Proxy de risco (feature de alta importância)
`Churn Score` (AUC-ROC = 0,94, PR-AUC = 0,87) — usado como referência de desempenho máximo atingível. O modelo MLP deverá ser comparado contra este baseline.

---

## 8. Referências dos Cálculos

Todos os valores deste documento foram calculados com o dataset real `Telco_customer_churn.xlsx` (IBM, n = 7.043).

```python
# Receita futura por cliente
df['gap']            = df['CLTV'] - df['Total Charges']
df['receita_futura'] = np.where(df['gap'] >= 0, df['gap'], df['Monthly Charges'] * 12)

# Custo de FN por cliente (individual, não média)
custo_fn_individual = df['receita_futura'].values

# Custo total por threshold
for t in thresholds:
    y_pred         = (y_score >= t).astype(int)
    idx_fn         = (y_true == 1) & (y_pred == 0)
    custo_fn_total = custo_fn_individual[idx_fn].sum()   # soma real, não estimada
    custo_fp_total = fp * 50.0
    custo_total    = custo_fn_total + custo_fp_total
```

> O gráfico de trade-off completo está em `outputs/tradeoff_provado_dados_reais.png`

---

*Documento gerado na Etapa 1 do Tech Challenge — MLOps · Predição de Churn · POSTECH*