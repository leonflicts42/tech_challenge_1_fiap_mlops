# Etapa 1 — Consolidação Final
## Tech Challenge · MLOps · Predição de Churn para Telecomunicações

> **Documento:** Relatório técnico consolidado da Etapa 1 — ML Canvas, métricas, trade-off e plano de deploy
> **Dataset:** Telco Customer Churn (IBM) · 7.043 clientes
> **Destinatários:** Stakeholders de negócio, time técnico e avaliação acadêmica
> **Status:** Fechamento da fundamentação antes do treino dos baselines e da MLP

---

## Sumário Executivo

A empresa perde 26,5% dos clientes ao ano. Em regime permanente de equilíbrio da base — onde o número de novos clientes iguala o número de cancelamentos — isso equivale a aproximadamente **114 clientes que saem por mês** sem qualquer ação preventiva. Esse número é derivado diretamente dos dados: taxa de churn anual observada (26,5%) dividida por 12, aplicada sobre os 5.174 clientes ativos.

A receita futura destruída por esses cancelamentos soma **US$ 5,16M** — o valor que o projeto de ML tenta recuperar parcialmente.

A fundamentação financeira do modelo parte de uma assimetria crítica: errar um cliente que ia sair (**Falso Negativo**) custa, em mediana, **US$ 2.845** em receita futura irrecuperável, enquanto oferecer desconto a um cliente que ficaria de qualquer forma (**Falso Positivo**) custa apenas **US$ 73,52**. Essa diferença de **~30 a 55 vezes** — dependendo do tempo de permanência do cliente — define que o modelo deve priorizar **Recall** mesmo ao custo de mais alertas desnecessários.

Com Recall ≥ 70% e taxa de conversão da campanha de 50%, estima-se recuperação líquida de **US$ 1,76M** sobre o conjunto de churners históricos, ou **US$ 1,29M/ano** em regime permanente — representando **34% da perda anual**.

---

## 1. ML Canvas Consolidado

### 1.1 Background

**End-User (Usuário Final)**

| Ator | Papel |
|---|---|
| Time de Retenção / CRM | Usuário direto — recebe lista priorizada de clientes em risco e executa campanhas |
| Gerente de Customer Success | Monitora KPIs de churn e eficiência das ações de retenção |
| Analista de Dados / MLOps | Opera, monitora e retreina o modelo via MLflow |
| Diretoria Comercial | Stakeholder estratégico — decisões orçamentárias baseadas em previsões |
| Clientes em risco | Afetados indiretamente — recebem abordagem proativa |

**Restrições éticas:** o modelo é ferramenta interna B2B; o cliente nunca é informado de sua classificação; grupos demográficos vulneráveis (ex.: `Senior Citizen = Yes`) devem ter tratamento monitorado como viés potencial.

**Value Proposition**

> *"Antecipar quais clientes irão cancelar o serviço nos próximos 30 dias, permitindo que o time de retenção atue preventivamente antes que o cancelamento ocorra."*

| Objetivo | Métrica de Sucesso |
|---|---|
| Reduzir churn voluntário | Redução ≥ 5 p.p. na taxa de churn mensal após 6 meses |
| Priorizar esforço de retenção | Top-decile lift ≥ 2,5x sobre seleção aleatória |
| Maximizar receita recorrente retida | ROI de campanha ≥ 3:1 |
| Recuperar receita futura | Receita líquida anual recuperada ≥ US$ 1M |

### 1.2 Problem Specification

| Item | Descrição |
|---|---|
| Pergunta a predizer | "Este cliente vai cancelar o contrato nos próximos 30 dias?" |
| Input | 19 features tabulares: demografia, contrato, serviços, cobranças |
| Output | `Churn = 0/1` + probabilidade contínua `P(Churn) ∈ [0, 1]` |
| Tipo de problema | Classificação binária supervisionada · desbalanceamento moderado (26,5%) |

### 1.3 Features utilizadas e descartadas

**Descartadas (14 colunas):**

| Categoria | Colunas | Motivo |
|---|---|---|
| Identificação | `CustomerID`, `Count` | Sem valor preditivo; risco de overfitting a IDs específicos |
| Geográficas | `Country`, `State`, `City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude` | **Evitar viés geográfico** — dataset concentrado em um estado; generalização comprometida |
| Oráculo | `Churn Score` | Score gerado pela IBM com acesso ao rótulo real — causaria data leakage imediato |
| Custo / pós-treinamento | `CLTV` | Estimativa prospectiva calculada com o perfil do cliente — descartada como feature para evitar leakage; **usada apenas para calcular o custo de FN** fora do pipeline de treino |
| Pós-evento | `Churn Reason` | Só existe para clientes que já saíram — não disponível em produção |
| Targets | `Churn Label`, `Churn Value` | São a variável a predizer, não features |

> **Nota sobre CLTV:** a coluna `CLTV` é utilizada exclusivamente na análise de custo (seções 2 e 3) como denominador do custo de FN. Ela não entra no pipeline de treino/validação do modelo, evitando qualquer risco de leakage.

**Mantidas (19 features):**

| Grupo | Features |
|---|---|
| Demográficas (4) | `Gender`, `Senior Citizen`, `Partner`, `Dependents` |
| Contrato/Billing (4) | `Contract`, `Paperless Billing`, `Payment Method`, `Monthly Charges` |
| Relacionamento (2) | `Tenure Months`, `Total Charges` |
| Serviços telefônicos (2) | `Phone Service`, `Multiple Lines` |
| Internet e add-ons (7) | `Internet Service`, `Online Security`, `Online Backup`, `Device Protection`, `Tech Support`, `Streaming TV`, `Streaming Movies` |

### 1.4 Data Sources

| Fonte | Uso | Disponibilidade |
|---|---|---|
| CRM Interno | Dados contratuais: tenure, tipo de contrato, serviços ativos | Produção |
| Sistema de Billing | Cobranças mensais, forma de pagamento | Produção |
| Dataset IBM Telco | Desenvolvimento e treinamento inicial (7.043 clientes) | Kaggle / IBM |
| Sistema de Atendimento | NPS, chamados (feature futura) | Produção (futuro) |

---

## 2. Contexto Financeiro da Empresa

### 2.1 Estado atual

| Métrica | Valor |
|---|---|
| Total de clientes | 7.043 |
| Ativos | 5.174 (73,5%) |
| Churned | 1.869 (26,5%) |
| MRR atual | US$ 316.985,75 / mês |
| MRR perdido pelos churned | US$ 139.130,85 / mês |
| Receita histórica total (Σ Total Charges) | US$ 16.056.168,70 |
| **Receita futura destruída pelo churn** | **US$ 5.163.882,40** |
| CLTV total dos churned | US$ 7.755.256,00 |

### 2.2 Por que a receita destruída diverge do CLTV total

A leitura simultânea de US$ 5,16M e US$ 7,75M pode causar confusão — são métricas que medem coisas diferentes, e entender a diferença é essencial para interpretar corretamente o impacto do churn.

| Métrica | O que representa | Valor |
|---|---|---|
| **CLTV total dos churned** | Valor vitalício total estimado do cliente, do primeiro ao último mês possível de contrato | US$ 7.755.256 |
| **Total Charges já pagos** | Receita que esses clientes efetivamente pagaram antes de cancelar | US$ 2.862.927 |
| **Receita futura destruída** | Parcela ainda não realizada do CLTV no momento do cancelamento | US$ 5.163.882 |

```
CLTV total (US$ 7,75M)
    = Receita já capturada (US$ 2,86M)   ← está no caixa, não é perda
    + Receita futura destruída (US$ 5,16M)  ← esta sim é a perda real
    − Ajuste técnico (US$ 0,27M)          ← clientes que já geraram mais
                                             receita do que o CLTV estimado
```

**Em palavras diretas:** o CLTV é uma estimativa do valor total que o cliente geraria ao longo de toda a sua vida como cliente. Quando ele cancela, a empresa já recebeu parte desse valor (Total Charges). O que se perde é apenas a **parte ainda não realizada** — a receita que ele geraria nos meses e anos seguintes, caso permanecesse.

O CLTV total (US$ 7,75M) não pode ser tratado como perda porque inclui receita que **já está no caixa**. O número correto para dimensionar o problema que o modelo de ML deve resolver é **US$ 5,16M** — o valor contratado mas não realizado.

---

## 3. Trade-off Falso Negativo × Falso Positivo

### 3.1 Definição dos erros

**Falso Negativo (FN):** modelo classifica o cliente como "vai ficar" → empresa não age → cliente cancela → receita futura irrecuperável.

**Falso Positivo (FP):** modelo classifica o cliente como "vai sair" → empresa dispara campanha de desconto → cliente ficaria de qualquer forma → custo do desconto desperdiçado.

### 3.2 Custo do Falso Negativo

```
custo_FN = max(CLTV − Total Charges, 0)
```

Calculado individualmente para cada um dos 1.869 churners — não como média global.

| Estatística | Valor |
|---|---|
| **Mediana (referência principal)** | **US$ 2.845,25** |
| Média | US$ 2.762,91 |
| P25 | US$ 1.426,45 |
| P75 | US$ 4.165,65 |
| Churners com custo FN = 0 | 222 (11,9%) — já ultrapassaram o CLTV estimado |

### 3.3 Custo do Falso Positivo

A campanha de retenção consiste em oferecer migração para plano anual com desconto de 10%:

```
custo_FP = ticket_mensal_ativo × 12 × 10%
         = US$ 61,27 × 12 × 10%
         = US$ 73,52
```

Incide **apenas sobre os clientes que aceitam** a oferta. Alertas ignorados têm custo operacional próximo de zero.

### 3.4 Razão de assimetria e sua dependência do tempo de permanência

A razão FN/FP não é um valor fixo — ela varia conforme o tempo que o cliente já permaneceu na empresa, porque clientes que cancelam cedo ainda têm muito CLTV residual pela frente, enquanto clientes que ficam anos antes de cancelar já realizaram quase todo o valor esperado.

| Faixa de tenure | N | % churners | CLTV resid. mediana | Razão FN/FP |
|---|---|---|---|---|
| **0 – 6 meses** | 784 | 41,9% | US$ 3.980 | **54,1x** |
| **7 – 12 meses** | 253 | 13,5% | US$ 3.241 | **44,1x** |
| 13 – 24 meses | 294 | 15,7% | US$ 2.537 | 34,5x |
| 25 – 48 meses | 325 | 17,4% | US$ 771 | 10,5x |
| 49 – 72 meses | 213 | 11,4% | US$ 0 | 0,0x¹ |
| **Geral (mediana)** | **1.869** | **100%** | **US$ 2.845** | **38,7x** |

> ¹ Mediana zero porque 58% desse grupo já ultrapassou o CLTV estimado. Média ainda é US$ 573 (razão de 7,8x).

O **range operacional relevante é de 30x a 54x**: abaixo de 30x estão clientes que já ficaram tempo suficiente para extrair grande parte do CLTV (baixo impacto de FN); acima de 54x estão os churners mais precoces (altíssimo impacto). O modelo deve ser otimizado para capturar prioritariamente os clientes nas faixas de menor tenure, que concentram 55,5% dos churners e as maiores razões de custo.

Essa variação de 30x a 54x reforça a mesma conclusão em qualquer ponto do intervalo: **o custo de não detectar um churner é sempre dominante sobre o custo de uma campanha desnecessária**, e o modelo deve ser configurado para minimizar Falsos Negativos.

---

## 4. Métricas Técnicas e Custo de Churn Evitado

### 4.1 Hierarquia de métricas

| Métrica | Papel | SLO mínimo |
|---|---|---|
| **AUC-ROC** | **Técnica primária** — ranking de risco, comparação entre modelos | **≥ 0,85** |
| **Recall (Churn)** | **Operacional primária** — cobertura dos churners detectados | **≥ 0,70** |
| **PR-AUC** | Técnica complementar — desempenho na classe minoritária | ≥ 0,65 |
| **F1-Score (Churn)** | Comparativa — equilíbrio entre Precision e Recall | ≥ 0,62 |
| **Precision (Churn)** | Controle operacional — evita sobrecarga da campanha | ≥ 0,55 |

### 4.2 Justificativa técnica

**AUC-ROC como métrica técnica primária:** mede a qualidade do ranking de risco independentemente do threshold escolhido. É a métrica que define se o modelo aprendeu a separar churners de não-churners — e é a referência para comparar MLP contra baselines. Uma AUC-ROC alta garante que, independentemente de onde o threshold for posicionado, o modelo entrega informação útil.

**Recall como métrica operacional primária:** com razão FN/FP entre 30x e 54x, cada ponto percentual de recall a mais representa um churner capturado (US$ 2.845 salvos) ao custo marginal de alguns alertas extras (US$ 73,52 cada). A conta sempre favorece aumentar recall — o modelo deve ser calibrado para esse objetivo após a otimização do AUC-ROC.

**F1 como métrica comparativa, não operacional:** o F1 trata FP e FN com peso igual, ignorando a assimetria de custo. É útil para comparar modelos entre si, mas **não deve determinar o threshold de decisão operacional**. O threshold será determinado pelo critério de custo total mínimo (seção 5.3, Etapa 2).

### 4.3 Comparativo de cenários: Dummy, Sem Modelo e MLP

O ponto ótimo do projeto não é eliminar 100% do churn (inviável e desnecessário) nem não fazer nada (perda total). É encontrar o equilíbrio financeiro entre o valor recuperado pelos clientes retidos e o custo das campanhas disparadas.

A tabela abaixo apresenta quatro cenários distintos sobre os **1.869 churners históricos**, com taxa de conversão da campanha fixada em 50%:

| Cenário | Clientes abordados | Retidos | Receita recuperada | Custo campanhas | **Receita líquida** |
|---|---|---|---|---|---|
| **Sem modelo** (status quo) | 0 | 0 | US$ 0 | US$ 0 | **−US$ 5.163.882** |
| **Dummy** (aborda todos os ativos) | 5.174 | 1.869² | US$ 5.163.882 | US$ 380.383 | **US$ 4.783.499** |
| **Modelo — Recall 70%** | ~1.308 | 654 | US$ 1.807.000 | US$ 48.000 | **US$ 1.759.000** |
| **Modelo — Recall 90%** | ~1.682 | 841 | US$ 2.324.000 | US$ 62.000 | **US$ 2.262.000** |

> ² O Dummy assume conversão de 100% para os churners (evita todo o churn) mais conversão de 50% para os não-churners que recebem campanha.

**O que a tabela revela:**

O cenário Dummy tem o melhor resultado bruto porque aborda toda a base e não perde nenhum churner. No entanto, ele gasta **US$ 243.000 em campanhas para os 5.174 clientes ativos que ficariam de qualquer forma** — um desperdício que o modelo elimina ao focar apenas nos clientes em risco. O modelo com Recall 70% recupera 34% do valor destruído com apenas 13% do custo de campanha do Dummy.

O **ponto de equilíbrio financeiro** do projeto está onde o ganho marginal de aumentar o recall supera o custo marginal das campanhas adicionais. Como cada cliente retido gera **US$ 2.689 líquidos** (CLTV residual médio de US$ 2.763 menos o desconto de US$ 73,52), e cada alerta adicional custa no máximo US$ 73,52, a matemática sempre favorece aumentar o recall — o verdadeiro limitante é a **capacidade operacional** da equipe de retenção de processar os alertas, não o custo financeiro das campanhas.

### 4.4 Benchmarks de referência (literatura IBM Telco)

| Modelo | AUC-ROC esperado | Recall esperado | F1 esperado |
|---|---|---|---|
| DummyClassifier (piso) | 0,50 | 0,00 | 0,00 |
| Regressão Logística | ~0,84 | ~0,55 | ~0,60 |
| **Meta MLP PyTorch** | **≥ 0,85** | **≥ 0,70** | **≥ 0,62** |

### 4.5 Custo de Churn Evitado — relação entre investimento e retorno

**O custo de churn evitado** mede, em dólares, o ganho líquido de cada cliente retido pela campanha:

```
Custo de Churn Evitado (por cliente) =
    CLTV_residual_cliente − custo_desconto_10%
  = US$ 2.762,91 − US$ 73,52
  = US$ 2.689,39 por cliente efetivamente retido
```

Para cada **US$ 1 investido** em desconto, a empresa recupera em média **US$ 37,60** em receita futura. Isso é possível porque o desconto representa apenas 2,7% do valor que o cliente ainda geraria.

A tabela abaixo quantifica a recuperação para diferentes níveis de recall, **sobre os 1.869 churners do dataset histórico**, com conversão de 50%:

| Recall | Churners detectados | Retidos | Receita bruta | Custo descontos | **Receita líquida** | ROI da campanha |
|---|---|---|---|---|---|---|
| 50% | 935 | 467 | US$ 1.290.000 | US$ 34.000 | **US$ 1.256.000** | 36,9:1 |
| **70%** | **1.308** | **654** | **US$ 1.807.000** | **US$ 48.000** | **US$ 1.759.000** | **36,6:1** |
| 80% | 1.495 | 748 | US$ 2.066.000 | US$ 55.000 | **US$ 2.011.000** | 36,6:1 |
| 90% | 1.682 | 841 | US$ 2.324.000 | US$ 62.000 | **US$ 2.262.000** | 36,5:1 |

**Leitura prática:** o ROI da campanha é virtualmente constante em ~36,6:1 em qualquer nível de recall porque o custo do desconto (US$ 73,52) é sempre uma fração mínima do CLTV residual recuperado (US$ 2.763). Isso significa que **não existe recall alto demais do ponto de vista financeiro** — o único limitante real é quantos clientes a equipe de retenção consegue atender por mês.

### 4.6 Cenário alvo em regime permanente (Recall ≥ 70% · Conversão 50%)

Projetando sobre o fluxo anual esperado de churners (1.371/ano em regime permanente):

| Indicador | Valor |
|---|---|
| Churners esperados/ano | 1.371 |
| Detectados pelo modelo | 960 |
| Efetivamente retidos | 480 |
| Receita bruta recuperada | US$ 1.326.000 |
| Custo dos descontos concedidos | US$ 35.000 |
| **Receita líquida anual recuperada** | **US$ 1.291.000** |
| **% da perda anual recuperada** | **34,1%** |

---

## 5. Threshold e Otimização do Modelo

### 5.1 Seleção do threshold (Etapa 2)

O threshold operacional será determinado após o treinamento do modelo real (Etapa 2), aplicando a lógica de minimização de custo total sobre a curva de probabilidades do MLP:

```python
threshold_otimo = argmin_{t ∈ [0,1]}  Σ custo_FN_individual(t) + n_FP(t) × 73.52

onde:
    custo_FN_individual(i) = max(CLTV_i − Total_Charges_i, 0)
```

**Processo de escolha (Etapa 2):**
1. Treinar MLP e obter curva de probabilidades no conjunto de validação
2. Para cada threshold candidato: calcular `custo_total(t)` com CLTV residual individual
3. Filtrar candidatos que atendam Recall ≥ 0,70
4. Dentro do subconjunto filtrado, escolher o de menor `custo_total`
5. Validar escolha em teste cego (holdout estratificado, seed = 42)

**Thresholds a evitar:**
- **t = 0,50 (default):** ignora a assimetria de custo — opera como se FN e FP tivessem mesmo peso
- **t = argmax F1:** otimiza equilíbrio aritmético entre Precision e Recall, mas ignora a razão de 30–54x do custo real

### 5.2 Aplicação nas próximas etapas

Este processo de seleção de threshold será registrado no MLflow como experimento separado, com os parâmetros `custo_fp`, `razao_fn_fp_mediana` e `threshold_selecionado` versionados junto ao modelo.

### 5.3 Uso na otimização da MLP (Etapa 2)

A curva de custo total em função do threshold funciona também como **função de perda de negócio** para guiar a otimização da MLP. Em vez de otimizar apenas por cross-entropy, a validação do modelo em cada época pode incluir o custo total no threshold ótimo como métrica de acompanhamento — permitindo identificar se o modelo está melhorando em termos de custo financeiro real, não apenas em métricas técnicas padrão.

---

## 6. Model Card

### 6.1 Identificação

| Campo | Valor |
|---|---|
| Nome | `telco_churn_predictor` |
| Versão | 1.0.0 (a ser emitida após Etapa 2) |
| Tipo | Classificador binário supervisionado |
| Arquitetura | MLP (Multilayer Perceptron) com PyTorch — baselines: Dummy, Logistic Regression |
| Responsáveis | Equipe MLOps do Tech Challenge |

### 6.2 Uso pretendido

- **Uso primário:** gerar score de risco diário para clientes ativos, priorizando abordagens de retenção
- **Usuários primários:** time de Retenção / CRM, Customer Success
- **Fora de escopo:** decisões automatizadas sem revisão humana; comunicação direta ao cliente; segmentação de marketing; segmentos B2B ou enterprise; regiões geográficas não representadas no treino

### 6.3 Performance esperada

| Métrica | Meta mínima | Meta objetivo |
|---|---|---|
| AUC-ROC | 0,80 | **0,85** |
| PR-AUC | 0,60 | 0,70 |
| Recall (Churn) | **0,70** | 0,75 |
| Precision (Churn) | 0,55 | 0,60 |
| F1-Score (Churn) | 0,62 | 0,65 |

### 6.4 Dados de treinamento

- **Fonte:** IBM Telco Customer Churn (7.043 clientes)
- **Balanceamento:** 73,5% não-churn · 26,5% churn
- **Divisão:** 80/20 estratificado (StratifiedShuffleSplit, seed = 42)
- **Validação cruzada:** StratifiedKFold 5-fold (shuffle = True, seed = 42)
- **Features:** 19 (sem identificadores, geográficas, CLTV e Churn Score)
- **Tratamento de missing:** 11 valores em `Total Charges` imputados pela mediana

### 6.5 Limitações conhecidas

- **Snapshot estático:** sem série temporal — não captura sazonalidade nem drift evolutivo
- **Concentração regional:** dados de um único estado — generalização não garantida; features geográficas removidas preventivamente
- **Viés de produto:** apenas telecom residencial — não aplicável a B2B ou enterprise
- **Churners precoces subrepresentados em sinal:** 42% dos churners têm tenure < 6 meses e pouco histórico de comportamento — são os mais caros de perder e os mais difíceis de detectar
- **CLTV como referência externa:** usada apenas para quantificação de custo, não como feature — limitações do CLTV estimado afetam a precisão do custo calculado, não o modelo em si

### 6.6 Vieses potenciais monitorados

| Dimensão | Variável | O que monitorar |
|---|---|---|
| Idade | `Senior Citizen` | Recall diferencial entre grupos |
| Gênero | `Gender` | Taxa de FP por grupo |
| Estrutura familiar | `Partner`, `Dependents` | Disparidade de tratamento na campanha |
| Tipo de internet | `Internet Service` | Clientes Fiber optic têm churn 2x maior — validar se o modelo amplifica o viés |
| Forma de pagamento | `Payment Method` | Electronic check tem churn 3x maior — investigar causalidade vs correlação |

### 6.7 Cenários de falha

**Falha silenciosa — modelo degradado sem alerta:**
- Causa provável: drift nos dados de entrada
- Detecção: PSI > 0,20 em `Tenure Months`, `Monthly Charges` ou `Contract`
- Mitigação: retreinamento emergencial; fallback para regra de negócio manual (`Contract = Month-to-month` + `Tenure < 12`)

**Falha de viés — grupo demográfico com Recall significativamente menor:**
- Causa provável: subamostragem no treino
- Detecção: diferença de Recall > 10 p.p. entre grupos sensíveis
- Mitigação: rebalanceamento com pesos amostrais; revisão de features

**Falha de dependência — feature essencial indisponível:**
- Causa provável: indisponibilidade do sistema de billing
- Detecção: schema validation falha no pipeline de ingestão
- Mitigação: fallback para regra de negócio; alerta imediato para o time de MLOps

**Falha por mudança de premissa — política de desconto alterada:**
- Causa provável: negócio muda o valor ou formato da campanha
- Detecção: revisão trimestral dos parâmetros financeiros
- Mitigação: recalcular `custo_fp` e threshold ótimo; nova versão do modelo registrada no MLflow

---

## 7. Arquitetura de Deploy

### 7.1 Modalidade escolhida: Batch diário como primária, Real-time como secundária

**Justificativa:** churn é um fenômeno de semanas a meses — a decisão de cancelamento não acontece em milissegundos. O time de retenção opera com uma lista priorizada diária, não com alertas instantâneos. Quatro fatores definem a escolha:

- **Alinhamento com o negócio:** a lista de risco é consultada uma vez ao dia pela equipe de retenção
- **Volume gerenciável:** ~5.000 clientes processados em janela overnight sem pressão de latência
- **Consistência:** mesma versão do modelo aplicada a todos no mesmo ciclo
- **Auditabilidade:** output diário versionado em parquet facilita análise retroativa e validação de impacto

O endpoint real-time é mantido como modo secundário para casos específicos: atendimento reativo quando o cliente entra em contato, pré-venda de upgrade de plano, análise pontual sob demanda.

### 7.2 Arquitetura batch

```
┌───────────────┐    02:00 UTC    ┌───────────────┐    ┌──────────────┐
│  Data Lake    │ ───────────────►│  Batch Job    │───►│   MLflow     │
│  (CRM/Bill)   │   daily sync    │  (Docker)     │    │  Registry    │
└───────────────┘                 └───────┬───────┘    └──────────────┘
                                          │
                                          ▼
                                  ┌───────────────┐
                                  │ predictions   │
                                  │ {date}.parquet│
                                  └───────┬───────┘
                                          │
                                          ▼
                                  ┌───────────────┐
                                  │  CRM/Salesforce│
                                  │  (via API)    │
                                  └───────────────┘
```

**SLOs batch:** execução concluída antes das 04:00 UTC · 95% das execuções < 2h · disponibilidade ≥ 99,5%/mês

### 7.3 Arquitetura real-time

```
┌──────────────┐   HTTPS   ┌──────────────┐    ┌──────────────┐
│   Consumer   │──────────►│   FastAPI    │───►│  Model.pkl   │
│  (CRM/App)   │ /predict  │  (uvicorn)   │    │  (em memória)│
└──────────────┘           └──────┬───────┘    └──────────────┘
                                  │
                                  ▼
                          ┌───────────────┐
                          │  Structured   │
                          │    Logging    │
                          └───────────────┘
```

**SLOs real-time:** latência p95 < 200 ms · latência p99 < 500 ms · taxa de erro < 0,1% · disponibilidade 99,9%

### 7.4 Endpoints da API

| Endpoint | Método | Propósito |
|---|---|---|
| `/predict` | POST | Predição individual com validação Pydantic |
| `/predict/batch` | POST | Lote de até 1.000 clientes por chamada |
| `/health` | GET | Verifica se a API e o modelo estão operacionais |
| `/metrics` | GET | Métricas Prometheus (latência, contadores de predição) |
| `/version` | GET | Versão do modelo em produção |

---

## 8. Plano de Monitoramento

### 8.1 SLOs técnicos (modelo)

| Métrica | SLO | Frequência | Ação se violado |
|---|---|---|---|
| AUC-ROC | ≥ 0,80 | Semanal | Alerta + investigação em 48h |
| Recall (Churn) | ≥ 0,65 | Semanal | Revisão de threshold |
| PSI (Population Stability Index) | < 0,20 | Semanal | Retreinamento emergencial se > 0,25 |

### 8.2 SLOs de infraestrutura

| Métrica | SLO | Frequência | Ação se violado |
|---|---|---|---|
| Latência API (p95) | < 200 ms | Contínuo | Scale-up automático |
| Taxa de erro API | < 0,1% | Contínuo | Alerta imediato |
| Job batch concluído | antes das 04:00 UTC | Diário | Investigação no dia seguinte |

### 8.3 SLOs de negócio

| Métrica | SLO | Frequência |
|---|---|---|
| Receita líquida recuperada | ≥ US$ 90.000 / mês | Mensal |
| Taxa de conversão da campanha | ≥ 40% | Mensal |
| ROI da campanha de retenção | ≥ 5:1 | Trimestral |

### 8.4 Playbook de resposta

**AUC-ROC abaixo de 0,80 por 2 semanas consecutivas:**
1. Executar avaliação offline com dados dos últimos 60 dias
2. Calcular PSI por feature crítica (`Tenure Months`, `Monthly Charges`, `Contract`)
3. PSI > 0,25 em ≥ 3 features → retreinamento emergencial
4. Caso contrário, investigar possível mudança no conceito de churn

**Receita recuperada abaixo de US$ 60.000 por 2 meses consecutivos:**
1. Verificar se a equipe de retenção está operando as listas diárias
2. Auditar taxa de conversão por tipo de ação
3. Revisar threshold — pode estar restringindo recall desnecessariamente
4. Se necessário, retreinar com dados mais recentes

---

## 9. Próximos Passos (Etapas 2–4)

**Etapa 2 — Modelagem com Redes Neurais**
- [ ] Treinar DummyClassifier e Regressão Logística como baselines; registrar no MLflow
- [ ] Implementar MLP em PyTorch com early stopping e batching
- [ ] Comparar MLP × baselines usando as 5 métricas desta Etapa 1
- [ ] Determinar threshold operacional pelo critério de custo total mínimo (seção 5.1)
- [ ] Usar custo total no threshold ótimo como métrica de acompanhamento por época (seção 5.3)

**Etapa 3 — Engenharia e API**
- [ ] Refatorar em estrutura modular (`src/`)
- [ ] Pipeline sklearn reprodutível (seed = 42 em todos os passos)
- [ ] FastAPI com `/predict`, `/health`, validação Pydantic, logging estruturado
- [ ] Testes automatizados ≥ 3: smoke test, schema (pandera), teste de API
- [ ] Linting com ruff sem erros; `pyproject.toml` como single source of truth

**Etapa 4 — Documentação e Entrega**
- [ ] Atualizar Model Card com métricas reais do modelo treinado
- [ ] README completo com instruções de setup, execução e arquitetura
- [ ] Vídeo STAR de 5 minutos
- [ ] (Opcional) Deploy em nuvem com endpoint público

---

## 10. Referências dos Cálculos

Todos os valores foram calculados com `Telco_customer_churn.xlsx` (IBM, n = 7.043). Features geográficas, identificadores, `Churn Score` e `CLTV` foram excluídos do pipeline de treino. `CLTV` e `Total Charges` são usados exclusivamente para cômputo do custo de FN fora do modelo.

```python
# Origem dos 114 churners/mês
churn_rate_anual  = 1869 / 7043          # = 26,5%
churn_rate_mensal = churn_rate_anual / 12 # = 2,21%
churners_mes      = 5174 * churn_rate_mensal  # = 114,4 → ~114

# Custo FN por cliente churned (calculado fora do pipeline de treino)
custo_fn = max(CLTV - Total_Charges, 0)

# Custo FP (desconto 10% do plano anual dos clientes ativos)
custo_fp = ticket_mensal_ativo * 12 * 0.10  # = US$ 73,52

# Razão de assimetria por faixa de tenure
razao_fn_fp = cltv_residual_mediana_faixa / custo_fp
# Range: 0,0x (49-72m) a 54,1x (0-6m) · mediana geral: 38,7x

# Custo de Churn Evitado (por cliente retido)
churn_evitado = cltv_residual_medio - custo_fp  # = US$ 2.689,39

# Threshold ótimo (a ser determinado na Etapa 2)
def custo_total(t, y_true, y_score, custo_fn_individual, custo_fp):
    y_pred = (y_score >= t).astype(int)
    idx_fn = (y_true == 1) & (y_pred == 0)
    idx_fp = (y_true == 0) & (y_pred == 1)
    return custo_fn_individual[idx_fn].sum() + idx_fp.sum() * custo_fp
```

---

*Documento gerado na Etapa 1 do Tech Challenge — MLOps · Predição de Churn · POSTECH*
*ML Canvas v0.1 (Louis Dorard) · machinelearningcanvas.com*
*Todos os valores calculados com IBM Telco Customer Churn (n = 7.043)*