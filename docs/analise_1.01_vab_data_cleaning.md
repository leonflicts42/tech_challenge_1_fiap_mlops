# Relatório — Data Cleaning (1.01)

**Notebook:** `1.01_vab_data_cleaning`
**Entrada:** `data/interim/telco_typed.parquet` — shape (7043, 20)
**Saída:** `data/interim/telco_cleaned.parquet` — shape (7021, 20)
**Status:** concluído com sucesso

---

## Resumo executivo

| Métrica | Antes | Depois | Delta |
|---|---|---|---|
| Linhas | 7.043 | 7.021 | -22 (duplicatas) |
| Colunas | 20 | 20 | 0 |
| Nulos | 11 | 0 | -11 |
| Colunas snake_case | não | sim | — |
| Binárias Yes/No encodadas | não | sim (11 colunas) | — |
| Categóricas nominais tipadas | não | sim (5 colunas) | — |

---

## 1. Validações iniciais

### Shape e schema

O dataset foi carregado com shape `(7043, 20)`, dentro do esperado (`DATA_ROWS=7043`, delta=0). A validação de schema reportou 13 colunas ausentes — comportamento esperado e correto, pois as colunas de ID/Geo (`CustomerID`, `Country`, `City`, etc.) e as colunas de leakage (`Churn Score`, `CLTV`, `Churn Reason`) já haviam sido removidas no notebook `0.01_data_source` antes da persistência em `telco_typed.parquet`. As etapas de remoção do `1.01` encontraram zero colunas a remover, confirmando que o dataset de entrada já estava parcialmente tratado.

### Padronização de colunas

Todos os 20 nomes de colunas foram convertidos para `snake_case` via `to_snake_case()` importado do `config.py`. Exemplo: `Monthly Charges` → `monthly_charges`. Essa padronização é necessária para o acesso consistente no `ColumnTransformer` do `1.03` e nos módulos `src/`.

---

## 2. Tratamento de nulos

### `total_charges` — 11 nulos imputados

O log confirmou a hipótese levantada na EDA: os 11 registros nulos em `total_charges` correspondem exclusivamente a clientes com `tenure_months=0`, ou seja, clientes recém-chegados que ainda não receberam nenhuma cobrança acumulada.

```
total_charges | nulos=11 | tenure dos nulos=[0] | hipótese tenure=0: True
total_charges | 11 nulos imputados com mediana=1397.47
```

**Decisão:** imputação com mediana (`1397.47`) em vez de `0.0`. Justificativa: o valor `0.0` é artificialmente fora da distribuição real da coluna e distorceria o `StandardScaler` na etapa de preprocessing. A mediana é robusta à assimetria à direita confirmada pela EDA (`skewness=0.962`). Essa decisão está alinhada com a recomendação da literatura (Springer/2024).

---

## 3. Normalização semântica

### "No internet service" → "No" — 1.526 registros por coluna

Seis colunas de serviços de internet apresentavam a categoria redundante `"No internet service"`, que semanticamente equivale a `"No"` (o cliente simplesmente não tem acesso ao serviço). A normalização foi aplicada nas seguintes colunas:

- `online_security`
- `online_backup`
- `device_protection`
- `tech_support`
- `streaming_tv`
- `streaming_movies`

**Impacto:** redução de cardinalidade de 3 para 2 categorias em cada coluna, eliminando uma dimensão desnecessária no `OneHotEncoder` do `1.03` e tornando o encoding mais eficiente.

**Observação:** a etapa de correção de inconsistências lógicas (clientes sem internet com serviços ativos) foi executada sem encontrar registros inconsistentes — confirmando que o dataset IBM Telco é internamente consistente nesse aspecto após a normalização.

---

## 4. Encoding de variáveis categóricas

### Label Encoding — 11 colunas binárias Yes/No → 0/1

```
senior_citizen, partner, dependents, phone_service, online_security,
online_backup, device_protection, tech_support, streaming_tv,
streaming_movies, paperless_billing
```

Colunas genuinamente binárias sem hierarquia implícita. O mapeamento `{"Yes": 1, "No": 0}` é reversível e não introduz ordenação artificial. Essa abordagem é recomendada pela literatura para variáveis binárias e simplifica o `ColumnTransformer` do `1.03`.

### Tipagem category — 5 colunas nominais

```
gender, multiple_lines, internet_service, contract, payment_method
```

Colunas com 3 ou mais categorias sem hierarquia natural foram tipadas como `category`. O `OneHotEncoder` com `drop="first"` será aplicado no `1.03` para evitar a dummy trap.

---

## 5. Validação de ranges numéricos

Todas as três variáveis numéricas estão dentro dos limites esperados:

| Coluna | Range esperado | Status |
|---|---|---|
| `tenure_months` | [0, 72] | OK |
| `monthly_charges` | [0, 200] | OK |
| `total_charges` | [0, 10.000] | OK |

Nenhuma anomalia detectada — confirma a ausência de outliers críticos identificada na EDA univariada (`outliers_iqr=0` e `outliers_z=0` para todas as numéricas).

---

## 6. Remoção de duplicatas

```
Duplicatas removidas: 22
RESUMO LIMPEZA | linhas: 7043→7021 (delta=22)
```

**Achado relevante:** 22 linhas duplicadas identificadas e removidas. Esse dado não havia sido detectado na EDA do notebook `0.01` — o `df.duplicated()` retornou 0 naquele momento porque o dataset ainda continha todas as 33 colunas originais (incluindo `CustomerID`, que é único por definição). Com a remoção do `CustomerID` no `0.01`, registros que diferiam apenas nesse identificador passaram a ser duplicatas estruturais.

**Impacto:** redução de 7.043 para 7.021 linhas (0.31% do dataset). A proporção de churn permanece estável:

- Churn: ~26.54% → ~26% (inalterada)
- Não churn: ~73.46% → ~74% (inalterada)

---

## 7. Checklist final

| Validação | Resultado |
|---|---|
| Sem nulos | True |
| Sem duplicatas | True |
| Target binário {0, 1} | True |
| Ranges numéricos OK | True |
| Persistência validada (shape pós-save) | True |

---

## 8. Decisões técnicas pendentes

As seguintes decisões foram tomadas nessa etapa mas serão implementadas nas próximas:

`gender` e `phone_service` foram mantidas — candidatas ao descarte após análise de feature importance nos baselines (p>0.05 e Cramer's V < 0.02 na EDA bivariada).

`total_charges` permanece no dataset — a literatura recomenda manutenção com median imputer (já aplicado). A transformação `log1p` será aplicada no `1.02` para tratar a assimetria (`skewness=0.962`).

`OneHotEncoder` para as 5 colunas nominais (`internet_service`, `contract`, `payment_method`, `gender`, `multiple_lines`) será aplicado no `ColumnTransformer` do `1.03`.

---

## 9. Próximo passo

**Notebook:** `1.02_feature_engineering`
**Entrada:** `data/interim/telco_cleaned.parquet` — shape (7021, 20)
**Features a criar:** `num_services`, `charges_per_month`, `is_month_to_month`, `tenure_group`, `has_security_support`
**Transformação:** `total_charges_log = log1p(total_charges)`