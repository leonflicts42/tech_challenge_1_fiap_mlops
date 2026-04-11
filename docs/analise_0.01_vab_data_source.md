# Análise Inicial do Dataset — Data Understanding

## Dimensionalidade e Estrutura

O dataset contém 7.043 clientes e 33 variáveis brutas, reduzidas para 20 após a remoção de
identificadores, colunas geográficas e variáveis de leakage. A distribuição por tipo ficou:
3 numéricas (`Tenure Months`, `Monthly Charges`, `Total Charges`), 16 categóricas e 9 de
ID/Geo descartadas.

---

## Qualidade dos Dados

Dois problemas identificados:

`Total Charges` chegou como `object` no raw — convertida para `float64` via
`pd.to_numeric(errors='coerce')`, gerando 11 nulos (0.16%) correspondentes a clientes com
`Tenure Months = 0`, ou seja, recém-chegados sem nenhuma cobrança acumulada. Estratégia:
imputar com `0` no notebook `1.01-cleaning`, pois a ausência reflete um estado real do
negócio, não um dado faltante.

`Churn Reason` tem 73.46% de nulos (5.174 registros) — estruturalmente esperado, pois só
clientes que cancelaram têm motivo registrado. Essa coluna está isolada em `cols_pos` e não
entrará no modelo.

Zero duplicatas confirmado.

---

## Desbalanceamento de Classes

| Classe            | Contagem |    %  |
|-------------------|----------|-------|
| Não cancelou (0)  |   5.174  | 73.46% |
| Cancelou (1)      |   1.869  | 26.54% |
| **Ratio**         | **0.36** | —      |

O ratio de 0.36 confirma desbalanceamento moderado. Com ratio < 0.5 duas estratégias serão
avaliadas na modelagem:

- `class_weight='balanced'` nos modelos sklearn (Regressão Logística, baseline)
- `pos_weight` no `BCEWithLogitsLoss` do PyTorch (MLP)

SMOTE será avaliado como alternativa se as métricas de Recall ficarem insatisfatórias.

---

## Implicação Direta no Negócio

O desbalanceamento reflete a realidade operacional — a maioria dos clientes não cancela.
Isso torna o Recall da classe positiva (Cancelou=1) a métrica mais crítica:

- **Falso Negativo**: cliente que vai cancelar e não foi identificado → receita perdida
permanentemente.
- **Falso Positivo**: campanha de retenção desnecessária → custo controlável.

Essa assimetria de custos será formalizada no ML Canvas na próxima etapa.

---

## Próximos Passos

O dataset está íntegro e pronto para avançar. O `telco_typed.parquet` com shape `(7043, 20)`
é a entrada do `0.02-eda-univariate`, que analisará as distribuições individuais das
3 numéricas e 16 categóricas sem segmentação por target.