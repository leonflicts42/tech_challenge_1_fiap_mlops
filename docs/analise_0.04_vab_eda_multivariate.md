# 📊 Análise Multivariada e de Multicolinearidade

Esta fase da análise exploratória ultrapassa a descrição individual das variáveis para investigar como elas interagem entre si e, fundamentalmente, como explicam o fenómeno do **Churn**. O foco aqui é garantir a estabilidade do modelo e eliminar redundâncias que possam inflar métricas ou instabilizar os pesos da Rede Neural (MLP).

## 1. Correlação com o Target (Sinal Preditivo)
A análise de correlação de Pearson revelou a hierarquia de influência das variáveis numéricas sobre a variável alvo:

* **Tenure Months (-0.3522):** Apresenta a correlação mais forte (moderada e negativa). Isto confirma que quanto maior o tempo de permanência do cliente, menor a probabilidade de churn. É o principal "âncora" de retenção no dataset.
* **Total Charges (-0.1995):** Correlação fraca/moderada negativa. Embora pareça relevante, o seu sinal é derivado diretamente do tempo de casa.
* **Monthly Charges (0.1934):** Correlação positiva. Indica que faturas mais elevadas tendem a impulsionar o cancelamento, sugerindo uma sensibilidade ao preço por parte dos utilizadores.

## 2. O Problema da Multicolinearidade (VIF & Redundância)
A análise multivariada detetou uma relação estrutural que pode prejudicar a performance do modelo se não for tratada:

* **Alta Correlação Detetada:** O par **Tenure Months vs Total Charges** apresenta uma correlação de **0.8259**.
    * **Explicação:** Esta é uma relação matemática óbvia — o valor total pago é, em grande parte, o produto do tempo de contrato pelo valor mensal. Ter ambas as variáveis no modelo é fornecer a mesma informação duas vezes.
* **Variance Inflation Factor (VIF):**
    * **Total Charges (8.0792):** Valor elevado (próximo do limite crítico de 10). Indica que esta variável é altamente explicada pelas outras.
    * **Tenure Months (6.3324):** Valor de atenção, inflado pela sua ligação com o custo total.
    * **Monthly Charges (3.3611):** Nível saudável, indicando que traz informação única ao modelo.

## 3. Descobertas Vitais para o Avanço do Problema
* **Redundância Estrutural:** A variável **Total Charges** não adiciona valor incremental significativo à predição que já não esteja contido em **Tenure** e **Monthly Charges**. Manter variáveis com VIF alto pode tornar os gradientes da MLP instáveis durante o treino.
* **Predominância Categórica:** Embora as variáveis numéricas forneçam o "motor" do modelo, a teoria de domínio em Telecomunicações indica que o comportamento decisório (Churn) é dominado por variáveis categóricas como **Contract** e **Internet Service**.

## 4. Insights e Próximas Etapas (Feature Selection)
Com base nos logs, as seguintes decisões técnicas serão aplicadas na construção do pipeline:

* **Estratégia de Descarte:** Recomenda-se a **remoção de Total Charges**. Ao remover esta variável, o VIF de **Tenure Months** cairá drasticamente, isolando o seu efeito real e purificando o sinal enviado à Rede Neural.
* **Foco na Engenharia de Categóricas:** Dado que as numéricas têm correlações moderadas, o sucesso do **F1-Score** dependerá de como transformamos as variáveis categóricas (**One-Hot Encoding** para serviços e **Ordinal** para contratos).
* **Preparação para a MLP:** Como as Redes Neurais tratam bem interações não-lineares, a manutenção de **Tenure** e **Monthly Charges** (devidamente normalizados) permitirá que a rede aprenda a relação complexa que **Total Charges** tentava representar de forma linear e redundante.

**Conclusão desta Fase:** O dataset está pronto para a etapa de pré-processamento. A limpeza de redundâncias (Multicolinearidade) garantirá um modelo mais enxuto, interpretável e com melhor capacidade de generalização para a API de inferência.