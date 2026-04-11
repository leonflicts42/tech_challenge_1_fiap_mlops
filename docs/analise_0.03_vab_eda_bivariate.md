# 📊 Análise Bivariada: Variáveis Numéricas (O Motor do Modelo)

Os dados confirmam que o comportamento financeiro e o tempo de relacionamento são os principais discriminadores de clientes.

## 1. Tenure Months (Tempo de Casa) — A Variável de Ouro
* **Estatística:** Apresentou um Cohen’s d de -0.89 (Efeito Grande) e correlação de -0.35.
* **Descoberta:** O grupo Churn tem uma mediana de apenas 10 meses, enquanto o grupo Non-Churn tem 38 meses.
* **Insight de Negócio:** A empresa perde clientes muito cedo. Existe uma "curva de aprendizado" ou de "encantamento" crítica nos primeiros 12 meses.
* **Ação para ML:** Esta variável deve ser preservada sem cortes, pois é o separador mais forte entre as classes.

## 2. Monthly Charges (Valor da Fatura)
* **Estatística:** Cohen’s d de 0.46 (Efeito Médio).
* **Descoberta:** Clientes que saem pagam, em média, R$ 74,44, contra R$ 61,27 dos que ficam.
* **Insight de Negócio:** O churn está concentrado em clientes de ticket mais alto. Isso sugere que a sensibilidade ao preço ou a percepção de valor em planos premium está em xeque.

## 3. Total Charges (Custo Total)
* **Estatística:** Cohen’s d de -0.48 (Efeito Médio).
* **Descoberta:** O valor total é menor no Churn porque o tempo de casa (Tenure) é curto.
* **Alerta Técnico:** O log apontou 5.83% de outliers no grupo Churn para esta variável. Como a MLP é sensível a escalas, o uso de transformações logarítmicas ou robustas aqui é obrigatório para evitar instabilidade.

# 🏷️ Análise Bivariada: Variáveis Categóricas (O Perfil do Risco)

Aqui separamos as variáveis que "explicam" o problema daquelas que são apenas descritivas.

## 1. Os "Candidatos ao Descarte" (Baixo Impacto)
* **Gender (Gênero):** p-value de 0.48 e Cramer’s V de 0.008. Não há diferença estatística entre homens e mulheres na propensão ao churn.
* **Phone Service:** p-value de 0.33 e Cramer’s V de 0.011.
* **Ação para ML:** Considere remover estas colunas. Elas adicionam complexidade (colunas no One-Hot Encoding) sem agregar poder preditivo, o que pode gerar ruído na sua Rede Neural.

## 2. As "Váriaveis de Alto Impacto" (Sinal Forte)
* **Contract (Cramer's V = 0.41 - Forte):** A taxa de churn no contrato Month-to-month é de 42.71%, enquanto no de dois anos é de apenas 2.83%. É o maior preditor categórico do dataset.
* **Internet Service (Cramer's V = 0.32 - Moderado):** O serviço de Fibra Óptica tem um churn alarmante de 41.89%.
* **Hipótese:** Ou o serviço é caro demais, ou a qualidade técnica da fibra nesta operadora está gerando insatisfação.
* **Online Security & Tech Support (Cramer's V ~ 0.34):** Clientes que não possuem estes serviços têm churn acima de 41%. Serviços de suporte técnico agem como "âncoras" de retenção.
* **Payment Method (Cramer's V = 0.30):** O Electronic check tem churn de 45.29%. Métodos manuais ou menos estáveis facilitam a saída do cliente.

# 💡 Insights Estratégicos para as Próximas Etapas

Com base nesta análise, seu projeto ganha o seguinte direcionamento para a Etapa 2 (Modelagem):

* **Feature Selection Baseada em Evidência:** Você tem justificativa estatística (p-value e Cramer's V) para focar em Contract, Internet Service, Tech Support e Tenure, enquanto ignora Gender e Phone Service.
* **Tratamento de Desbalanceamento:** A análise bivariada mostrou que as proporções de churn variam drasticamente entre categorias. Isso reforça a necessidade de usar StratifiedKFold na validação da sua MLP para garantir que cada dobra de treino represente essas proporções.
* **Engenharia de Atributos:**
    * Crie uma variável binária "Is_Fiber_Optic", dado o alto impacto desse serviço no churn.
    * Crie uma feature "Has_Security_Support" que consolide se o cliente tem serviços de proteção, já que ambos mostraram Cramer's V alto.
* **Foco Financeiro:** Como o churn de planos mais caros (Monthly Charges alto) é mais frequente, seu modelo não deve apenas buscar acurácia, mas sim Recall (sensibilidade) alto, para não deixar passar esses clientes valiosos que estão saindo.

**Conclusão:** Os dados mostram que o churn não é aleatório; ele é impulsionado por contratos curtos, ausência de suporte técnico e insatisfação com serviços premium (Fibra). Sua MLP agora tem um "mapa do tesouro" para focar os pesos das conexões neurais.