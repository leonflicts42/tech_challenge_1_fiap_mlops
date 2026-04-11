## Análise univariada das features

### Descobertas Vitais: Variáveis Numéricas
A análise revelou comportamentos muito claros sobre as variáveis financeiras e de tempo de casa do cliente:

- A "Armadilha" dos Valores Nulos Ocultos: Observe o log com atenção. O dataset possui 7043 linhas, mas a variável Total Charges registrou n=7032. Isso significa que existem 11 valores nulos que precisam ser tratados. Como são clientes com Tenure = 0 (acabaram de assinar e ainda não pagaram o primeiro mês), a melhor estratégia costuma ser preencher esses nulos com 0.0.

- Ausência de Outliers Extremos: O log indica outliers_iqr=0 (0.00%) e outliers_zscore=0 (0.00%) para todas as três variáveis. Isso é uma excelente notícia! Significa que você não precisará aplicar técnicas drásticas de corte (clipping) ou usar algoritmos ultrarrobustos a outliers.

- Distribuições Não Normais (Assimetria e Curtose):  
  * O Teste de Shapiro-Wilk (p=0.0000) confirmou que nenhuma das três variáveis segue uma distribuição normal.  

  Tenure Months e Monthly Charges são platicúrticas (caudas leves, valores mais espalhados sem um pico central forte).  

  Total Charges possui uma forte assimetria à direita (skewness=0.962). Isso ocorre porque a maioria dos clientes tem cobranças totais baixas (estão há pouco tempo ou têm planos baratos), enquanto uma minoria acumula valores altíssimos ao longo dos anos.

---

### Descobertas Vitais: Variáveis Categóricas

O seu log processou 16 variáveis categóricas sem encontrar dados nulos ou categorias raras (abaixo de 1%), o que mostra uma base bem estruturada. No entanto, temos alertas importantes:

- Variáveis de Alta Concentração (Baixa Variância):

  Phone Service: 90.3% dos clientes possuem o serviço.  

  Senior Citizen: 83.8% não são idosos.  

  Alerta: Variáveis com quase toda a massa de dados em uma única categoria tendem a ter baixo poder preditivo. O modelo não consegue extrair padrões claros de diferenciação. Elas são fortes candidatas a serem descartadas após a análise bivariada, simplificando sua arquitetura.

- Distribuição de Contratos: O Contract do tipo Month-to-month domina 55.0% da base. Historicamente, em problemas de Telco, essa é a variável com maior peso preditivo para o Churn, pois não há barreira de saída (multa rescisória) para esses clientes.

---

## Insights e Plano de Ação para as Próximas Etapas

Com base nesses dados, aqui estão as diretrizes técnicas que você deve levar para a construção do seu pipeline no Scikit-Learn e PyTorch:

- Transformação de Dados (Pipeline Sklearn):

  Como sugerido pelo seu próprio log, aplique uma transformação matemática (como np.log1p ou PowerTransformer do tipo Box-Cox/Yeo-Johnson) nas variáveis numéricas, especialmente em Total Charges, para reduzir a assimetria à direita.  

  Após a transformação, aplique um StandardScaler ou MinMaxScaler. Redes Neurais (MLP) são extremamente sensíveis à escala das features; alimentá-las com valores brutos de Total Charges na casa dos milhares causará instabilidade no gradiente.

- Tratamento de Categóricas:

  Para as variáveis binárias (como Partner, Dependents, Paperless Billing), utilize um OrdinalEncoder ou mapeamento simples para 0 e 1.  

  Para as variáveis multiclasse sem hierarquia (como Payment Method, Internet Service), aplique OneHotEncoder.

- Validação de Hipóteses (Preparação para Bivariada):

  A próxima etapa natural é cruzar essas variáveis com o nosso alvo (Churn). A alta concentração de Phone Service realmente importa para o Churn, ou podemos focar apenas no tipo de Internet Service?