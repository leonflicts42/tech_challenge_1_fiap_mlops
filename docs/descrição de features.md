# Dataset de Churn de usuários de serviços de telefonia

## Sobre o Conjunto de Dados
### Contexto 
Uma empresa fictícia de telecomunicações que forneceu serviços de telefonia fixa e internet para 7.043 clientes na Califórnia no terceiro trimestre. 

### Fonte 
[Dataset Original Kaggle: Telco_customer_churn.xlsx](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset")

### Objetivo
Criar um modelo de machine learning para prever o churn de cliente no proximo mês.

### Descrição dos Dados 
7.043 observações com 33 variáveis 

**CustomerID**: Um ID único que identifica cada cliente.
 
**Count**: Um valor usado em relatórios e dashboards para somar o número de clientes em um conjunto filtrado.
 
**Country**: O país de residência principal do cliente.
 
**State**: O estado de residência principal do cliente.
 
**City**: A cidade de residência principal do cliente.
 
**Zip Code**: O CEP da residência principal do cliente.
 
**Lat Long**: A combinação de latitude e longitude da residência principal do cliente.
 
**Latitude**: A latitude da residência principal do cliente.
 
**Longitude**: A longitude da residência principal do cliente.
 
**Gender**: O gênero do cliente: Masculino, Feminino.
 
**Senior Citizen**: Indica se o cliente tem 65 anos ou mais: Sim, Não.
 
**Partner**: Indica se o cliente tem um parceiro(a): Sim, Não.
 
**Dependents**: Indica se o cliente mora com dependentes: Sim, Não. Dependentes podem ser filhos, pais, avós, etc.
 
**Tenure Months**: Indica o total de meses que o cliente está com a empresa até o final do trimestre especificado.
 
**Phone Service**: Indica se o cliente assina o serviço de telefone fixo da empresa: Sim, Não.
 
**Multiple Lines**: Indica se o cliente assina múltiplas linhas telefônicas com a empresa: Sim, Não.
 
**Internet Service**: Indica se o cliente assina o serviço de internet da empresa: Não, DSL, Fibra Óptica, Cabo.
 
**Online Security**: Indica se o cliente assina um serviço adicional de segurança online fornecido pela empresa: Sim, Não.
 
**Online Backup**: Indica se o cliente assina um serviço adicional de backup online fornecido pela empresa: Sim, Não.
 
**Device Protection**: Indica se o cliente assina um plano adicional de proteção de dispositivos para seus equipamentos de internet fornecido pela empresa: Sim, Não.
 
**Tech Support**: Indica se o cliente assina um plano adicional de suporte técnico da empresa com tempos de espera reduzidos: Sim, Não.
 
**Streaming TV**: Indica se o cliente utiliza seu serviço de internet para transmitir programação de televisão de um provedor terceiro: Sim, Não. A empresa não cobra taxa adicional por este serviço.
 
**Streaming Movies**: Indica se o cliente utiliza seu serviço de internet para transmitir filmes de um provedor terceiro: Sim, Não. A empresa não cobra taxa adicional por este serviço.
 
**Contract**: Indica o tipo de contrato atual do cliente: Mês a Mês, Um Ano, Dois Anos.
 
**Paperless Billing**: Indica se o cliente optou pela fatura sem papel (digital): Sim, Não.
 
**Payment Method**: Indica como o cliente paga sua fatura: Débito Bancário, Cartão de Crédito, Cheque pelos Correios.
 
**Monthly Charge**: Indica o valor mensal total atual do cliente por todos os seus serviços contratados.
 
**Total Charges**: Indica o total de cobranças do cliente, calculado até o final do trimestre especificado.
 
**Churn Label**: Sim = o cliente deixou a empresa neste trimestre. Não = o cliente permaneceu com a empresa. Diretamente relacionado ao Churn Value.
 
**Churn Value**: 1 = o cliente deixou a empresa neste trimestre. 0 = o cliente permaneceu com a empresa. Diretamente relacionado ao Churn Label.
 
**Churn Score**: Um valor de 0 a 100 calculado pela ferramenta preditiva IBM SPSS Modeler. O modelo incorpora múltiplos fatores conhecidos por causar churn. Quanto maior a pontuação, maior a probabilidade de o cliente cancelar.
 
**CLTV**: Customer Lifetime Value (Valor do Tempo de Vida do Cliente). O CLTV previsto é calculado usando fórmulas corporativas e dados existentes. Quanto maior o valor, mais valioso é o cliente. Clientes de alto valor devem ser monitorados quanto ao risco de churn.
 
**Churn Reason**: O motivo específico do cliente para deixar a empresa. Diretamente relacionado à Churn Category.

>>
cols_drop = COLS_ID + COLS_POS + [LABEL_COL]
COLS_ID: list[str] = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
]

# Colunas geradas APÓS o evento de churn — data leakage, removidas no cleaning
COLS_POS: list[str] = [
    "Churn Score",
    "CLTV",
    "Churn Reason",
]

# Features numéricas contínuas do dataset original
COLS_NUM: list[str] = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
]

# Features categóricas do dataset original
COLS_CAT: list[str] = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
]
>>
shape final: (7043, 18) | colunas: ['Senior Citizen', 'Partner', 'Dependents', 'Tenure Months', 'Multiple Lines', 'Internet Service', 'Online Security', 'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies', 'Contract', 'Paperless Billing', 'Payment Method', 'Monthly Charges', 'Total Charges', 'Churn Value']

