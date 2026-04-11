# Projeto de MLOps: Predição de Churn em Telecomunicações

## 1. Descrição do Problema e Contexto
[cite_start]Uma operadora de telecomunicações está enfrentando uma perda acelerada de clientes, fenômeno conhecido como **churn**[cite: 10]. A retenção é vital, pois o custo de aquisição de um novo cliente é significativamente superior ao custo de manter um atual. [cite_start]Para mitigar esse impacto, a diretoria necessita de um modelo preditivo que classifique clientes com risco de cancelamento antes que a evasão ocorra[cite: 11].

O desafio central não é apenas prever a saída, mas **otimizar as métricas de precisão e sensibilidade (Recall)** para equilibrar o trade-off financeiro:
* **Identificação Precisa:** Priorizar a detecção de clientes com alto potencial de retorno financeiro.
* [cite_start]**Eficiência de Custo:** Minimizar falsos positivos para evitar gastos inúteis com campanhas de retenção e reduzir falsos negativos para evitar a perda de receita por falha de detecção[cite: 46, 50].

## 2. Objetivo do Projeto
[cite_start]Desenvolver um pipeline profissional *end-to-end* cujo núcleo é uma **Rede Neural Multicamadas (MLP)** treinada com **PyTorch**[cite: 8, 13]. [cite_start]O objetivo técnico é superar modelos *baseline* em métricas como **AUC-ROC**, **PR-AUC** e **F1-Score**, garantindo que o modelo seja servido via uma **API de inferência** funcional[cite: 32, 46].

## 3. Descrição do Dataset (IBM Telco Customer Churn)
[cite_start]O projeto utiliza, preferencialmente, o dataset público da **IBM**, composto por variáveis tabulares que detalham a jornada do cliente[cite: 70, 71, 72]:
* **Geolocalização e Perfil:** Identificação, localização geográfica e atributos demográficos (gênero, dependentes, etc.).
* **Serviços Contratados:** Detalhes sobre serviços de telefonia, internet, segurança online e suporte técnico.
* **Dados Financeiros:** Tipo de contrato, tempo de casa (`Tenure Months`), métodos de pagamento e cobranças (`Monthly/Total Charges`).
* **Target:** `Churn Value` ou `Churn Label`, indicando se o cliente deixou a empresa no período especificado.

## 4. Estrutura Geral do Projeto (Requisitos Tech Challenge)
[cite_start]O desenvolvimento é estruturado em quatro etapas fundamentais, integrando ciência de dados e engenharia de ML[cite: 40]:

### **Etapa 1: Entendimento e Preparação**
* [cite_start]Preenchimento do **ML Canvas** (stakeholders, métricas de negócio e SLOs)[cite: 44].
* [cite_start]**EDA completa** para avaliar volume, qualidade, distribuição e prontidão dos dados (*data readiness*)[cite: 46].
* [cite_start]Definição de métricas técnicas e de negócio (custo de churn evitado)[cite: 46].
* [cite_start]Treinamento e registro de **baselines** (Dummy e Regressão Logística) no **MLflow**[cite: 46, 47].

### **Etapa 2: Modelagem com Redes Neurais**
* [cite_start]Construção da **MLP em PyTorch** com loop de treinamento, *batching* e *early stopping*[cite: 49, 50].
* [cite_start]Comparação da MLP contra baselines usando ao menos 4 métricas[cite: 50, 52].
* [cite_start]Análise rigorosa do **trade-off de custo** entre falsos positivos e negativos[cite: 50].

### **Etapa 3: Engenharia e API**
* [cite_start]Refatoração do código em módulos profissionais dentro da pasta `src/`[cite: 56, 57].
* [cite_start]Criação de **pipelines reprodutíveis** e testes automatizados com `pytest` (smoke, schema e API)[cite: 37, 57].
* [cite_start]Construção da API com **FastAPI**, incluindo validação Pydantic e logging estruturado[cite: 32, 57].

### **Etapa 4: Documentação e Entrega Final**
* [cite_start]Geração de um **Model Card** completo detalhando limitações, vieses e cenários de falha[cite: 36, 61].
* [cite_start]Documentação da arquitetura de deploy e plano de monitoramento[cite: 61].
* [cite_start]Elaboração do **README.md** e gravação do vídeo de 5 minutos seguindo o **método STAR**[cite: 16, 21, 64].