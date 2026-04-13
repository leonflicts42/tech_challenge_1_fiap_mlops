# 📊 Relatório — Feature Engineering (1.02_vab_feature_engineering)

## 🎯 Objetivo

Este notebook tem como objetivo enriquecer o dataset de churn com novas features derivadas, visando melhorar o poder preditivo de modelos futuros.

---

## 📥 Entrada de Dados

* Arquivo: `telco_cleaned.parquet`
* Shape inicial: **(7021, 20)**

---

## ⚙️ Processamento

Foram criadas 6 novas features:

### 🔹 1. num_services

Contagem de serviços contratados pelo cliente.

### 🔹 2. charges_per_month

Razão entre valor total pago e tempo de permanência.

### 🔹 3. is_month_to_month

Indica se o contrato é mensal.

### 🔹 4. tenure_group

Segmentação do tempo de contrato:

* novo
* médio
* longo

### 🔹 5. has_security_support

Indica presença de serviços adicionais de segurança/suporte.

### 🔹 6. is_fiber_optic

Indica uso de internet fibra.

---

## 📊 Principais Insights

### 📉 Relação com churn

* Clientes com **mais serviços** apresentam menor churn
* Contratos **month-to-month** têm alto churn (42.6%)
* Clientes **novos** têm maior risco (47.4%)
* Serviços de **segurança/suporte reduzem churn**
* Clientes com **fibra óptica apresentam churn elevado**

---

## 📈 Correlação com Target

| Feature              | Correlação |
| -------------------- | ---------- |
| charges_per_month    | 0.4094     |
| is_month_to_month    | 0.4049     |
| is_fiber_optic       | 0.3082     |
| has_security_support | -0.1817    |
| num_services         | -0.0842    |

---

## 📤 Saída

* Arquivo: `telco_features.parquet`
* Shape final: **(7021, 26)**
* Features adicionadas: 6

---

## ✅ Conclusão

O processo de feature engineering foi bem-sucedido, gerando variáveis com forte relação com churn, especialmente:

* tipo de contrato
* valor pago por mês
* tempo de permanência

Essas variáveis devem contribuir significativamente para a performance de modelos preditivos.

---

## 🚀 Próximos Passos

* Encoding de variáveis categóricas
* Feature selection
* Treinamento de modelos
* Avaliação de performance

---
