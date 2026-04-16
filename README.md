total charge veio como string, converter para float

Como outra pessoa rodará seu projeto?
Em vez de baixar a pasta .venv, a pessoa baixará seu código e os arquivos do uv, então rodará:

Bash
uv sync
Isso criará um ambiente virtual idêntico ao seu, de forma limpa e rápida.

# 1. Clona
git clone https://github.com/seu-usuario/churn-telecom.git
cd churn-telecom

# 2. Cria e ativa o ambiente
python -m venv .venv
.venv\Scripts\Activate.ps1   # PowerShell Windows

# 3. Instala o projeto — UMA linha resolve tudo
pip install -e .


# Git workflow — guia definitivo

Ciclo de vida completo de uma feature, do início ao fim, usando `git` e `uv`.

---

## 1. Sincronização inicial (obrigatório)

Sempre comece garantindo que sua `main` local é o espelho exato da remota.

```bash
git checkout main
git pull origin main
```

---

## 2. Criação da feature branch

Crie a branch a partir da `main` atualizada. Use nomes descritivos.

```bash
git checkout -b feature/analise-vif-mlflow
```

---

## 3. Ciclo de desenvolvimento

Trabalhe no código. Como você está usando o `uv`, se adicionar pacotes, lembre-se do `uv add`.

```bash
# após as alterações
git add .
git commit -m "feat: implementa cálculo de VIF com add_constant e logs mlflow"
git push origin feature/analise-vif-mlflow
```

### Convenção de mensagens de commit

| Prefixo | Quando usar |
|---|---|
| `feat:` | nova funcionalidade |
| `fix:` | correção de bug |
| `chore:` | configuração, dependências |
| `docs:` | documentação |
| `refactor:` | refatoração sem mudança de comportamento |
| `test:` | adição ou correção de testes |

---

## 4. Pull request (PR) e code review

1. Vá ao GitHub e abra o Pull Request.
2. Analise seu próprio código (auto-review) e rode os testes de CI.
3. No botão de merge do GitHub, selecione **Squash and Merge** — isso achata seus commits de "tentativa e erro" em um único commit limpo na `main`.

---

## 5. Limpeza e sincronização

Após o merge na interface web, limpe o repositório local.

```bash
# volta para a main
git checkout main

# atualiza a main local com o squash commit recém-mergeado
git pull origin main

# apaga a branch local que não serve mais
git branch -d feature/analise-vif-mlflow

# limpa referências de branches apagadas no remoto
git fetch --prune
```

---

## Resumo — loop mental para cada feature

| Ação | Comando |
|---|---|
| Limpar/atualizar | `git checkout main && git pull origin main` |
| Nova tarefa | `git checkout -b feature/nome-da-tarefa` |
| Salvar | `git add . && git commit -m "..." && git push origin HEAD` |
| Pós-merge | `git checkout main && git pull origin main && git branch -d feature/anterior` |

---

## Dica — lint e format antes do push

Essencial para o critério de qualidade de código (20% da nota do Tech Challenge).
Antes de cada `git commit`, rode:

```bash
uv run ruff check .    # linting — aponta erros
uv run ruff format .   # formatação automática
```

Para automatizar, adicione no `pyproject.toml`:

```toml
[tool.ruff]
line-length = 88
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

E no `Makefile` (raiz do projeto):

```makefile
lint:
	uv run ruff check .

format:
	uv run ruff format .

pre-commit: format lint
```

Uso:

```bash
make pre-commit   # formata e checa antes de commitar
```
```
projeto_final
├─ .python-version
├─ churn_telecom
│  ├─ config.py
│  ├─ plots.py
│  └─ __init__.py
├─ data
│  ├─ interim
│  │  ├─ telco_cleaned.parquet
│  │  ├─ telco_features.parquet
│  │  └─ telco_typed.parquet
│  ├─ processed
│  │  ├─ test.parquet
│  │  └─ train.parquet
│  └─ raw
│     └─ raw_telco_customer_churn.xlsx
├─ docs
│  ├─ analise_0.00_vab_project_descrition.md
│  ├─ analise_0.01_vab_data_source.md
│  ├─ analise_0.02_vab_eda_univariate.md
│  ├─ analise_0.03_vab_eda_bivariate.md
│  ├─ analise_0.04_vab_eda_multivariate.md
│  ├─ analise_1.01_vab_data_cleaning.md
│  ├─ analise_1.02_vab_feature_engineering.md
│  ├─ analise_1.03_vab_preprocessing.md
│  ├─ analise_eda.md
│  ├─ metricas_tecnicas_negocios.md
│  ├─ ml_canvas.md
│  └─ pipeline_data_to_baseline.html
├─ estrutura.txt
├─ LICENSE
├─ logs
├─ main.py
├─ mlflow.db
├─ models
│  └─ preprocessor.pkl
├─ notebooks
│  ├─ 0.01_vab_data_source.ipynb
│  ├─ 0.02_vab_eda_univariate.ipynb
│  ├─ 0.03_vab_eda_bivariate.ipynb
│  ├─ 0.04_vab_eda_multivariate.ipynb
│  ├─ 1.01_vab_data_cleaning.ipynb
│  ├─ 1.02_vab_feature_engineering.ipynb
│  ├─ 1.03_vab_preprocessing.ipynb
│  ├─ 3.01_vab_baseline_dummy.ipynb
│  └─ 3.02_vab_baseline_logistic.ipynb
├─ pyproject.toml
├─ README.md
├─ references
├─ reports
│  └─ figures
│     ├─ baselines
│     │  ├─ dummy_confusion_matrix.png
│     │  ├─ dummy_roc_curve.png
│     │  ├─ logistic_confusion_matrix.png
│     │  ├─ logistic_feature_importance.png
│     │  └─ logistic_roc_curve.png
│     ├─ bivariate_cat_contract.png
│     ├─ bivariate_cat_dependents.png
│     ├─ bivariate_cat_device_protection.png
│     ├─ bivariate_cat_gender.png
│     ├─ bivariate_cat_internet_service.png
│     ├─ bivariate_cat_multiple_lines.png
│     ├─ bivariate_cat_online_backup.png
│     ├─ bivariate_cat_online_security.png
│     ├─ bivariate_cat_paperless_billing.png
│     ├─ bivariate_cat_partner.png
│     ├─ bivariate_cat_payment_method.png
│     ├─ bivariate_cat_phone_service.png
│     ├─ bivariate_cat_senior_citizen.png
│     ├─ bivariate_cat_streaming_movies.png
│     ├─ bivariate_cat_streaming_tv.png
│     ├─ bivariate_cat_tech_support.png
│     ├─ bivariate_num_monthly charges.png
│     ├─ bivariate_num_tenure months.png
│     ├─ bivariate_num_total charges.png
│     ├─ churn_distribution.png
│     ├─ correlation_matrix_numeric.png
│     ├─ missing_values.png
│     ├─ univariate_cat_contract.png
│     ├─ univariate_cat_dependents.png
│     ├─ univariate_cat_device_protection.png
│     ├─ univariate_cat_gender.png
│     ├─ univariate_cat_internet_service.png
│     ├─ univariate_cat_multiple_lines.png
│     ├─ univariate_cat_online_backup.png
│     ├─ univariate_cat_online_security.png
│     ├─ univariate_cat_paperless_billing.png
│     ├─ univariate_cat_partner.png
│     ├─ univariate_cat_payment_method.png
│     ├─ univariate_cat_phone_service.png
│     ├─ univariate_cat_senior_citizen.png
│     ├─ univariate_cat_streaming_movies.png
│     ├─ univariate_cat_streaming_tv.png
│     ├─ univariate_cat_tech_support.png
│     ├─ univariate_num_monthly_charges.png
│     ├─ univariate_num_tenure_months.png
│     ├─ univariate_num_total_charges.png
│     ├─ univariate_skew_kurt_monthly_charges.png
│     ├─ univariate_skew_kurt_tenure_months.png
│     ├─ univariate_skew_kurt_total_charges.png
│     └─ vif_table.csv
├─ requirements.txt
├─ src
└─ uv.lock

```