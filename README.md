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