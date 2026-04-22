#mlflow-ui:
    #.venv\Scripts\mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

makefile
.PHONY: format lint pre-commit test

format:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest

pre-commit: format lint test