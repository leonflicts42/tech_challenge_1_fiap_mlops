"""
api/predictor.py — ChurnPredictor: orquestra o pipeline completo de inferência.

Responsabilidades:
    1. Carregar preprocessor.pkl e modelo .pt uma única vez (via lifespan)
    2. Receber ChurnRequest, executar todas as transformações de pré-processamento
    3. Executar inferência com o ChurnMLPv2 e retornar ChurnResponse

Fluxo interno de transform():
    ChurnRequest
        → dict  (model_dump)
        → DataFrame  (pd.DataFrame)
        → snake_case nos nomes de colunas  (config.to_snake_case)
        → SemanticNormalizer.transform()   (preprocessing.py)
        → FeatureEngineer.transform()      (features.py)
        → preprocessor.pkl.transform()     (ColumnTransformer)
        → tensor float32  (torch.from_numpy)
        → ChurnMLPv2.forward()             (logit)
        → torch.sigmoid()                  (probabilidade)
        → ChurnResponse
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from api.schemas import ChurnRequest, ChurnResponse
from config import (
    COST_FP,
    DEVICE,
    MODELS_DIR,
    N_FEATURES_FINAL,
    PROJECT_ROOT,
    get_logger,
    to_snake_case,
)
from data.features import FeatureEngineer
from data.preprocessing import SemanticNormalizer
from models.mlp2 import ChurnMLPInference

logger: logging.Logger = get_logger(__name__)

# ── Patch: sklearn bug — np.isnan falha em categorias string (dtype=object) ──
# Afeta OrdinalEncoder/OneHotEncoder em sklearn >= 1.5 quando categories_
# são arrays de strings. O isnan foi adicionado para detectar NaN em
# categorias numéricas, mas não trata o caso de dtype=object.
import sklearn.utils._encode as _sklearn_encode  # noqa: E402

_orig_check_unknown = _sklearn_encode._check_unknown


def _patched_check_unknown(
    values: np.ndarray,
    known_values: np.ndarray,
    return_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    try:
        return _orig_check_unknown(values, known_values, return_mask=return_mask)
    except TypeError:
        diff = np.setdiff1d(values, known_values)
        if return_mask:
            return diff, np.isin(values, known_values)
        return diff


_sklearn_encode._check_unknown = _patched_check_unknown
# ─────────────────────────────────────────────────────────────────────────────

# Threshold padrão (substituído pelo ótimo do Optuna quando disponível)
_DEFAULT_THRESHOLD: float = 0.50
_OPTUNA_PARAMS_GLOB: str = "optuna_best_params_*.json"
REPORTS_JSON_DIR: Path = PROJECT_ROOT / "reports" / "json"
OPTUNA_PARAMS_PATH: Path = REPORTS_JSON_DIR / "optuna_best_params.json"


class ChurnPredictor:
    """Orquestra o pipeline completo de inferência de churn.

    Instanciado uma única vez no lifespan da aplicação FastAPI e
    armazenado em app.state.predictor.

    Args:
        preprocessor_path: caminho para o preprocessor.pkl
        model_path:        caminho para o arquivo .pt dos pesos da MLP
        threshold:         limiar de decisão (sobrescrito se Optuna params existir)
        device:            dispositivo PyTorch ("cpu" ou "cuda")

    Raises:
        FileNotFoundError: se preprocessor_path ou model_path não existirem
        RuntimeError:      se o modelo carregado tiver arquitetura incompatível
    """

    def __init__(
        self,
        preprocessor_path: Path | str = MODELS_DIR / "preprocessor.pkl",
        model_path: Path | str = MODELS_DIR / "best_model_mlp.pt",
        threshold: float = _DEFAULT_THRESHOLD,
        device: str = DEVICE,
    ) -> None:
        self.device = device
        self.threshold = threshold
        self.model_version = "unknown"

        self._normalizer = SemanticNormalizer()
        self._engineer = FeatureEngineer()

        self._preprocessor = self._load_preprocessor(Path(preprocessor_path))
        self._model = self._load_model(Path(model_path))
        self._threshold = self._load_threshold(threshold)

        logger.info(
            "ChurnPredictor pronto | device=%s | threshold=%.4f | model=%s",
            self.device,
            self._threshold,
            self.model_version,
        )

    # ── Carregamento de artefatos ─────────────────────────────────────────────

    def _load_preprocessor(self, path: Path) -> object:
        """Carrega o ColumnTransformer serializado."""
        if not path.exists():
            raise FileNotFoundError(
                f"preprocessor.pkl não encontrado: {path}\n"
                "Execute o notebook 2_vab_preprocessing antes de iniciar a API."
            )
        preprocessor = joblib.load(path)
        logger.info("preprocessor.pkl carregado | path=%s", path)
        return preprocessor

    def _load_model(self, path: Path) -> ChurnMLPInference:
        """Carrega pesos da MLP (LayerNorm) e coloca em modo eval."""
        if not path.exists():
            raise FileNotFoundError(
                f"Arquivo de modelo não encontrado: {path}\n"
                "Execute o notebook de treinamento antes de iniciar a API."
            )

        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        model = ChurnMLPInference(
            input_dim=N_FEATURES_FINAL, hidden_dims=[128, 64], dropout=0.15
        )
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        self.model_version = path.stem
        logger.info(
            "Modelo carregado | path=%s | version=%s | device=%s",
            path,
            self.model_version,
            self.device,
        )
        return model

    def _load_threshold(self, default: float) -> float:
        """Carrega threshold ótimo: Optuna > winner_model_report.json > default."""
        import json

        # 1. Arquivo gerado pelo Optuna em runtime
        optuna_files = sorted(MODELS_DIR.glob(_OPTUNA_PARAMS_GLOB))
        if optuna_files:
            latest = optuna_files[-1]
            try:
                params = json.loads(latest.read_text(encoding="utf-8"))
                threshold = float(params.get("threshold", default))
                logger.info(
                    "Threshold do Optuna | file=%s | threshold=%.4f",
                    latest.name,
                    threshold,
                )
                return threshold
            except Exception as exc:  # noqa: BLE001
                logger.warning("Falha ao ler Optuna (%s)", exc)

        # 2. Relatório do modelo vencedor (gerado pelo notebook 4)
        winner_report = PROJECT_ROOT / "reports" / "json" / "winner_model_report.json"
        if winner_report.exists():
            try:
                data = json.loads(winner_report.read_text(encoding="utf-8"))
                threshold = float(
                    data.get("test_metrics", {}).get("threshold", default)
                )
                logger.info(
                    "Threshold do winner_model_report | threshold=%.4f", threshold
                )
                return threshold
            except Exception as exc:  # noqa: BLE001
                logger.warning("Falha ao ler winner_model_report (%s)", exc)

        logger.info("Usando threshold padrão=%.4f", default)
        return default

    # ── Pipeline de predição ──────────────────────────────────────────────────

    def predict(self, request: ChurnRequest) -> ChurnResponse:
        """Executa o pipeline completo e retorna ChurnResponse.

        Etapas internas:
            1. ChurnRequest → dict → DataFrame (1 linha)
            2. Normaliza nomes de colunas para snake_case
            3. SemanticNormalizer.transform()
            4. FeatureEngineer.transform()
            5. preprocessor.pkl.transform() → array float64
            6. array → tensor float32
            7. ChurnMLPv2.forward() → logit
            8. sigmoid(logit) → probability
            9. probability + threshold → ChurnResponse

        Args:
            request: payload validado pelo Pydantic

        Returns:
            ChurnResponse com probabilidade, label e estimativa de custo

        Raises:
            ValueError: se o array transformado tiver shape inesperado
        """
        # ── [1] ChurnRequest → DataFrame ──────────────────────────────────────
        raw_dict = request.model_dump()
        df = pd.DataFrame([raw_dict])

        logger.info(
            "predict() | input recebido | n_colunas=%d | colunas=%s",
            len(df.columns),
            list(df.columns),
        )

        # ── [2] Normalizar nomes para snake_case ──────────────────────────────
        # Garante compatibilidade com o preprocessor (treinado em snake_case)
        df.columns = [to_snake_case(col) for col in df.columns]

        # ── [2b] Normalizar "No phone service" → "No" para multiple_lines ──────
        # O OrdinalEncoder foi treinado com categorias ["No", "Yes"].
        # "No phone service" seria codificado como -1 (unknown_value), gerando
        # predições fora da distribuição de treino.
        if "multiple_lines" in df.columns:
            df["multiple_lines"] = df["multiple_lines"].replace(
                {"No phone service": "No"}
            )

        # ── [3] SemanticNormalizer ────────────────────────────────────────────
        # "No internet service" → "No" nas 6 colunas de serviço
        # Corrige inconsistências lógicas (internet="No" mas serviço ativo)
        df = self._normalizer.transform(df)

        # ── [4] FeatureEngineer ───────────────────────────────────────────────
        # Cria as 6 features derivadas:
        # num_services, charges_per_month, is_month_to_month,
        # tenure_group, has_security_support, is_fiber_optic
        df = self._engineer.transform(df)

        logger.info(
            "predict() | pós-FE | shape=%s | novas_features_presentes=%s",
            df.shape,
            all(
                f in df.columns
                for f in [
                    "num_services",
                    "charges_per_month",
                    "is_month_to_month",
                    "tenure_group",
                    "has_security_support",
                    "is_fiber_optic",
                ]
            ),
        )

        # ── [5] ColumnTransformer (preprocessor.pkl) ─────────────────────────
        # Aplica: log1p+StandardScaler (num), OrdinalEncoder (bin),
        #         OneHotEncoder (ohe), passthrough (pass)
        # Saída: array float64 shape (1, 30)
        X_transformed = self._preprocessor.transform(df)

        if X_transformed.shape[1] != N_FEATURES_FINAL:
            raise ValueError(
                f"Shape inesperado após preprocessamento: "
                f"{X_transformed.shape} — esperado (1, {N_FEATURES_FINAL})"
            )

        logger.info(
            "predict() | pós-preprocessor | shape=%s | dtype=%s",
            X_transformed.shape,
            X_transformed.dtype,
        )

        # ── [6] Array → tensor float32 ───────────────────────────────────────
        # ColumnTransformer pode retornar scipy.sparse quando OHE representa
        # >30% das features (sparse_threshold=0.3 padrão). torch.from_numpy
        # exige ndarray denso — convertemos antes.
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()

        tensor = torch.from_numpy(X_transformed.astype(np.float32)).to(
            self.device
        )  # shape (1, 30)

        # ── [7+8] Forward pass + sigmoid → probabilidade ─────────────────────
        with torch.no_grad():
            logit = self._model(tensor)  # shape (1, 1)
            prob = torch.sigmoid(logit).item()  # float escalar ∈ [0, 1]

        # ── [9] Threshold → label + custo estimado ───────────────────────────
        is_churn = prob >= self._threshold
        label = "churn" if is_churn else "no_churn"

        # Custo estimado:
        # - Predição positiva (churn detectado): custo da campanha de retenção
        # - Predição negativa: 0.0 (nenhuma ação tomada)
        cost_estimate = COST_FP if is_churn else 0.0

        logger.info(
            "predict() | prob=%.4f | threshold=%.4f | label=%s | cost_brl=%.2f",
            prob,
            self._threshold,
            label,
            cost_estimate,
        )

        return ChurnResponse(
            churn_probability=round(prob, 6),
            churn_label=label,
            threshold_used=self._threshold,
            cost_estimate_brl=cost_estimate,
            model_version=self.model_version,
        )
