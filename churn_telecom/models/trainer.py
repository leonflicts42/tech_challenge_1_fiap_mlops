"""Training loop with early stopping, batching and MLflow tracking."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ── Configuração ─────────────────────────────────────────────────────────────


@dataclass
class TrainerConfig:
    """Hiperparâmetros e controles de treinamento."""

    lr: float = 1e-3
    epochs: int = 100
    batch_size: int = 256
    patience: int = 10
    min_delta: float = 1e-4
    weight_decay: float = 1e-4
    device: str = "cpu"
    seed: int = 42
    checkpoint_dir: Path = Path("models/checkpoints")
    pos_weight: Optional[float] = None


@dataclass
class TrainHistory:
    """Histórico de métricas por época."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_auc: list[float] = field(default_factory=list)
    val_auc: list[float] = field(default_factory=list)
    best_epoch: int = 0
    stopped_early: bool = False


# ── Utilidades ────────────────────────────────────────────────────────────────


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_tensor(array: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(array, dtype=dtype)


def _compute_auc(logits: torch.Tensor, targets: torch.Tensor) -> float:
    from sklearn.metrics import roc_auc_score

    probs = torch.sigmoid(logits).detach().cpu().numpy()
    labels = targets.detach().cpu().numpy()
    try:
        return float(roc_auc_score(labels, probs))
    except ValueError:
        return float("nan")


# ── Early Stopping ────────────────────────────────────────────────────────────


class EarlyStopping:
    """Para o treinamento quando val_loss não melhora por `patience` épocas."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._best: float = float("inf")
        self._counter: int = 0
        self._best_state: Optional[dict] = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Retorna True se o treinamento deve ser interrompido."""
        if val_loss < self._best - self.min_delta:
            self._best = val_loss
            self._counter = 0
            self._best_state = {k: v.clone() for k, v in model.state_dict().items()}
            logger.debug("early_stopping | melhora | val_loss=%.6f", val_loss)
        else:
            self._counter += 1
            logger.debug(
                "early_stopping | sem melhora | counter=%d/%d",
                self._counter,
                self.patience,
            )
        return self._counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        """Restaura os pesos do melhor checkpoint."""
        if self._best_state is not None:
            model.load_state_dict(self._best_state)
            logger.info("early_stopping | pesos restaurados")


# ── Trainer principal ─────────────────────────────────────────────────────────


class ChurnTrainer:
    """Loop de treinamento da ChurnMLP com early stopping e MLflow tracking."""

    def __init__(self, model: nn.Module, config: TrainerConfig) -> None:
        self.model = model
        self.cfg = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

        _set_seed(config.seed)

        pos_weight = (
            torch.tensor([config.pos_weight], device=self.device)
            if config.pos_weight is not None
            else None
        )
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        # FIX: verbose= removido — argumento descontinuado no PyTorch >= 2.2
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5
        )
        self._early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
        )
        self.history = TrainHistory()
        logger.info(
            "ChurnTrainer | device=%s | lr=%.5f | patience=%d",
            config.device,
            config.lr,
            config.patience,
        )

    def _make_loader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shuffle: bool = True,
    ) -> DataLoader:
        dataset = TensorDataset(_to_tensor(X), _to_tensor(y))
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    def _train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.train()
        total_loss, all_logits, all_targets = 0.0, [], []

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(X_batch).squeeze(1)
            loss = self.criterion(logits, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * len(y_batch)
            all_logits.append(logits.detach())
            all_targets.append(y_batch.detach())

        avg_loss = total_loss / len(loader.dataset)  # type: ignore[arg-type]
        auc = _compute_auc(torch.cat(all_logits), torch.cat(all_targets))
        return avg_loss, auc

    def _val_epoch(self, loader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss, all_logits, all_targets = 0.0, [], []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(X_batch).squeeze(1)
                loss = self.criterion(logits, y_batch)

                total_loss += loss.item() * len(y_batch)
                all_logits.append(logits)
                all_targets.append(y_batch)

        avg_loss = total_loss / len(loader.dataset)  # type: ignore[arg-type]
        auc = _compute_auc(torch.cat(all_logits), torch.cat(all_targets))
        return avg_loss, auc

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> TrainHistory:
        """Treina com early stopping e registra métricas no MLflow por época."""
        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = self._make_loader(X_val, y_val, shuffle=False)

        logger.info(
            "fit | epochs=%d | batch_size=%d | train=%d | val=%d",
            self.cfg.epochs,
            self.cfg.batch_size,
            len(X_train),
            len(X_val),
        )

        for epoch in range(1, self.cfg.epochs + 1):
            train_loss, train_auc = self._train_epoch(train_loader)
            val_loss, val_auc = self._val_epoch(val_loader)

            self.scheduler.step(val_loss)

            self.history.train_loss.append(train_loss)
            self.history.val_loss.append(val_loss)
            self.history.train_auc.append(train_auc)
            self.history.val_auc.append(val_auc)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_auc": train_auc,
                    "val_auc": val_auc,
                    "lr": self.optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )

            logger.info(
                "epoch %03d/%03d | train_loss=%.4f | val_loss=%.4f | "
                "train_auc=%.4f | val_auc=%.4f",
                epoch,
                self.cfg.epochs,
                train_loss,
                val_loss,
                train_auc,
                val_auc,
            )

            if self._early_stopping.step(val_loss, self.model):
                self.history.stopped_early = True
                self.history.best_epoch = epoch - self.cfg.patience
                logger.info(
                    "fit | early stopping na época %d | melhor época=%d",
                    epoch,
                    self.history.best_epoch,
                )
                break
        else:
            self.history.best_epoch = self.cfg.epochs

        self._early_stopping.restore_best(self.model)
        self._save_checkpoint()

        logger.info(
            "fit | concluído | best_epoch=%d | stopped_early=%s",
            self.history.best_epoch,
            self.history.stopped_early,
        )
        return self.history

    def _save_checkpoint(self) -> Path:
        self.cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.cfg.checkpoint_dir / "best_mlp.pt"
        torch.save(self.model.state_dict(), path)
        mlflow.log_artifact(str(path), artifact_path="checkpoints")
        logger.info("checkpoint | salvo em %s", path)
        return path

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna probabilidade da classe positiva (churn=1)."""
        self.model.eval()
        tensor = _to_tensor(X).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor).squeeze(1)
        return torch.sigmoid(logits).cpu().numpy()
