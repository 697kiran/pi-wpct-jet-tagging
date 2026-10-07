"""Gradient Boosted Trees (GBT) backbone wrapped to the uniform model interface.

Supports XGBoost and LightGBM backends. Operates on high-level jet features
and outputs 3-class logits/scores as PyTorch tensors.
"""

import os
import pickle
from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn as nn


class GBTModel(nn.Module):
    """Wrapper around XGBoost/LightGBM providing PyTorch-compatible interface."""

    def __init__(self, backend: str = "xgboost", params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.backend = backend.lower()
        self.params = params or {}
        self.model: Any = None
        self.is_fitted: bool = False
        # Dummy parameter to make torch.nn.Module happy and hold device
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)

    def fit_from_arrays(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Fit the gradient boosted decision trees."""
        if self.backend == "xgboost":
            import xgboost as xgb
            default_params = {
                "n_estimators": 150,
                "max_depth": 5,
                "learning_rate": 0.1,
                "objective": "multi:softprob",
                "num_class": 3,
                "random_state": 42,
                "verbosity": 0,
            }
            default_params.update(self.params)
            self.model = xgb.XGBClassifier(**default_params)
            eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
            self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        elif self.backend == "lightgbm":
            import lightgbm as lgb
            default_params = {
                "n_estimators": 150,
                "max_depth": 5,
                "learning_rate": 0.1,
                "objective": "multiclass",
                "num_class": 3,
                "random_state": 42,
                "verbose": -1,
            }
            default_params.update(self.params)
            self.model = lgb.LGBMClassifier(**default_params)
            eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
            self.model.fit(X_train, y_train, eval_set=eval_set)
        else:
            raise ValueError(f"Unsupported GBT backend: {self.backend}")

        self.is_fitted = True

    def predict_logits(self, X: np.ndarray) -> torch.Tensor:
        """Predict unnormalized or pseudo-logits from tree probabilities."""
        if not self.is_fitted or self.model is None:
            # Fallback uniform pseudo-logits
            n = X.shape[0]
            return torch.zeros((n, 3), device=self.dummy_param.device)

        probs = self.model.predict_proba(X)  # [N, 3]
        probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
        # Convert probabilities to logits: log(p)
        logits = np.log(probs)
        return torch.tensor(logits, dtype=torch.float32, device=self.dummy_param.device)

    def forward(
        self,
        features: torch.Tensor,
        points: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        vectors: Optional[torch.Tensor] = None,
        high_level: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with uniform backbone interface."""
        if high_level is not None:
            X = high_level.detach().cpu().numpy()
        else:
            # Flatten constituent features as emergency fallback
            B = features.shape[0]
            X = features.view(B, -1).detach().cpu().numpy()

        return self.predict_logits(X)

    def save_model(self, path: str) -> None:
        """Save fitted tree model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"backend": self.backend, "model": self.model, "params": self.params}, f)

    def load_model(self, path: str) -> None:
        """Load fitted tree model."""
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.backend = data["backend"]
            self.model = data["model"]
            self.params = data["params"]
            self.is_fitted = True
