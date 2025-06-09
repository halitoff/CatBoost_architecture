import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from catboost import CatBoostRegressor, Pool

from utils.config import models_dir, ModelConfig

logger = logging.getLogger(__name__)


def _latest_model() -> Path:
    path = models_dir()
    models = sorted(path.glob("model_v*.cbm"))
    return models[-1] if models else None


class Predictor:
    def __init__(self, config: ModelConfig, feature_engineer: Any, version: Optional[int] = None):
        self.config = config
        self.feature_engineer = feature_engineer
        self.model = CatBoostRegressor()
        self.model_path = self._resolve_model_path(version)
        self.model.load_model(self.model_path)

    def _resolve_model_path(self, version: Optional[int]) -> Path:
        if version is None:
            path = _latest_model()
            if path is None:
                raise FileNotFoundError("No model found")
            return path
        return models_dir() / f"model_v{version}.cbm"

    def predict(self, data: pd.DataFrame) -> pd.Series:
        X = data[self.config.features]
        X_processed = self.feature_engineer.transform(X)
        pool = Pool(X_processed)
        preds = self.model.predict(pool)
        return pd.Series(preds)

