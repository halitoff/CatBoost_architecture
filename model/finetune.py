import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error

from utils.config import ModelConfig, models_dir

logger = logging.getLogger(__name__)


def _latest_model() -> Path:
    path = models_dir()
    models = sorted(path.glob("model_v*.cbm"))
    return models[-1] if models else None


class ModelFineTuner:
    """Fine tune existing model with new data."""

    def __init__(self, config: ModelConfig, feature_engineer: Any):
        self.config = config
        self.feature_engineer = feature_engineer
        self.model = CatBoostRegressor()

    def finetune(self, df: pd.DataFrame, target_col: str = "Количество") -> Path:
        logger.info("Fine tuning latest model")
        latest = _latest_model()
        if latest is None:
            raise FileNotFoundError("No model found to finetune")
        self.model.load_model(latest)

        X = df[self.config.features]
        y = df[target_col]
        X_processed = self.feature_engineer.fit_transform(X)
        pool = Pool(X_processed, y)
        self.model.fit(pool, init_model=latest, verbose=False)

        preds = self.model.predict(pool)
        mae = mean_absolute_error(y, preds)

        version = int(latest.stem.split('_v')[1]) + 1
        model_path = models_dir() / f"model_v{version}.cbm"
        self.model.save_model(model_path)
        self._update_metadata(df.shape[0], version, mae)
        return model_path

    def _update_metadata(self, rows: int, version: int, mae: float) -> None:
        meta_path = models_dir() / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        else:
            meta = {}
        meta[version] = {
            "finetuned_at": datetime.utcnow().isoformat(),
            "rows": rows,
            "mae": mae,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

