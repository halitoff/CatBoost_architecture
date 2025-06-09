import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error

from utils.config import ModelConfig, models_dir

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, config: ModelConfig, feature_engineer: Any):
        self.config = config
        self.feature_engineer = feature_engineer
        self.model = None

    def _next_version(self) -> int:
        path = models_dir()
        existing = list(path.glob("model_v*.cbm"))
        versions = [int(p.stem.split('_v')[1]) for p in existing if '_v' in p.stem]
        return max(versions, default=0) + 1

    def train(self, df: pd.DataFrame, target_col: str = "Количество") -> Path:
        logger.info("Starting training")
        X = df[self.config.features]
        y = df[target_col]
        X_processed = self.feature_engineer.fit_transform(X)
        train_pool = Pool(X_processed, y)
        self.model = CatBoostRegressor(
            iterations=self.config.iterations,
            depth=self.config.depth,
            loss_function=self.config.loss_function,
            verbose=False,
        )
        self.model.fit(train_pool)

        preds = self.model.predict(train_pool)
        mae = mean_absolute_error(y, preds)
        logger.info("Training MAE: %.4f", mae)

        version = self._next_version()
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
            "trained_at": datetime.utcnow().isoformat(),
            "rows": rows,
            "mae": mae,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


