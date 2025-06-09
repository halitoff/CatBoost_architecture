import logging
from pathlib import Path

import pandas as pd

from .train import ModelTrainer
from utils.config import ModelConfig

logger = logging.getLogger(__name__)


class ModelReTrainer(ModelTrainer):
    """Retrain model from scratch with new version."""

    def retrain(self, df: pd.DataFrame, target_col: str = "Количество") -> Path:
        logger.info("Retraining model from scratch")
        return self.train(df, target_col)

