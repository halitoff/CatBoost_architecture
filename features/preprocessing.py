import logging
from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Preprocess raw data for modeling."""

    def __init__(self, categorical_features: List[str], numerical_features: List[str]):
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.preprocessor = None

    def fit(self, df: pd.DataFrame) -> None:
        logger.info("Fitting feature pipeline")
        df = self._prepare_dates(df)
        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    self.categorical_features,
                ),
                (
                    "num",
                    Pipeline([
                        ("imputer", SimpleImputer()),
                        ("scaler", StandardScaler()),
                    ]),
                    self.numerical_features,
                ),
            ]
        )
        self.preprocessor.fit(df)

    def transform(self, df: pd.DataFrame):
        df = self._prepare_dates(df)
        return self.preprocessor.transform(df)

    def fit_transform(self, df: pd.DataFrame):
        self.fit(df)
        return self.transform(df)

    def _prepare_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'Период' in df.columns:
            df['Период'] = pd.to_datetime(df['Период'])
            df['день_недели'] = df['Период'].dt.dayofweek
            df['месяц'] = df['Период'].dt.month
        return df
