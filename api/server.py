import logging
from typing import Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from data_loader.from_postgres import PostgresLoader
from data_loader.from_excel import ExcelLoader
from features.preprocessing import FeatureEngineer
from model.train import ModelTrainer
from model.finetune import ModelFineTuner
from model.predictor import Predictor
from utils.config import ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Default config and feature lists
CONFIG = ModelConfig(
    features=[
        'Период', 'Номенклатура', 'ВидНоменклатуры', 'Поставщик',
        'Производитель', 'Вес', 'Артикул', 'Код', 'Группа', 'Сумма',
        'Срок годности (час)', 'Наличие товара', 'Наличие товара в магазине',
        'Категория товара', 'Задержка поставки (дн)', 'Адрес точки',
        'Заканчивался ли продукт', 'Цена на полке', 'Часов работала точка'
    ]
)

CATEGORICAL = [
    'Номенклатура', 'ВидНоменклатуры', 'Поставщик', 'Производитель',
    'Артикул', 'Код', 'Группа', 'Наличие товара', 'Наличие товара в магазине',
    'Категория товара', 'Адрес точки', 'Заканчивался ли продукт'
]
NUMERICAL = ['Вес', 'Сумма', 'Срок годности (час)', 'Задержка поставки (дн)',
             'Цена на полке', 'Часов работала точка']

feature_engineer = FeatureEngineer(CATEGORICAL, NUMERICAL)
trainer = ModelTrainer(CONFIG, feature_engineer)
fine_tuner = ModelFineTuner(CONFIG, feature_engineer)

# Cache predictor instance
_predictor: Optional[Predictor] = None


class TrainRequest(BaseModel):
    source: str  # 'postgres' or 'excel'
    path: str
    table: Optional[str] = None


class PredictRequest(BaseModel):
    features: dict
    version: Optional[int] = None


@app.post('/train')
def train(req: TrainRequest):
    """Train model from scratch."""
    if req.source == 'postgres':
        loader = PostgresLoader(req.path)
        df = loader.load(req.table or '')
    else:
        loader = ExcelLoader(req.path)
        df = loader.load()
    model_path = trainer.train(df)
    global _predictor
    _predictor = None  # reset predictor cache
    return {'model_path': str(model_path)}


@app.post('/finetune')
def finetune(req: TrainRequest):
    """Finetune latest model with new data."""
    if req.source == 'postgres':
        loader = PostgresLoader(req.path)
        df = loader.load(req.table or '')
    else:
        loader = ExcelLoader(req.path)
        df = loader.load()
    model_path = fine_tuner.finetune(df)
    global _predictor
    _predictor = None
    return {'model_path': str(model_path)}


@app.post('/predict')
def predict(req: PredictRequest):
    """Predict quantity for provided features."""
    global _predictor
    if _predictor is None:
        _predictor = Predictor(CONFIG, feature_engineer, version=req.version)
    df = pd.DataFrame([req.features])
    pred = _predictor.predict(df)
    return {'prediction': float(pred.iloc[0])}


@app.get('/')
def health():
    return {'status': 'ok'}
