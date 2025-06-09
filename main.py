import logging
from argparse import ArgumentParser

import pandas as pd

from data_loader.from_postgres import PostgresLoader
from data_loader.from_excel import ExcelLoader
from features.preprocessing import FeatureEngineer
from model.train import ModelTrainer
from model.predictor import Predictor
from utils.config import ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser()
    parser.add_argument("--source", choices=["postgres", "excel"], required=True)
    parser.add_argument("--path")
    parser.add_argument("--table")
    args = parser.parse_args()

    if args.source == "postgres":
        loader = PostgresLoader(args.path)
        df = loader.load(args.table)
    else:
        loader = ExcelLoader(args.path)
        df = loader.load()

    config = ModelConfig(features=[
        'Период', 'Номенклатура', 'ВидНоменклатуры', 'Поставщик',
        'Производитель', 'Вес', 'Артикул', 'Код', 'Группа', 'Сумма',
        'Срок годности (час)', 'Наличие товара', 'Наличие товара в магазине',
        'Категория товара', 'Задержка поставки (дн)', 'Адрес точки',
        'Заканчивался ли продукт', 'Цена на полке', 'Часов работала точка'
    ])

    categorical = [
        'Номенклатура', 'ВидНоменклатуры', 'Поставщик', 'Производитель',
        'Артикул', 'Код', 'Группа', 'Наличие товара', 'Наличие товара в магазине',
        'Категория товара', 'Адрес точки', 'Заканчивался ли продукт'
    ]
    numerical = ['Вес', 'Сумма', 'Срок годности (час)', 'Задержка поставки (дн)', 'Цена на полке', 'Часов работала точка']

    fe = FeatureEngineer(categorical, numerical)
    trainer = ModelTrainer(config, fe)
    trainer.train(df)

    predictor = Predictor(config, fe)
    sample = df.head(1)
    pred = predictor.predict(sample)
    logger.info("Prediction for sample: %s", pred.iloc[0])


if __name__ == "__main__":
    main()

