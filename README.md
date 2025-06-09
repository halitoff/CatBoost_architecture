# CatBoost Architecture

Пример архитектуры проекта для регрессии на CatBoost. Модель предсказывает `Количество` товара по дате и другим признакам.

## Запуск через Docker

```bash
docker build -t catboost-app .
docker run -p 8000:8000 catboost-app
```

После запуска FastAPI доступен на `http://localhost:8000`. Основные эндпоинты:
- `POST /train` – обучение модели
- `POST /finetune` – дообучение модели
- `POST /predict` – получение предсказания

