FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir fastapi uvicorn pandas catboost sqlalchemy scikit-learn pydantic openpyxl

EXPOSE 8000

CMD ["uvicorn", "api.server:app", "--host", "localhost", "--port", "8000"]
