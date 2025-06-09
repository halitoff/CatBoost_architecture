from pathlib import Path
from typing import List
from pydantic import BaseModel


class ModelConfig(BaseModel):
    iterations: int = 500
    depth: int = 6
    loss_function: str = "MAE"
    features: List[str] = []


def models_dir() -> Path:
    path = Path("models")
    path.mkdir(exist_ok=True)
    return path

