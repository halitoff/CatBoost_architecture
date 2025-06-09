import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ExcelLoader:
    """Load data from Excel files."""

    def __init__(self, path: str):
        self.path = Path(path)

    def load(self, sheet_name: Optional[str] = None) -> pd.DataFrame:
        logger.info("Loading data from Excel: %s", self.path)
        df = pd.read_excel(self.path, sheet_name=sheet_name, encoding="cp1251")
        return df

