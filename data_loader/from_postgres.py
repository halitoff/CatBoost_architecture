import logging
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine


logger = logging.getLogger(__name__)


class PostgresLoader:
    """Load data from a PostgreSQL table."""

    def __init__(self, url: str):
        self.url = url
        self.engine = None

    def connect(self) -> None:
        logger.info("Connecting to PostgreSQL")
        self.engine = create_engine(self.url)

    def load(self, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Load table into pandas DataFrame."""
        if self.engine is None:
            self.connect()
        query = f"SELECT * FROM {table_name}"
        if limit:
            query += f" LIMIT {limit}"
        logger.info("Executing query: %s", query)
        df = pd.read_sql(query, self.engine)
        return df

