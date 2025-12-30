import pandas as pd
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class RainfallDataService:

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.load_data()

    def load_data(self):
        try:
            self.df = pd.read_csv(self.data_path)
            self.df['date'] = pd.to_datetime(self.df['date'], format='mixed')
            self.df['year'] = self.df['date'].dt.year
            logger.info(f"Successfully loaded {len(self.df)} rainfall records")
        except Exception as e:
            logger.error(f"Error loading rainfall data: {e}")
            raise

    def get_available_years(self) -> List[int]:
        if self.df is None:
            return []
        return sorted(self.df['year'].unique().tolist())

    def get_data_by_year(self, year: int) -> Optional[pd.DataFrame]:
        if self.df is None:
            return None

        year_data = self.df[self.df['year'] == year].sort_values('date')
        logger.info(f"Retrieved {len(year_data)} records for year {year}")
        return year_data

    def get_statistics(self, year: Optional[int] = None) -> dict:
        if self.df is None:
            return {}

        data = self.df if year is None else self.df[self.df['year'] == year]

        if data.empty:
            return {}

        stats = {
            'total_records': len(data),
            'total_rainfall': float(data['rainfall'].sum()),
            'average_rainfall': float(data['rainfall'].mean()),
            'max_rainfall': float(data['rainfall'].max()),
            'min_rainfall': float(data['rainfall'].min()),
            'max_date': data.loc[data['rainfall'].idxmax(), 'date'].strftime('%Y-%m-%d'),
            'min_date': data.loc[data['rainfall'].idxmin(), 'date'].strftime('%Y-%m-%d')
        }

        return stats
