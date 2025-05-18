from typing import Dict, List, Any
import pandas as pd
from pathlib import Path
import logging

class DataProcessor:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.logger = logging.getLogger(__name__)
        
    def load_dataset(self, file_name: str) -> bool:
        """Load and preprocess a dataset from a CSV file."""
        try:
            file_path = self.data_dir / file_name
            if not file_path.exists():
                self.logger.error(f"File not found: {file_path}")
                return False
            
            # Load the dataset
            dataset_name = file_name.rsplit('.', 1)[0]
            df = pd.read_csv(file_path)
            
            # Preprocess the data
            df = self._preprocess_data(df)
            
            # Store the preprocessed dataset
            self.datasets[dataset_name] = df
            self.logger.info(f"Successfully loaded and preprocessed dataset: {dataset_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading dataset: {str(e)}")
            return False
            
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data with necessary transformations."""
        try:
            # Convert timestamp to datetime if it exists
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Calculate net weight if required columns exist
            if 'total weight' in df.columns and 'vessel weight' in df.columns:
                df['net_weight'] = df['total weight'] - df['vessel weight']
            
            # Create time-based features if timestamp exists
            if 'timestamp' in df.columns:
                df['hour'] = df['timestamp'].dt.hour
                df['date'] = df['timestamp'].dt.date
                df['day_of_week'] = df['timestamp'].dt.day_name()
            
            # Ensure all numeric columns are float
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_columns:
                df[col] = df[col].astype(float)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {str(e)}")
            return df  # Return original dataframe if preprocessing fails
    
    def list_datasets(self) -> List[str]:
        """List all loaded datasets."""
        return list(self.datasets.keys())
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about a specific dataset."""
        try:
            if dataset_name not in self.datasets:
                return {"error": f"Dataset {dataset_name} not found"}
            
            df = self.datasets[dataset_name]
            
            # Convert dtypes to strings for JSON serialization
            dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
            
            return {
                "columns": list(df.columns),
                "shape": df.shape,
                "dtypes": dtypes,
                "summary": df.describe().to_dict()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting dataset info: {str(e)}")
            return {"error": str(e)}
    
    def get_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Get a specific dataset."""
        return self.datasets.get(dataset_name) 