# src/data_loader.py
import pandas as pd
import logging
from src.config import DATA_PATH

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Responsible for loading data safely from the source.
    """
    def load_data(self):
        try:
            logger.info(f"Attempting to load data from {DATA_PATH}")
            df = pd.read_csv(DATA_PATH)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            logger.error("Dataset not found. Please check the path in config.py")
            raise
        except Exception as e:
            logger.error(f"An error occurred while loading data: {e}")
            raise