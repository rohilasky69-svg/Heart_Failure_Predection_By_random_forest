# src/config.py
import os

# 1. Define the base directory of your project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. Define where the data and models are located
# This fixes the "cannot import DATA_PATH" error
DATA_PATH = os.path.join(BASE_DIR, 'data', 'heart_failure_clinical_records_dataset.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'heart_failure_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

# 3. Model Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = 'DEATH_EVENT'