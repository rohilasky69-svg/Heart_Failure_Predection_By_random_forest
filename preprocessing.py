# src/preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
from src.config import TARGET_COLUMN, TEST_SIZE, RANDOM_STATE, SCALER_PATH

class DataPreprocessor:
    """
    Handles feature scaling and splitting.
    """
    def __init__(self):
        self.scaler = StandardScaler()

    def preprocess_for_training(self, df):
        """Splits data and saves the scaler for later use."""
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        # Fit and save scaler (essential for the pre-trained model to work later)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Save the scaler so we can use it for predictions later
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        return X_train_scaled, X_test_scaled, y_train, y_test

    def preprocess_single_input(self, input_data):
        """Loads the saved scaler to transform a single new user input."""
        with open(SCALER_PATH, 'rb') as f:
            loaded_scaler = pickle.load(f)
        
        # input_data must be a 2D array
        return loaded_scaler.transform(input_data)