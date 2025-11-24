# src/model_engine.py
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.config import MODEL_PATH

class HeartFailureModel:
    def __init__(self):
        # Using Random Forest as it is robust for clinical records
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, X_train, y_train):
        """Trains the model."""
        print("Training the model...")
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """Evaluates performance."""
        predictions = self.model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        print(f"Model Accuracy: {acc * 100:.2f}%")
        return acc

    def save_model(self):
        """Saves the model to a file (Creating the Pre-trained Model)."""
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {MODEL_PATH}")

    def load_model(self):
        """Loads the pre-trained model from file."""
        try:
            with open(MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
            print("Pre-trained model loaded successfully.")
        except FileNotFoundError:
            print("Error: Pre-trained model not found. Run training first.")