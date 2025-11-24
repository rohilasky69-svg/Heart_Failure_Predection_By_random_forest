# main.py (Interactive Version)
import sys
import numpy as np
import os

# --- Checks to ensure you don't get yellow line errors ---
try:
    from src.data_loader import DataLoader
    from src.preprocessing import DataPreprocessor
    from src.model_engine import HeartFailureModel
except ImportError as e:
    print("\nCRITICAL ERROR: Could not import modules.")
    print(f"Details: {e}")
    print("Make sure you are running this from the 'HeartFailureProject' folder.")
    sys.exit()

def get_user_input():
    """
    Prompts the user for all 12 features required by the model.
    """
    print("\n--- Please enter Patient Details ---")
    try:
        age = float(input("1. Age: "))
        anaemia = int(input("2. Anaemia (0=No, 1=Yes): "))
        cpk = int(input("3. CPK Level (e.g., 582): "))
        diabetes = int(input("4. Diabetes (0=No, 1=Yes): "))
        ejection_fraction = int(input("5. Ejection Fraction (e.g., 20): "))
        hbp = int(input("6. High Blood Pressure (0=No, 1=Yes): "))
        platelets = float(input("7. Platelets (e.g., 265000): "))
        serum_creatinine = float(input("8. Serum Creatinine (e.g., 1.9): "))
        serum_sodium = int(input("9. Serum Sodium (e.g., 130): "))
        sex = int(input("10. Sex (0=Female, 1=Male): "))
        smoking = int(input("11. Smoking (0=No, 1=Yes): "))
        time = int(input("12. Follow-up Time (days, e.g., 4): "))
        
        # Return as a 2D numpy array
        return np.array([[age, anaemia, cpk, diabetes, ejection_fraction, 
                          hbp, platelets, serum_creatinine, serum_sodium, 
                          sex, smoking, time]])
    except ValueError:
        print("\nError: Please enter valid numbers only.")
        return None

def run_training_pipeline():
    print("\n--- Starting Training Pipeline ---")
    try:
        loader = DataLoader()
        df = loader.load_data()
        
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess_for_training(df)
        
        engine = HeartFailureModel()
        engine.train(X_train, y_train)
        engine.evaluate(X_test, y_test)
        engine.save_model()
        print("--- Training Complete ---")
    except Exception as e:
        print(f"Error during training: {e}")

def run_prediction_pipeline():
    print("\n--- Starting Prediction System ---")
    
    # 1. Load Pre-trained Model
    engine = HeartFailureModel()
    engine.load_model()
    
    # 2. Get Input from User
    user_input = get_user_input()
    
    if user_input is not None:
        # 3. Process Input
        preprocessor = DataPreprocessor()
        processed_input = preprocessor.preprocess_single_input(user_input)
        
        # 4. Predict
        prediction = engine.model.predict(processed_input)
        probability = engine.model.predict_proba(processed_input)
        
        result = "HIGH RISK of Heart Failure" if prediction[0] == 1 else "Patient is Stable"
        confidence = np.max(probability) * 100
        
        print(f"\n-----------------------------")
        print(f"PREDICTION RESULT: {result}")
        print(f"CONFIDENCE: {confidence:.2f}%")
        print(f"-----------------------------")

if __name__ == "__main__":
    while True:
        print("\n=== HEART FAILURE PREDICTION SYSTEM ===")
        print("1. Train New Model (Reset System)")
        print("2. Predict (Enter Patient Data)")
        print("3. Exit")
        choice = input("Select option: ")
        
        if choice == '1':
            run_training_pipeline()
        elif choice == '2':
            run_prediction_pipeline()
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice.")