# Problem Statement
Heart failure is one of the biggest causes of death in the world right now. The main problem doctors face is that predicting heart failure requires looking at many different health factors at the same time—like age, blood pressure, sodium levels, and more. Doing this manually is slow and sometimes hard to get right.

I wanted to solve this by building a tool that uses Machine Learning to analyze these patterns automatically. This system helps by giving a quick "Risk" or "Safe" prediction based on patient data, which can act as a second opinion for medical staff.

# Scope of the Project
This project is focused on the **Heart Failure Clinical Records** dataset.
* **What it does:** It takes 12 specific health inputs (like Age, Diabetes status, Creatinine levels) and predicts the likelihood of a "Death Event".
* **Limitations:** The model is trained on a specific dataset of 299 records. It is meant for educational and testing purposes, not to replace a real doctor.
* **Platform:** It runs as a desktop command-line application using Python.

# Target Users
* **Doctors/Nurses:** To get a quick estimation of patient risk during checkups.
* **Medical Students:** To learn how different biological factors (like high CPK or low ejection fraction) contribute to heart risks.
* **Data Researchers:** To analyze patterns in clinical records.

# High-Level Features
1.  **Prediction System:** You can enter a patient's details and get an instant result telling you if they are at "High Risk" or "Stable."
2.  **Model Training:** The system allows you to retrain the AI model if you have a new dataset file.
3.  **Data Saving:** It automatically saves the trained model and the scaler tools so you don't have to wait for training every time you open the app.
4.  **Error Handling:** If you type text instead of numbers (like typing "sixty" instead of "60"), the app won't crash; it will just ask you to try again.
