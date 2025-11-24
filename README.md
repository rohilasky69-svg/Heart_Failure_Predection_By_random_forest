Heart Failure Prediction System

Overview

This is a Machine Learning project I built for my "Build Your Own Project" submission. Basically, the goal is to predict if a patient is at risk of heart failure using their medical records.

I chose this topic because heart failure is a huge issue globally, and doctors often have to look at a ton of different data points (like age, blood pressure, sodium levels) to figure it out. I wanted to see if I could train a model to do this automatically. I used a dataset of about 299 patients and trained a Random Forest model to classify them as either "High Risk" or "Stable."

Features

Simple Menu: When you run the app, you get a simple command-line menu to choose between training or predicting.

Data Cleaning: The code automatically cleans up the data and scales the numbers. For example, it makes sure Age (which is small) and Platelets (which is huge) are on the same scale so the model doesn't get confused.

Save & Load: I didn't want to retrain the model every single time. So, the system saves the trained model to a file (.pkl), and it loads instantly when you run it next time.

Error Handling: I added some checks so if you accidentally type text instead of a number, the program won't just crash. It asks you to try again.

Technologies Used

Python 3: The main language I used.

Pandas: To load the CSV file and handle the data tables.

Scikit-Learn: This does the heavy lifting for the Machine Learning (Random Forest) and the scaling.

Pickle: I used this to save the trained model to the disk.

Numpy: For some of the math operations.

Steps to Install & Run

Here is how you can get this running on your laptop:

Download the Code:
Just download this folder to your computer.

Set up the Environment:
Open your terminal in this folder. It's best to create a virtual environment so things don't get messy:

python -m venv venv


Activate it:

Windows: .\venv\Scripts\activate

Mac/Linux: source venv/bin/activate

Install Libraries:
You need a few libraries like Pandas. Run this:

pip install -r requirements.txt


Run the App:
Start the main script:

python main.py


Instructions for Testing

Once the app is running, follow these steps to see if it works:

Train the Model First

On the menu, type 1 and hit Enter.

You should see a message saying "Loading Data" and then "Success".

This creates a file inside the models/ folder. You only need to do this once.

Try a Prediction

Run the app again (or pick option 2).

It will ask for patient details. You can try these inputs to check accuracy:

Test Case 1 (High Risk): Age: 75, CPK: 582, Ejection Fraction: 20, Creatinine: 1.9. -> Result should be HIGH RISK.

Test Case 2 (Stable): Age: 45, Ejection Fraction: 40, Creatinine: 1.0. -> Result should be STABLE.