# =============================================================
# 📌 Logistic Regression Classification (Default Hyperparameters)
# Dataset: Kaka_classification.csv
# Target: pass (Pass/Fail or Yes/No)
# Features: learninghrs, hwdone, classatend
# =============================================================

from pycaret.classification import *
import pandas as pd

# Load the dataset
data = pd.read_csv("Kaka_classification.csv")

# Setup PyCaret classification environment
clf = setup(
    data=data,
    target='pass',                       # Binary target column
    categorical_features=['hwdone', 'classatend'], # Categorical inputs
    session_id=123,
    silent=True,
    verbose=False
)

# Create logistic regression model with default parameters
logreg_default = create_model('lr')

# Evaluate the model (confusion matrix, ROC curve, etc.)
evaluate_model(logreg_default)

# Save the trained model for future use
save_model(logreg_default, 'logistic_regression_default')

# Input for new predictions
source = input("Source of input (csv/self): ").strip().lower()

if source == 'csv':
    filename = input("Enter CSV filename (with .csv): ")
    new_data = pd.read_csv(filename)
    preds = predict_model(logreg_default, data=new_data)
    print("\nPredictions from CSV input:")
    print(preds)

elif source == 'self':
    learninghrs = float(input("Enter number of learning hours: "))
    hwdone = input("Homework done? (Yes/No): ").strip().capitalize()
    classatend = input("Class attended? (Yes/No): ").strip().capitalize()

    manual_data = pd.DataFrame({
        'learninghrs': [learninghrs],
        'hwdone': [hwdone],
        'classatend': [classatend]
    })

    preds = predict_model(logreg_default, data=manual_data)
    print("\nPrediction from manual input:")
    print(preds)

else:
    print("Invalid input source. Please enter 'csv' or 'self'.")