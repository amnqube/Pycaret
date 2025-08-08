# ============================================================
# 📌 MULTIVARIATE LINEAR REGRESSION with PyCaret (Default)
# Dataset: Kaka_regression.csv
# Features: learninghrs (numeric), hwdone (Yes/No), classatend (Yes/No)
# Target: score (numeric)
# ============================================================

from pycaret.regression import *
import pandas as pd

# Load data
data = pd.read_csv("Kaka_regression.csv")

# Setup regression environment
reg = setup(
    data=data,
    target='score',                   # Numeric target column
    categorical_features=['hwdone', 'classatend'],  # Yes/No encoded automatically
    session_id=123,
    silent=True,
    verbose=False
)

# Create linear regression model (default hyperparameters)
lr_model = create_model('lr')

# Evaluate model interactively
evaluate_model(lr_model)

# Save the trained model
save_model(lr_model, 'linear_regression_default')

# Predict new data input
source = input("Source of input (csv/self): ").strip().lower()

if source == 'csv':
    filename = input("Enter CSV file name: ")
    new_data = pd.read_csv(filename)
    preds = predict_model(lr_model, data=new_data)
    print("\nPredictions from CSV:\n", preds)

elif source == 'self':
    learninghrs = float(input("Enter learning hours (numeric): "))
    hwdone = input("Homework done? (Yes/No): ").strip().capitalize()
    classatend = input("Class attended? (Yes/No): ").strip().capitalize()
    manual_data = pd.DataFrame({
        'learninghrs': [learninghrs],
        'hwdone': [hwdone],
        'classatend': [classatend]
    })
    preds = predict_model(lr_model, data=manual_data)
    print("\nPrediction from manual input:\n", preds)

else:
    print("Invalid input source.")