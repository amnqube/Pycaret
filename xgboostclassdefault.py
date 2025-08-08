# =========================================================
# 📌 1. CLASSIFICATION with XGBoost (Default Parameters)
# Dataset: Kaka.csv
# Goal: Predict whether student will Pass or Fail
# =========================================================

# 1️⃣ Install PyCaret if not installed
# (Run only once in your environment, skip if already installed)
# pip install pycaret

# 2️⃣ Importing required libraries
from pycaret.classification import *   # For classification tasks in PyCaret
import pandas as pd                    # For reading and handling CSV files

# 3️⃣ Load the dataset
# Assumes 'Kaka.csv' is in the same folder as this script
data = pd.read_csv("Kaka.csv")

# Example: Kaka.csv
# learninghrs,hwdone,classatend,pass
# 5,Yes,Yes,Pass
# 2,No,No,Fail

# 4️⃣ Initialize PyCaret setup for classification
clf = setup(
    data=data,                        # Our input dataset
    target='pass',                    # The column we want to predict
    categorical_features=['hwdone', 'classatend'],  # These are Yes/No categorical values
    session_id=123,                    # Random seed for reproducibility
    silent=True,                       # Skip asking for confirmations
    verbose=False                      # Hide setup summary
)

# 5️⃣ Create XGBoost model with DEFAULT hyperparameters
xgb_model = create_model('xgboost')

# 6️⃣ Evaluate the model interactively (metrics, plots, etc.)
evaluate_model(xgb_model)

# 7️⃣ Save the trained model for later use
save_model(xgb_model, 'xgb_classification_default')

# 8️⃣ Prediction for NEW data
# Ask user where input is coming from
source = input("Source of input (csv/self): ").strip().lower()

if source == 'csv':
    filename = input("Enter CSV file name (with .csv extension): ")
    new_data = pd.read_csv(filename)
    preds = predict_model(xgb_model, data=new_data)
    print("\n📌 Predictions from CSV file:")
    print(preds)

elif source == 'self':
    # Ask for values manually
    learninghrs = int(input("Enter number of learning hours: "))
    hwdone = input("Homework done? (Yes/No): ").strip().capitalize()
    classatend = input("Class attended? (Yes/No): ").strip().capitalize()

    # Create a DataFrame for prediction
    manual_data = pd.DataFrame({
        'learninghrs': [learninghrs],
        'hwdone': [hwdone],
        'classatend': [classatend]
    })

    preds = predict_model(xgb_model, data=manual_data)
    print("\n📌 Prediction for manual input:")
    print(preds)

else:
    print("❌ Invalid input source. Please enter 'csv' or 'self'.")