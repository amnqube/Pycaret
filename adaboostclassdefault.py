# ================================================================
# 📌 AdaBoost Classification with Default Hyperparameters
# Dataset: Kaka_classification.csv
# Columns:
#   learninghrs (numeric) - e.g., 5
#   hwdone (categorical Yes/No)
#   classatend (categorical Yes/No)
#   pass (target: Pass/Fail or 1/0)
# ================================================================

from pycaret.classification import *
import pandas as pd

# Load dataset
data = pd.read_csv("Kaka_classification.csv")

# Setup PyCaret classification environment
clf = setup(
    data=data,
    target='pass',  # Target is categorical: Pass/Fail — suitable for classification
    categorical_features=['hwdone', 'classatend'],
    session_id=123,
    silent=True,
    verbose=False
)

# Create AdaBoost classifier with default hyperparameters
ada_default = create_model('ada')

# Evaluate model performance
evaluate_model(ada_default)

# Save model to disk
save_model(ada_default, 'ada_classification_default')

# Input source for new prediction data
source = input("Source of input (csv/self): ").strip().lower()

if source == 'csv':
    filename = input("Enter CSV file name (in same folder): ")
    new_data = pd.read_csv(filename)
    preds = predict_model(ada_default, data=new_data)
    print("\nPredictions from CSV data:\n", preds)

elif source == 'self':
    learninghrs = int(input("Enter number of learning hours: "))
    hwdone = input("Homework done? (Yes/No): ").strip().capitalize()
    classatend = input("Class attended? (Yes/No): ").strip().capitalize()
    manual_data = pd.DataFrame({
        'learninghrs': [learninghrs],
        'hwdone': [hwdone],
        'classatend': [classatend]
    })
    preds = predict_model(ada_default, data=manual_data)
    print("\nPrediction from manual input:\n", preds)

else:
    print("Invalid input source.")