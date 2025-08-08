# ================================================================
# 📌 AdaBoost Regression with Default Hyperparameters
# Dataset: Kaka_regression.csv
# Columns:
#   learninghrs (numeric)
#   hwdone (categorical Yes/No)
#   classatend (categorical Yes/No)
#   score (numeric target: e.g., test score 0-100)
#
# NOTE: We use 'score' as numeric target for regression.
# Yes/No columns cannot be target in regression tasks,
# because regression predicts continuous numeric values.
# ================================================================

from pycaret.regression import *
import pandas as pd

data = pd.read_csv("Kaka_regression.csv")

reg = setup(
    data=data,
    target='score',  # Numeric target for regression, not Yes/No
    categorical_features=['hwdone', 'classatend'],
    session_id=123,
    silent=True,
    verbose=False
)

# Create AdaBoost regressor with default parameters
ada_default_reg = create_model('ada')

evaluate_model(ada_default_reg)
save_model(ada_default_reg, 'ada_regression_default')

source = input("Source of input (csv/self): ").strip().lower()

if source == 'csv':
    filename = input("Enter CSV file name: ")
    new_data = pd.read_csv(filename)
    preds = predict_model(ada_default_reg, data=new_data)
    print(preds)

elif source == 'self':
    learninghrs = int(input("Enter number of learning hours: "))
    hwdone = input("Homework done? (Yes/No): ").strip().capitalize()
    classatend = input("Class attended? (Yes/No): ").strip().capitalize()
    manual_data = pd.DataFrame({
        'learninghrs': [learninghrs],
        'hwdone': [hwdone],
        'classatend': [classatend]
    })
    preds = predict_model(ada_default_reg, data=manual_data)
    print(preds)

else:
    print("Invalid source.")