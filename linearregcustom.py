# ============================================================
# 📌 MULTIVARIATE LINEAR REGRESSION with PyCaret (Custom)
# Note: Linear regression has minimal hyperparameters in PyCaret
# Mainly handles 'fit_intercept' and 'normalize' internally
# ============================================================

from pycaret.regression import *
import pandas as pd

data = pd.read_csv("Kaka_regression.csv")

reg = setup(
    data=data,
    target='score',
    categorical_features=['hwdone', 'classatend'],
    session_id=123,
    silent=True,
    verbose=False
)

# Create linear regression model with optional custom parameters
# PyCaret does not expose many hyperparameters for 'lr' model
# You can specify 'fit_intercept' or other sklearn args via tune_model()
lr_model_custom = create_model('lr')

# You can tune the model hyperparameters (e.g., fit_intercept)
# tune_model(lr_model_custom)  # Uncomment if you want interactive tuning

evaluate_model(lr_model_custom)
save_model(lr_model_custom, 'linear_regression_custom')

source = input("Source of input (csv/self): ").strip().lower()

if source == 'csv':
    filename = input("Enter CSV file name: ")
    new_data = pd.read_csv(filename)
    preds = predict_model(lr_model_custom, data=new_data)
    print(preds)

elif source == 'self':
    learninghrs = float(input("Enter learning hours (numeric): "))
    hwdone = input("Homework done? (Yes/No): ").strip().capitalize()
    classatend = input("Class attended? (Yes/No): ").strip().capitalize()
    manual_data = pd.DataFrame({
        'learninghrs': [learninghrs],
        'hwdone': [hwdone],
        'classatend': [classatend]
    })
    preds = predict_model(lr_model_custom, data=manual_data)
    print(preds)

else:
    print("Invalid source.")