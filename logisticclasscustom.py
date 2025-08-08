# =============================================================
# 📌 Logistic Regression Classification (Custom Hyperparameters)
# PyCaret’s logistic regression has limited hyperparameters.
# You can tune regularization (C) and penalty via tune_model.
# =============================================================

from pycaret.classification import *
import pandas as pd

data = pd.read_csv("Kaka_classification.csv")

clf = setup(
    data=data,
    target='pass',
    categorical_features=['hwdone', 'classatend'],
    session_id=123,
    silent=True,
    verbose=False
)

# Create logistic regression model (default parameters)
logreg_custom = create_model('lr')

# Optionally tune hyperparameters interactively
# Uncomment to enable tuning:
# logreg_custom = tune_model(logreg_custom)

evaluate_model(logreg_custom)
save_model(logreg_custom, 'logistic_regression_custom')

source = input("Source of input (csv/self): ").strip().lower()

if source == 'csv':
    filename = input("Enter CSV filename: ")
    new_data = pd.read_csv(filename)
    preds = predict_model(logreg_custom, data=new_data)
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
    preds = predict_model(logreg_custom, data=manual_data)
    print(preds)

else:
    print("Invalid source.")