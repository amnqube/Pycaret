# =========================================================
# 📌 2. CLASSIFICATION with XGBoost (Custom Parameters)
# Dataset: Kaka.csv
# Goal: Predict whether student will Pass or Fail
# =========================================================

from pycaret.classification import *
import pandas as pd

data = pd.read_csv("Kaka.csv")

clf = setup(
    data=data,
    target='pass',
    categorical_features=['hwdone', 'classatend'],
    session_id=123,
    silent=True,
    verbose=False
)

# Creating XGBoost model with FULL control over hyperparameters
xgb_custom = create_model(
    'xgboost',
    n_estimators=500,          # Number of trees (boosting rounds)
    max_depth=8,               # Max depth of each tree
    learning_rate=0.05,        # Step size for each tree
    min_child_weight=3,        # Min weight in a child node
    gamma=0.2,                  # Loss reduction needed to split
    subsample=0.85,             # Fraction of data per tree
    colsample_bytree=0.8,       # Fraction of features per tree
    reg_alpha=0.1,              # L1 regularization
    reg_lambda=1.5,             # L2 regularization
    objective='binary:logistic',
    eval_metric='logloss'
)

evaluate_model(xgb_custom)
save_model(xgb_custom, 'xgb_classification_custom')

source = input("Source of input (csv/self): ").strip().lower()

if source == 'csv':
    filename = input("Enter CSV file name: ")
    new_data = pd.read_csv(filename)
    preds = predict_model(xgb_custom, data=new_data)
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
    preds = predict_model(xgb_custom, data=manual_data)
    print(preds)

else:
    print("Invalid source.")