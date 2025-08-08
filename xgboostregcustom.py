# =========================================================
# 📌 4. REGRESSION with XGBoost (Custom Parameters)
# Dataset: Kaka_reg.csv
# Goal: Predict student marks (numeric)
# =========================================================

from pycaret.regression import *
import pandas as pd

data = pd.read_csv("Kaka_reg.csv")

reg = setup(
    data=data,
    target='marks',
    categorical_features=['hwdone', 'classatend'],
    session_id=123,
    silent=True,
    verbose=False
)

xgb_custom = create_model(
    'xgboost',
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    min_child_weight=3,
    gamma=0.2,
    subsample=0.85,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.5,
    objective='reg:squarederror',
    eval_metric='rmse'
)

evaluate_model(xgb_custom)
save_model(xgb_custom, 'xgb_regression_custom')

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