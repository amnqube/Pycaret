# ================================================================
# 📌 AdaBoost Regression with Custom Hyperparameters
# Same dataset and columns as above
# ================================================================

from pycaret.regression import *
import pandas as pd

data = pd.read_csv("Kaka_regression.csv")

reg = setup(
    data=data,
    target='score',  # Numeric target necessary for regression
    categorical_features=['hwdone', 'classatend'],
    session_id=123,
    silent=True,
    verbose=False
)

# Create AdaBoost regressor with custom hyperparameters
ada_custom_reg = create_model(
    'ada',
    n_estimators=300,
    learning_rate=0.1
)

evaluate_model(ada_custom_reg)
save_model(ada_custom_reg, 'ada_regression_custom')

source = input("Source of input (csv/self): ").strip().lower()

if source == 'csv':
    filename = input("Enter CSV file name: ")
    new_data = pd.read_csv(filename)
    preds = predict_model(ada_custom_reg, data=new_data)
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
    preds = predict_model(ada_custom_reg, data=manual_data)
    print(preds)

else:
    print("Invalid source.")