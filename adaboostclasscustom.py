# ================================================================
# 📌 AdaBoost Classification with Custom Hyperparameters
# Same dataset and columns as above
# ================================================================

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

# Create AdaBoost classifier with custom hyperparameters
ada_custom = create_model(
    'ada',
    n_estimators=300,   # Number of weak learners (trees)
    learning_rate=0.1   # Shrinkage applied to each weak learner
)

evaluate_model(ada_custom)
save_model(ada_custom, 'ada_classification_custom')

source = input("Source of input (csv/self): ").strip().lower()

if source == 'csv':
    filename = input("Enter CSV file name: ")
    new_data = pd.read_csv(filename)
    preds = predict_model(ada_custom, data=new_data)
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
    preds = predict_model(ada_custom, data=manual_data)
    print(preds)

else:
    print("Invalid source.")