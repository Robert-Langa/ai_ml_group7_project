## Student Name: Sarah Veluz
## Model : Random Forest

import pandas as pd
import time
import matplotlib.pyplot as plt
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

#For Hold-out file
use_holdout_testset = len(sys.argv) >= 2

# Load Data
print("Loading dataset...")
df = pd.read_csv(os.path.join("..", "data", "group7-adult.csv"))
print(df.head())

# Data Cleaning
print("\nCleaning data...")
df = df.replace(" ?", pd.NA)
df = df.dropna()
df = df.drop("fnlwgt", axis=1)
print("Shape after cleaning:", df.shape)

# Encode Target
print("\nEncoding target variable...")
df["income"] = df["income"].apply(lambda x: 1 if ">50K" in x else 0)
print(df["income"].value_counts())

# Data Encoding
print("\nEncoding categorical variables...")
df = pd.get_dummies(df, drop_first=True)
print("New shape:", df.shape)

# Train-Test Split
print("\nSplitting data...")
X = df.drop("income", axis=1)
y = df["income"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model Training + Runtime
print("\nTraining Random Forest model...")
start = time.time()

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

end = time.time()
print("Training time:", end - start, "seconds")

# Evaluation
print("\nEvaluating model...")
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Feature Importance (FIXED VERSION)
importances = model.feature_importances_
features = X.columns

indices = importances.argsort()[-10:]

plt.figure(figsize=(10,6))
plt.barh(features[indices], importances[indices])
plt.title("Top 10 Feature Importance")
plt.show()



# HOLD-OUT TEST FUNCTION

def run_holdout_tests(holdout_testset_filename, model, X_columns):
    print("\nRunning hold-out test set...")

    holdout_df = pd.read_csv(holdout_testset_filename)

    # SAME preprocessing
    holdout_df = holdout_df.replace(" ?", pd.NA)
    holdout_df = holdout_df.dropna()
    holdout_df = holdout_df.drop("fnlwgt", axis=1)

    # Handle target if exists
    if "income" in holdout_df.columns:
        holdout_df["income"] = holdout_df["income"].apply(
            lambda x: 1 if ">50K" in x else 0
        )
        y_holdout = holdout_df["income"]
        X_holdout = holdout_df.drop("income", axis=1)
    else:
        y_holdout = None
        X_holdout = holdout_df

    # Encode
    X_holdout = pd.get_dummies(X_holdout, drop_first=True)

    # Align columns
    X_holdout = X_holdout.reindex(columns=X_columns, fill_value=0)

    # Predict
    predictions = model.predict(X_holdout)

    print("\nSample predictions:")
    print(predictions[:10])

    # Evaluate if labels exist
    if y_holdout is not None:
        print("\nAccuracy on hold-out set:",
              accuracy_score(y_holdout, predictions))
        print("\nClassification Report:\n")
        print(classification_report(y_holdout, predictions))


# Run hold-out test if file is provided
if use_holdout_testset:
    holdout_testset_filename = sys.argv[1]
    run_holdout_tests(holdout_testset_filename, model, X.columns)