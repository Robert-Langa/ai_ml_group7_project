# Adult Income Prediction
# Random Forest Model

import pandas as pd
import time
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Data
df = pd.read_csv(os.path.join("..", "data", "group7-adult.csv"))

# Data Cleaning
df = df.replace(" ?", pd.NA)
df = df.dropna()
df = df.drop("fnlwgt", axis=1)

# Encode Target
df["income"] = df["income"].apply(lambda x: 1 if ">50K" in x else 0)

# Data Encoding
df = pd.get_dummies(df, drop_first=True)

# Train-Test Split
X = df.drop("income", axis=1)
y = df["income"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model Training + Runtime
start = time.time()

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

end = time.time()
print("Training time:", end - start, "seconds")

# Evaluation
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Feature Importance
importances = model.feature_importances_
features = X.columns

indices = importances.argsort()[-10:]

plt.figure(figsize=(10,6))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.title("Top Feature Importance")
plt.show()