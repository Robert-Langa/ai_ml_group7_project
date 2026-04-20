import numpy as np
import pandas as pd
import timeit
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier


# Load dataset
adult_data = pd.read_csv("data/group7-adult.csv")

# Handle missing values
adult_data.replace(" ?", np.nan, inplace=True)
adult_data.dropna(inplace=True)

# Remove/strip white spaces from all string columns
string_cols = adult_data.select_dtypes(include=['object']).columns
for col in string_cols:
    adult_data[col] = adult_data[col].str.strip()

# Features and target
y = adult_data["income"]
X = adult_data.drop("income", axis=1)

# Split data 
(X_train, X_test,
 y_train, y_test) = train_test_split(
    X, y, random_state=0, stratify=y
)

print(f'\nTraining set: {X_train.shape[0]}, features: {X_train.shape[1]}')
print(f'Test set: {X_test.shape[0]}, features: {X_test.shape[1]}\n')

# Convert categorical data to numeric
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Align train and test columns
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Dictionary to store results
results = {}


            # MODELS #

# MODEL 1: Default 
gbrt = GradientBoostingClassifier(random_state=0)

start_time = timeit.default_timer()
model1 = gbrt.fit(X_train, y_train)
end_time = timeit.default_timer()
train_time = round(1000 * (end_time - start_time), 2)

start_time = timeit.default_timer()
pred1 = model1.predict(X_test)
end_time = timeit.default_timer()
pred_time = round(1000 * (end_time - start_time), 2)

acc1 = metrics.accuracy_score(y_test, pred1)

print("\nModel 1 - Default")
print(f"Training Time: {train_time} ms")
print(f"Prediction Time: {pred_time} ms")
print(f"Accuracy: {acc1:.3f} ({round(100*acc1,1)}%)")

results["Default"] = (train_time, pred_time, acc1)


# MODEL 2: n_estimators=1000 
gbrt = GradientBoostingClassifier(n_estimators=1000, random_state=0)

start_time = timeit.default_timer()
model2 = gbrt.fit(X_train, y_train)
end_time = timeit.default_timer()
train_time = round(1000 * (end_time - start_time), 2)

start_time = timeit.default_timer()
pred2 = model2.predict(X_test)
end_time = timeit.default_timer()
pred_time = round(1000 * (end_time - start_time), 2)

acc2 = metrics.accuracy_score(y_test, pred2)

print("\nModel 2 - n_estimators=1000")
print(f"Training Time: {train_time} ms")
print(f"Prediction Time: {pred_time} ms")
print(f"Accuracy: {acc2:.3f} ({round(100*acc2,1)}%)")

results["n_estimators=1000"] = (train_time, pred_time, acc2)


# MODEL 3: learning_rate=0.01 
gbrt = GradientBoostingClassifier(learning_rate=0.01, random_state=0)

start_time = timeit.default_timer()
model3 = gbrt.fit(X_train, y_train)
end_time = timeit.default_timer()
train_time = round(1000 * (end_time - start_time), 2)

start_time = timeit.default_timer()
pred3 = model3.predict(X_test)
end_time = timeit.default_timer()
pred_time = round(1000 * (end_time - start_time), 2)

acc3 = metrics.accuracy_score(y_test, pred3)

print("\nModel 3 - learning_rate=0.01")
print(f"Training Time: {train_time} ms")
print(f"Prediction Time: {pred_time} ms")
print(f"Accuracy: {acc3:.3f} ({round(100*acc3,1)}%)")

results["learning_rate=0.01"] = (train_time, pred_time, acc3)


# MODEL 4: max_depth=5 
gbrt = GradientBoostingClassifier(max_depth=5, random_state=0)

start_time = timeit.default_timer()
model4 = gbrt.fit(X_train, y_train)
end_time = timeit.default_timer()
train_time = round(1000 * (end_time - start_time), 2)

start_time = timeit.default_timer()
pred4 = model4.predict(X_test)
end_time = timeit.default_timer()
pred_time = round(1000 * (end_time - start_time), 2)

acc4 = metrics.accuracy_score(y_test, pred4)

print("\nModel 4 - max_depth=5")
print(f"Training Time: {train_time} ms")
print(f"Prediction Time: {pred_time} ms")
print(f"Accuracy: {acc4:.3f} ({round(100*acc4,1)}%)")

results["max_depth=5"] = (train_time, pred_time, acc4)


            # BEST MODEL #
values = np.array(list(results.values()))

train_arr = values[:, 0]
pred_arr = values[:, 1]
acc_arr = values[:, 2]

model_names = list(results.keys())

best_train_idx = np.argmin(train_arr)
best_pred_idx = np.argmin(pred_arr)
best_acc_idx = np.argmax(acc_arr)

print("\nBest Models:")
print(f"Fastest Training Time: {model_names[best_train_idx]} ({np.min(train_arr)} ms)")
print(f"Fastest Prediction Time: {model_names[best_pred_idx]} ({np.min(pred_arr)} ms)")
print(f"Highest Accuracy: {model_names[best_acc_idx]} ({np.max(acc_arr):.3f})")


             # GRAPH SECTION #
model_names = list(results.keys())

train_times = [results[m][0] for m in model_names]
pred_times = [results[m][1] for m in model_names]
accuracies = [results[m][2] for m in model_names]

x = np.arange(len(model_names))
width = 0.25

best_model = [model1, model2, model3, model4][best_acc_idx]
importances = gbrt.feature_importances_
features = X_train.columns

indices = np.argsort(importances)[-15:]

top_features = features[indices]
top_importances = importances[indices]

order = np.argsort(top_importances)
top_features = top_features[order]
top_importances = top_importances[order]

plt.figure(figsize=(10,6))
plt.barh(top_features, top_importances)
plt.title("Top 15 Feature Importance")
plt.xlabel("Importance (Decimal)")
plt.ylabel("Features")
plt.grid(axis='x')
plt.tight_layout()
plt.savefig('ImportantFeatures.png')
plt.show()

plt.figure(figsize=(10,5))
plt.bar(x - width/2, train_times, width, label="Training Time (ms)")
plt.bar(x + width/2, pred_times, width, label="Prediction Time (ms)")
plt.yscale('log')
plt.xticks(x, model_names)
plt.title("Model Time Comparison")
plt.xlabel("Models")
plt.ylabel("Time")
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('TrainVsPredTime.png')
plt.show()

plt.figure(figsize=(10,5))
plt.bar(x, accuracies, width)
plt.xticks(x, model_names)
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('Accuracies.png')
plt.show()
