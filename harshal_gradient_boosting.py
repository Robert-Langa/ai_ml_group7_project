import numpy as np
import pandas as pd
import timeit
import memory_profiler
import sklearn.model_selection as skms
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier


# Load dataset
adult_data = pd.read_csv("data/group7-adult.csv")

# Handle missing values
adult_data.replace(" ?", np.nan, inplace=True)
adult_data.dropna(inplace=True)

# Convert categorical data to numeric
adult_data = pd.get_dummies(adult_data)

# Features and target
X = adult_data.drop(["income_ >50K", "income_ <=50K"], axis=1)
y = adult_data["income_ >50K"]

# Split data 
(Adult_training_ftrs, Adult_test_ftrs,
 adult_training_tgt, adult_test_tgt) = skms.train_test_split(
    X, y, random_state=0
)

print(f'Training features shape: {Adult_training_ftrs.shape}')
print(f'Test features shape: {Adult_test_ftrs.shape}')

# Dictionary to store results
results = {}


# ---------------- MODELS ----------------

@memory_profiler.profile(precision=4)
def run_model_default(Adult_training_ftrs, adult_training_tgt, Adult_test_ftrs, adult_test_tgt):
    gbrt = GradientBoostingClassifier(random_state=0)

    start_time = timeit.default_timer()
    model = gbrt.fit(Adult_training_ftrs, adult_training_tgt)
    end_time = timeit.default_timer()
    train_time = round(1000*(end_time - start_time), 2)

    print("\nModel 1 (Default)")
    print(f'Training time: {train_time} ms')

    start_time = timeit.default_timer()
    predictions = model.predict(Adult_test_ftrs)
    end_time = timeit.default_timer()
    pred_time = round(1000*(end_time - start_time), 2)

    print(f'Prediction time: {pred_time} ms')

    acc = metrics.accuracy_score(adult_test_tgt, predictions)
    print(f'Accuracy: {acc:.3f} ({round(100*acc,1)}%)\n')

    results["Default"] = {"train_time": train_time, "pred_time": pred_time, "accuracy": acc}


@memory_profiler.profile(precision=4)
def run_model_estimators(Adult_training_ftrs, adult_training_tgt, Adult_test_ftrs, adult_test_tgt):
    gbrt = GradientBoostingClassifier(n_estimators=1000, random_state=0)

    start_time = timeit.default_timer()
    model = gbrt.fit(Adult_training_ftrs, adult_training_tgt)
    end_time = timeit.default_timer()
    train_time = round(1000*(end_time - start_time), 2)

    print("\nModel 2 (n_estimators=1000)")
    print(f'Training time: {train_time} ms')

    start_time = timeit.default_timer()
    predictions = model.predict(Adult_test_ftrs)
    end_time = timeit.default_timer()
    pred_time = round(1000*(end_time - start_time), 2)

    print(f'Prediction time: {pred_time} ms')

    acc = metrics.accuracy_score(adult_test_tgt, predictions)
    print(f'Accuracy: {acc:.3f} ({round(100*acc,1)}%)\n')

    results["n_estimators=1000"] = {"train_time": train_time, "pred_time": pred_time, "accuracy": acc}


@memory_profiler.profile(precision=4)
def run_model_lr(Adult_training_ftrs, adult_training_tgt, Adult_test_ftrs, adult_test_tgt):
    gbrt = GradientBoostingClassifier(learning_rate=0.01, random_state=0)

    start_time = timeit.default_timer()
    model = gbrt.fit(Adult_training_ftrs, adult_training_tgt)
    end_time = timeit.default_timer()
    train_time = round(1000*(end_time - start_time), 2)

    print("\nModel 3 (learning_rate=0.01)")
    print(f'Training time: {train_time} ms')

    start_time = timeit.default_timer()
    predictions = model.predict(Adult_test_ftrs)
    end_time = timeit.default_timer()
    pred_time = round(1000*(end_time - start_time), 2)

    print(f'Prediction time: {pred_time} ms')

    acc = metrics.accuracy_score(adult_test_tgt, predictions)
    print(f'Accuracy: {acc:.3f} ({round(100*acc,1)}%)\n')

    results["learning_rate=0.01"] = {"train_time": train_time, "pred_time": pred_time, "accuracy": acc}


@memory_profiler.profile(precision=4)
def run_model_depth(Adult_training_ftrs, adult_training_tgt, Adult_test_ftrs, adult_test_tgt):
    gbrt = GradientBoostingClassifier(max_depth=5, random_state=0)

    start_time = timeit.default_timer()
    model = gbrt.fit(Adult_training_ftrs, adult_training_tgt)
    end_time = timeit.default_timer()
    train_time = round(1000*(end_time - start_time), 2)

    print("\nModel 4 (max_depth=5)")
    print(f'Training time: {train_time} ms')

    start_time = timeit.default_timer()
    predictions = model.predict(Adult_test_ftrs)
    end_time = timeit.default_timer()
    pred_time = round(1000*(end_time - start_time), 2)

    print(f'Prediction time: {pred_time} ms')

    acc = metrics.accuracy_score(adult_test_tgt, predictions)
    print(f'Accuracy: {acc:.3f} ({round(100*acc,1)}%)\n')

    results["max_depth=5"] = {"train_time": train_time, "pred_time": pred_time, "accuracy": acc}


# Run models
run_model_default(Adult_training_ftrs, adult_training_tgt, Adult_test_ftrs, adult_test_tgt)
run_model_estimators(Adult_training_ftrs, adult_training_tgt, Adult_test_ftrs, adult_test_tgt)
run_model_lr(Adult_training_ftrs, adult_training_tgt, Adult_test_ftrs, adult_test_tgt)
run_model_depth(Adult_training_ftrs, adult_training_tgt, Adult_test_ftrs, adult_test_tgt)


# ----------- GRAPH SECTION -----------

model_names = list(results.keys())
train_times = [results[m]["train_time"] for m in model_names]
pred_times = [results[m]["pred_time"] for m in model_names]
accuracies = [results[m]["accuracy"] for m in model_names]

x = np.arange(len(model_names))
width = 0.25

# Time Graph
plt.figure(figsize=(10,5))
plt.bar(x - width/2, train_times, width, label="Training Time (ms)")
plt.bar(x + width/2, pred_times, width, label="Prediction Time (ms)")
plt.yscale('log')
plt.xticks(x, model_names)
plt.title("Model Time Comparison")
plt.xlabel("Models")
plt.ylabel("Time (log scale)")
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('TrainVsPredTime.png')
plt.show()

# Accuracy Graph
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
