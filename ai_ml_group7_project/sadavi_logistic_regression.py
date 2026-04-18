import sys
import time
import tracemalloc
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, RocCurveDisplay
)

warnings.filterwarnings("ignore")

use_holdout_testset = len(sys.argv) >= 2


# a fn to run the model on the holdout test set and print scores:
def run_holdout_tests(holdout_testset_filename, model, feature_columns):
    # load the holdout test set into a pandas DataFrame:
    holdout_df = pd.read_csv(holdout_testset_filename)

    # apply the same cleaning as the training data:
    str_cols = holdout_df.select_dtypes(include="object").columns
    holdout_df[str_cols] = holdout_df[str_cols].apply(lambda col: col.str.strip())
    holdout_df.replace("?", np.nan, inplace=True)
    holdout_df.dropna(inplace=True)

    # apply the same feature engineering as the training data:
    holdout_df.drop(columns=["fnlwgt", "education"], errors="ignore", inplace=True)
    holdout_df["income"] = (holdout_df["income"] == ">50K").astype(int)
    cat_cols = holdout_df.select_dtypes(include="object").columns.tolist()
    holdout_df = pd.get_dummies(holdout_df, columns=cat_cols, drop_first=True)

    # separate into features matrix X_holdout and target vector y_holdout:
    X_holdout = holdout_df.drop("income", axis=1, errors="ignore")
    y_holdout = holdout_df["income"]

    # align columns to match the training set:
    X_holdout = X_holdout.reindex(columns=feature_columns, fill_value=0)

    # print out scores on the holdout test set:
    y_pred = model.predict(X_holdout)
    y_prob = model.predict_proba(X_holdout)[:, 1]
    print(f"\nHoldout test set results ({model.__class__.__name__}):")
    print(f"  Accuracy  : {accuracy_score(y_holdout, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_holdout, y_pred):.4f}")
    print(f"  Recall    : {recall_score(y_holdout, y_pred):.4f}")
    print(f"  F1 Score  : {f1_score(y_holdout, y_pred):.4f}")
    print(f"  ROC-AUC   : {roc_auc_score(y_holdout, y_prob):.4f}")
    print(classification_report(y_holdout, y_pred, target_names=["<=50K", ">50K"]))


# load the adult dataset into a pandas DataFrame:
adult_df = pd.read_csv("../data/group7-adult.csv")

# strip leading/trailing whitespace from all string columns:
str_cols = adult_df.select_dtypes(include="object").columns
adult_df[str_cols] = adult_df[str_cols].apply(lambda col: col.str.strip())

# replace '?' with NaN and drop rows with missing values:
adult_df.replace("?", np.nan, inplace=True)
adult_df.dropna(inplace=True)

# drop fnlwgt (census weight, not useful for prediction) and education (redundant with education-num):
adult_df.drop(columns=["fnlwgt", "education"], inplace=True)

# encode the target column: <=50K -> 0, >50K -> 1:
adult_df["income"] = (adult_df["income"] == ">50K").astype(int)

# one-hot encode all remaining categorical columns:
cat_cols = adult_df.select_dtypes(include="object").columns.tolist()
adult_df = pd.get_dummies(adult_df, columns=cat_cols, drop_first=True)

# separate into a features matrix X and a target vector y:
X = adult_df.drop("income", axis=1)
y = adult_df["income"]

# do a train/test split of the data fixing the random state for reproducibility:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# build a pipeline with a scaler and logistic regression model:
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

# tune hyperparameters using GridSearchCV with 5-fold cross-validation:
param_grid = {
    "clf__C": [0.01, 0.1, 1, 10, 100],
    "clf__solver": ["lbfgs", "liblinear"],
    "clf__penalty": ["l2"]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)

# track training time and memory usage:
tracemalloc.start()
t_start = time.perf_counter()
grid_search.fit(X_train, y_train)
train_time = time.perf_counter() - t_start
_, peak_mem_bytes = tracemalloc.get_traced_memory()
tracemalloc.stop()

# get the best model from the grid search:
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV accuracy: {grid_search.best_score_:.4f}")
print(f"Training time: {train_time:.4f}s  |  Peak memory: {peak_mem_bytes / (1024**2):.2f} MB")

# predict on the test set and measure prediction time:
t_pred = time.perf_counter()
y_pred = best_model.predict(X_test)
pred_time = time.perf_counter() - t_pred
y_prob = best_model.predict_proba(X_test)[:, 1]

# print out scores on the training and test data:
print(f"\nAccuracy on training data (LogisticRegression): {best_model.score(X_train, y_train):.4f}")
print(f"Accuracy on test data (LogisticRegression): {best_model.score(X_test, y_test):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(f"Prediction time: {pred_time:.4f}s")
print(f"\n{classification_report(y_test, y_pred, target_names=['<=50K', '>50K'])}")

# run 5-fold cross-validation on the training data:
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="accuracy")
print(f"Cross-validation accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
print(f"Per-fold scores: {np.round(cv_scores, 4)}")

# plot confusion matrix, cross-validation scores, and ROC curve:
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Logistic Regression - Adult Income Prediction", fontsize=13)

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["<=50K", ">50K"]).plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("Confusion Matrix")

axes[1].bar([f"Fold {i+1}" for i in range(5)], cv_scores, color="steelblue", edgecolor="black")
axes[1].axhline(cv_scores.mean(), color="red", linestyle="--", label=f"Mean = {cv_scores.mean():.3f}")
axes[1].set_ylim(0.7, 1.0)
axes[1].set_ylabel("Accuracy")
axes[1].set_title("5-Fold Cross-Validation Accuracy")
axes[1].legend()

RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[2], name="Logistic Regression")
axes[2].plot([0, 1], [0, 1], "k--")
axes[2].set_title(f"ROC Curve (AUC = {roc_auc_score(y_test, y_prob):.3f})")

plt.tight_layout()
plt.savefig("sadavi_logistic_regression_results.png", dpi=150)
print("\nPlot saved to sadavi_logistic_regression_results.png")

if use_holdout_testset:
    holdout_testset_filename = sys.argv[1]
    run_holdout_tests(holdout_testset_filename, best_model, X.columns.tolist())