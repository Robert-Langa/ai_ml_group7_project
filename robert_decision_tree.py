"""
PROG2590 Project : Decision Tree for Adult Dataset
Team member: Robert Langa
Task: Binary classification : predict income >50K
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, classification_report)
import time
import os

# 1. Load the data and give it column names

column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'sex',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
]

# I gave multiple file paths because I had issues when it came to reading the .csv
possible_paths = [
    'data/group7-adult.csv',
    '../data/group7-adult.csv',
    'group7-adult.csv',
    'adult.data'
]

# The data_file that will exist by path will be used
data_file = None
for path in possible_paths:
    if os.path.exists(path):
        data_file = path
        break

# If the data_file is not found an error will be thrown, this is to tell the one running the program that 
# the dataset could not be found
if data_file is None:
    raise FileNotFoundError("Could not find the dataset file.")

# After the data file has been confirmed it will be printed on the console so as you can know what columns
# you are working with
print(f"Loading data from: {data_file}")
df = pd.read_csv(data_file, names=column_names, skipinitialspace=True)


# 2. Data cleaning
# Strip whitespace from string columns only
string_cols = df.select_dtypes(include=['object', 'string']).columns
for col in string_cols:
    df[col] = df[col].str.strip()

# Replace '?' with NaN and drop rows with missing values
df = df.replace('?', pd.NA).dropna()

# Ensure numeric columns are actually numeric, and drop rows with non-numeric values in those columns
numeric_candidates = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
for col in numeric_candidates:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna()

print(f"\nDataset shape after cleaning: {df.shape}")
print("\nTarget distribution:")
print(df['income'].value_counts())
print(df['income'].value_counts(normalize=True))

# Separate features and target variable. The target variable 'income' is mapped to binary values (0 for <=50K and 1 for >50K). If there are any missing values in the target variable, those rows are removed from both X and y. Finally, y is converted to integer type.
X = df.drop('income', axis=1)
y = df['income'].map({'<=50K': 0, '>50K': 1})
if y.isna().any():
    valid = ~y.isna()
    X = X[valid]
    y = y[valid]
y = y.astype(int)

# 3. Feature types
# The categorical and numerical features are identified and printed. Categorical features are those with data types 'object' or 'string', while numerical features are those with data types 'int64' or 'float64'.
categorical_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical features: {categorical_cols}")
print(f"Numerical features: {numerical_cols}")


# 4. Train/Test Split
# The dataset is split into a training set (80%) and a test set (20%) using stratified sampling to maintain the class distribution. The sizes of the training and test sets are printed to the console.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# 5. Pipeline with One-Hot Encoding and Decision Tree Classifier
# A pipeline is created that first applies one-hot encoding to the categorical features and then fits a Decision Tree Classifier. The ColumnTransformer is used to apply the one-hot encoding only to the categorical columns, while leaving the numerical columns unchanged (passthrough).
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# 6. Baseline model evaluation (default parameters)
# The pipeline is fitted on the training data, and predictions are made on the test set. Performance metrics including accuracy, precision, recall, F1-score, and ROC AUC are computed and displayed. The training time for the baseline model is also measured and printed.

print("\n" + "="*60)
print("BASELINE DECISION TREE (default parameters)")
print("="*60)

start_time = time.time()
pipeline.fit(X_train, y_train)
train_time = time.time() - start_time

y_pred_baseline = pipeline.predict(X_test)
y_proba_baseline = pipeline.predict_proba(X_test)[:, 1]

acc_baseline = accuracy_score(y_test, y_pred_baseline)
prec_baseline = precision_score(y_test, y_pred_baseline)
rec_baseline = recall_score(y_test, y_pred_baseline)
f1_baseline = f1_score(y_test, y_pred_baseline)
auc_baseline = roc_auc_score(y_test, y_proba_baseline)

print(f"Accuracy:  {acc_baseline:.4f}")
print(f"Precision: {prec_baseline:.4f}")
print(f"Recall:    {rec_baseline:.4f}")
print(f"F1-score:  {f1_baseline:.4f}")
print(f"ROC AUC:   {auc_baseline:.4f}")
print(f"Training time: {train_time:.2f} seconds")

# 7. Hyperparameter tuning with GridSearchCV
# A grid search is performed to find the best hyperparameters for the Decision Tree. The parameters being tuned include max_depth, min_samples_split, min_samples_leaf, and criterion. The grid search uses 5-fold cross-validation and evaluates models based on ROC AUC score. The best parameters and the corresponding cross-validation score are printed after the search is completed.

print("\n" + "="*60)
print("HYPERPARAMETER TUNING (GridSearchCV with 5-fold CV)")
print("="*60)

param_grid = {
    'classifier__max_depth': [5, 10, 15, 20, None],
    'classifier__min_samples_split': [2, 5, 10, 20],
    'classifier__min_samples_leaf': [1, 2, 4, 8],
    'classifier__criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    pipeline, param_grid, cv=5, scoring='roc_auc',
    n_jobs=-1, verbose=1, return_train_score=True
)

start_time = time.time()
grid_search.fit(X_train, y_train)
grid_time = time.time() - start_time

print(f"\nGrid search completed in {grid_time:.2f} seconds")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation ROC AUC: {grid_search.best_score_:.4f}")

# 8. Tuned model evaluation
# The best pipeline from the grid search is used to predict on the test set, and the same performance metrics are computed as for the baseline model. The improvement over the baseline is also calculated and displayed.

best_pipeline = grid_search.best_estimator_
y_pred_tuned = best_pipeline.predict(X_test)
y_proba_tuned = best_pipeline.predict_proba(X_test)[:, 1]

acc_tuned = accuracy_score(y_test, y_pred_tuned)
prec_tuned = precision_score(y_test, y_pred_tuned)
rec_tuned = recall_score(y_test, y_pred_tuned)
f1_tuned = f1_score(y_test, y_pred_tuned)
auc_tuned = roc_auc_score(y_test, y_proba_tuned)

print("\n" + "="*60)
print("TUNED DECISION TREE PERFORMANCE")
print("="*60)
print(f"Accuracy:  {acc_tuned:.4f}")
print(f"Precision: {prec_tuned:.4f}")
print(f"Recall:    {rec_tuned:.4f}")
print(f"F1-score:  {f1_tuned:.4f}")
print(f"ROC AUC:   {auc_tuned:.4f}")

print(f"\nImprovement over baseline:")
print(f"Accuracy:  {acc_tuned - acc_baseline:+.4f}")
print(f"ROC AUC:   {auc_tuned - auc_baseline:+.4f}")

# 9. Confusion Matrix and Classification Report
# The confusion matrix is displayed as a heatmap, and the classification report provides precision, recall, and F1-score for each class.

cm = confusion_matrix(y_test, y_pred_tuned)
print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['<=50K', '>50K'],
            yticklabels=['<=50K', '>50K'])
plt.title('Confusion Matrix – Tuned Decision Tree')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred_tuned, target_names=['<=50K', '>50K']))

# 10. ROC Curve
# ROC curve is plotted for the tuned model, and the AUC is displayed in the legend. A dashed line represents a random classifier for reference.

fpr, tpr, _ = roc_curve(y_test, y_proba_tuned)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f'Tuned DT (AUC = {auc_tuned:.3f})')
plt.plot([0,1], [0,1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve.png')
plt.show()


# 11. Feature Importance
# The top 15 most important features are displayed in a horizontal bar chart. The feature names are obtained from the one-hot encoder for categorical features and passed through the preprocessor to align with the model's input.

best_tree = best_pipeline.named_steps['classifier']
preprocessor_fitted = best_pipeline.named_steps['preprocessor']
cat_feature_names = preprocessor_fitted.named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_feature_names = np.concatenate([cat_feature_names, numerical_cols])

importances = best_tree.feature_importances_
indices = np.argsort(importances)[::-1][:15]

plt.figure(figsize=(10,6))
plt.barh(range(len(indices)), importances[indices][::-1])
plt.yticks(range(len(indices)), [all_feature_names[i] for i in indices][::-1])
plt.xlabel('Feature Importance')
plt.title('Top 15 Feature Importances – Decision Tree')
plt.tight_layout()
plt.savefig('feature_importances.png')
plt.show()

# 12. Cross-validation scores
# The 5-fold cross-validation ROC AUC scores for the best pipeline are computed and displayed, along with the mean and standard deviation.

cv_scores = cross_val_score(best_pipeline, X_train, y_train, cv=5, scoring='roc_auc')
print(f"\n5-fold CV ROC AUC scores: {cv_scores}")
print(f"Mean CV ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 13. Save results to a text file

with open('decision_tree_results.txt', 'w') as f:
    f.write("PROG2590 Project – Decision Tree on Adult Dataset\n")
    f.write("="*60 + "\n\n")
    f.write(f"Training samples: {X_train.shape[0]}\n")
    f.write(f"Test samples: {X_test.shape[0]}\n")
    f.write(f"Class distribution (train): {y_train.value_counts(normalize=True).to_dict()}\n\n")
    f.write("BASELINE MODEL\n")
    f.write(f"Accuracy: {acc_baseline:.4f}\n")
    f.write(f"ROC AUC: {auc_baseline:.4f}\n\n")
    f.write("TUNED MODEL\n")
    f.write(f"Best parameters: {grid_search.best_params_}\n")
    f.write(f"Accuracy: {acc_tuned:.4f}\n")
    f.write(f"Precision: {prec_tuned:.4f}\n")
    f.write(f"Recall: {rec_tuned:.4f}\n")
    f.write(f"F1-score: {f1_tuned:.4f}\n")
    f.write(f"ROC AUC: {auc_tuned:.4f}\n\n")
    f.write(f"Cross-validation ROC AUC mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n")

print("\nResults saved to 'decision_tree_results.txt'")
print("\n" + "="*60)
print("DECISION TREE PROJECT SCRIPT COMPLETED SUCCESSFULLY")
print("="*60)