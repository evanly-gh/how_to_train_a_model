# ============================================================
# train.py - Train a Random Forest Classifier on the Iris Dataset
# ============================================================
# This script demonstrates the core machine learning pipeline:
#   1. DATA PREPARATION  - Load and split the dataset
#   2. MODEL TRAINING    - Fit a Random Forest classifier
#   3. EVALUATION        - Measure accuracy and show results
# ============================================================

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# PHASE 1: DATA PREPARATION
# ============================================================

# Load the Iris dataset — a classic benchmark with 150 samples of 3 flower species,
# each described by 4 measurements (sepal length/width, petal length/width).
iris = load_iris()

# Create a DataFrame so we can inspect the data in a readable table format.
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target

print("=" * 60)
print("PHASE 1: DATA PREPARATION")
print("=" * 60)
print(f"\nDataset shape: {df.shape[0]} samples, {df.shape[1] - 1} features")
print(f"Species: {[str(name) for name in iris.target_names]}")
print(f"\nFirst 5 rows:\n{df.head()}\n")

# Separate features (X) from the target label (y).
X = iris.data
y = iris.target

# Split into training (80%) and testing (20%) sets.
# random_state=42 ensures the split is the same every time you run this script,
# which makes your results reproducible.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set:     {X_test.shape[0]} samples")


# ============================================================
# PHASE 2: MODEL TRAINING
# ============================================================

print("\n" + "=" * 60)
print("PHASE 2: MODEL TRAINING")
print("=" * 60)

# A Random Forest is an ensemble of decision trees. Each tree sees a random
# subset of the data and features, then they vote together on the final
# prediction. This "wisdom of the crowd" approach is robust and accurate.
model = RandomForestClassifier(n_estimators=100, random_state=42)

# .fit() is where the actual learning happens — the model examines the
# training data and builds its internal decision trees.
model.fit(X_train, y_train)

print("\nModel trained successfully!")
print(f"  - Algorithm:  Random Forest")
print(f"  - Trees:      {model.n_estimators}")
print(f"  - Features:   {model.n_features_in_}")


# ============================================================
# PHASE 3: EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("PHASE 3: EVALUATION")
print("=" * 60)

# Use the trained model to predict species for the test set (data it has never seen).
y_pred = model.predict(X_test)

# Accuracy = percentage of correct predictions out of all predictions.
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2%}")

# Classification report shows per-class metrics:
#   - Precision: of all samples predicted as class X, how many actually were X?
#   - Recall:    of all actual class X samples, how many did we correctly predict?
#   - F1-score:  harmonic mean of precision and recall (balances both).
print(f"\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion matrix: rows = actual class, columns = predicted class.
# Diagonal values are correct predictions; off-diagonal values are errors.
print("Confusion Matrix:")
print(f"  (rows = actual, columns = predicted)\n")
cm = confusion_matrix(y_test, y_pred)
# Print with species labels for readability
cm_df = pd.DataFrame(cm, index=iris.target_names, columns=iris.target_names)
print(cm_df)

# Feature importances: how much each feature contributed to the model's decisions.
# Higher values mean the feature was more useful for distinguishing species.
print("\nFeature Importances:")
for name, importance in sorted(
    zip(iris.feature_names, model.feature_importances_),
    key=lambda x: x[1],
    reverse=True,
):
    print(f"  {name:20s} {importance:.4f}")


# ============================================================
# NEXT STEPS — Try these experiments to deepen your understanding:
#   - Change test_size to 0.3 or 0.1 and see how accuracy changes
#   - Change n_estimators to 10 or 500 — does more trees = better?
#   - Swap RandomForestClassifier for another model:
#       from sklearn.svm import SVC
#       from sklearn.neighbors import KNeighborsClassifier
#       from sklearn.linear_model import LogisticRegression
#   - Add cross-validation:
#       from sklearn.model_selection import cross_val_score
#       scores = cross_val_score(model, X, y, cv=5)
# ============================================================
