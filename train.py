# ============================================================
# train.py - Train a Random Forest Classifier on the Iris Dataset
# ============================================================
#
# WHAT THIS SCRIPT DOES:
#   This is the main training script. It walks through the 3 core steps
#   that every machine learning project follows:
#
#     1. DATA PREPARATION  - Load raw data and split it into training/test sets
#     2. MODEL TRAINING    - Feed training data to an algorithm so it learns patterns
#     3. EVALUATION        - Test the trained model on unseen data to measure performance
#
# HOW TO RUN:
#   python train.py
#
# LEARNING OBJECTIVES:
#   - Understand the difference between features (inputs) and targets (labels)
#   - Know why you split data into training and test sets
#   - See what "training" actually looks like in code (it's one line: model.fit())
#   - Learn to read evaluation metrics: accuracy, precision, recall, confusion matrix
#   - Understand feature importances — what the model relied on most
#
# ============================================================


# --- IMPORTS ---
# Each library serves a specific purpose in the pipeline.

# pandas: Creates tabular data structures (DataFrames) so we can view and
# manipulate data like a spreadsheet. We use it here to display the dataset
# in a readable format and to format the confusion matrix.
import pandas as pd

# load_iris: A function that returns the Iris dataset directly — no file
# downloads needed. It comes bundled with scikit-learn.
from sklearn.datasets import load_iris

# train_test_split: Randomly divides your data into two groups — one for
# training the model and one for testing it afterward. This is critical:
# if you test on the same data you trained on, you can't tell if the model
# actually learned patterns or just memorized the answers.
from sklearn.model_selection import train_test_split

# RandomForestClassifier: The algorithm we'll use. A Random Forest builds
# many decision trees, each trained on a random subset of the data, then
# combines their predictions by majority vote. Think of it as asking 100
# experts who each see slightly different evidence, then going with whatever
# most of them agree on.
from sklearn.ensemble import RandomForestClassifier

# Evaluation metrics: Functions that measure how well the model performed.
#   - accuracy_score: overall percentage of correct predictions
#   - classification_report: per-class precision, recall, and F1-score
#   - confusion_matrix: a table showing correct vs. incorrect predictions per class
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# PHASE 1: DATA PREPARATION
# ============================================================
# GOAL: Get the data into a format the model can learn from, and set aside
#        a portion of it for testing later.
#
# WHY THIS MATTERS:
#   A model is only as good as the data you give it. In this phase, you:
#   1. Load the raw data
#   2. Inspect it to make sure you understand its shape and contents
#   3. Separate features (X) from the target (y)
#   4. Split into training and test sets
#
#   Skipping the inspection step is a common beginner mistake. Always look
#   at your data before feeding it to a model.
# ============================================================

# Load the Iris dataset.
# load_iris() returns an object with several attributes:
#   .data          → a 2D array of shape (150, 4) — the feature measurements
#   .target        → a 1D array of shape (150,) — the species labels as integers (0, 1, 2)
#   .feature_names → the names of the 4 features
#   .target_names  → the names of the 3 species
iris = load_iris()

# Create a pandas DataFrame for easier inspection.
# A DataFrame is like a spreadsheet — columns have names, rows have indices,
# and you can filter/sort/group the data easily.
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target  # Add the target column (0 = setosa, 1 = versicolor, 2 = virginica)

print("=" * 60)
print("PHASE 1: DATA PREPARATION")
print("=" * 60)

# IMPORTANT: Always inspect your data before doing anything with it.
# df.shape tells you (rows, columns). Here we expect 150 rows and 5 columns
# (4 features + 1 target). If these numbers are wrong, something went wrong
# during loading.
print(f"\nDataset shape: {df.shape[0]} samples, {df.shape[1] - 1} features")
print(f"Species: {[str(name) for name in iris.target_names]}")

# df.head() shows the first 5 rows. This is a quick sanity check —
# do the numbers look reasonable? Are there any obvious problems?
print(f"\nFirst 5 rows:\n{df.head()}\n")

# Separate features (X) from the target label (y).
#
# KEY CONCEPT — Features vs. Target:
#   X (features): The input data the model uses to make predictions.
#                 In this case, the 4 flower measurements.
#   y (target):   The label we want the model to predict.
#                 In this case, the species (0, 1, or 2).
#
# The model's job is to learn the mapping: given X, predict y.
X = iris.data   # Shape: (150, 4) — 150 samples, each with 4 feature values
y = iris.target  # Shape: (150,)   — 150 labels, one per sample

# Split into training (80%) and testing (20%) sets.
#
# WHY WE SPLIT:
#   Imagine studying for a test by memorizing the answer key. You'd score 100%
#   on that exact test, but you didn't actually learn the material. The same
#   thing happens with ML models — if you evaluate on training data, the model
#   can "cheat" by recalling memorized answers instead of using learned patterns.
#
#   The test set acts as a final exam with questions the model has never seen.
#   A high score on the test set means the model genuinely learned.
#
# WHAT random_state DOES:
#   The split is random (samples are shuffled before splitting). Setting
#   random_state=42 fixes the random seed so you get the exact same split
#   every time you run the script. This makes your results reproducible —
#   anyone running this code will see the same numbers you do.
#   (42 is just a convention — any integer works.)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Verify the split. You should see roughly 80% in training, 20% in testing.
# With 150 samples: 120 training + 30 testing = 150 total.
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set:     {X_test.shape[0]} samples")


# ============================================================
# PHASE 2: MODEL TRAINING
# ============================================================
# GOAL: Create a model and train it on the training data.
#
# WHAT "TRAINING" MEANS:
#   The algorithm examines the training features (X_train) and their
#   corresponding labels (y_train). It finds patterns — for example,
#   "flowers with petal length < 2.5 cm are almost always setosa."
#   These patterns are stored internally as decision rules.
#
#   After training, the model can take new, unseen measurements and
#   predict which species they belong to.
#
# WHY RANDOM FOREST:
#   It's a great first algorithm because:
#   - It works well out of the box with minimal tuning
#   - It's resistant to overfitting (memorizing noise in the data)
#   - It naturally tells you which features were most important
#   - It handles both numerical and categorical data
# ============================================================

print("\n" + "=" * 60)
print("PHASE 2: MODEL TRAINING")
print("=" * 60)

# Create the model.
#   n_estimators=100: Build 100 decision trees. Each tree trains on a
#     random subset of the data and features. More trees generally means
#     better performance, but with diminishing returns and slower training.
#   random_state=42: Makes the random tree-building process reproducible.
model = RandomForestClassifier(n_estimators=100, random_state=42)

# THIS IS WHERE THE LEARNING HAPPENS.
# .fit() is the single most important line in any ML script.
# Under the hood, scikit-learn:
#   1. Builds 100 decision trees, each using a random subset of X_train
#   2. Each tree learns rules like "if petal_length > 2.45 and petal_width > 1.75, predict virginica"
#   3. Stores all these trees internally in the model object
#
# After this line, `model` is trained and ready to make predictions.
model.fit(X_train, y_train)

print("\nModel trained successfully!")
print(f"  - Algorithm:  Random Forest")
print(f"  - Trees:      {model.n_estimators}")
print(f"  - Features:   {model.n_features_in_}")


# ============================================================
# PHASE 3: EVALUATION
# ============================================================
# GOAL: Measure how well the model performs on data it has never seen.
#
# WHY MULTIPLE METRICS:
#   Accuracy alone can be misleading. For example, if 95% of your data
#   is class A, a model that always guesses "A" gets 95% accuracy — but
#   it's useless. That's why we also look at precision, recall, the
#   confusion matrix, and feature importances.
#
# THINGS TO WATCH FOR:
#   - Is accuracy high on the test set? (Good generalization)
#   - Are precision and recall balanced across classes? (No class is being ignored)
#   - Does the confusion matrix show a specific pair of classes being confused?
#   - Do feature importances align with what EDA showed? (Sanity check)
# ============================================================

print("\n" + "=" * 60)
print("PHASE 3: EVALUATION")
print("=" * 60)

# Generate predictions on the test set.
# .predict() takes feature measurements and returns predicted labels.
# The model has NEVER seen these 30 test samples during training.
y_pred = model.predict(X_test)

# --- METRIC 1: ACCURACY ---
# The simplest metric: what fraction of predictions were correct?
# Formula: (number of correct predictions) / (total predictions)
# Example: 29 out of 30 correct = 96.67%
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2%}")

# --- METRIC 2: CLASSIFICATION REPORT ---
# This breaks down performance for EACH class separately.
#
# HOW TO READ IT:
#   Precision — "When the model predicted setosa, how often was it actually setosa?"
#     High precision = the model rarely gives false positives for this class.
#     Example: precision=0.90 means 10% of predictions for this class were wrong.
#
#   Recall — "Of all actual setosa samples, how many did the model find?"
#     High recall = the model rarely misses samples of this class.
#     Example: recall=0.80 means the model missed 20% of this class.
#
#   F1-score — Combines precision and recall into a single number.
#     It's the harmonic mean: 2 * (precision * recall) / (precision + recall).
#     Useful when you want one number that balances both.
#
#   Support — How many test samples belonged to this class.
print(f"\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# --- METRIC 3: CONFUSION MATRIX ---
# A table that shows EXACTLY what the model predicted vs. what was correct.
#
# HOW TO READ IT:
#   - Rows    = the ACTUAL species (ground truth)
#   - Columns = what the model PREDICTED
#   - Diagonal values (top-left to bottom-right) = CORRECT predictions
#   - Off-diagonal values = MISTAKES
#
# Example: if the cell at row "versicolor", column "virginica" shows 2,
# that means the model incorrectly predicted "virginica" for 2 flowers
# that were actually "versicolor".
#
# A perfect model has all values on the diagonal and zeros everywhere else.
print("Confusion Matrix:")
print(f"  (rows = actual, columns = predicted)\n")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=iris.target_names, columns=iris.target_names)
print(cm_df)

# --- METRIC 4: FEATURE IMPORTANCES ---
# Random Forest can tell you how much each feature contributed to its decisions.
#
# HOW IT WORKS:
#   The model measures how much each feature helped reduce prediction errors
#   across all 100 trees. Features that led to better splits (cleaner
#   separation of classes) get higher importance scores.
#
# WHY IT MATTERS:
#   - Confirms what you found in EDA (petal features should rank highest)
#   - If an unrelated feature ranks high, it might indicate data leakage
#   - Helps you understand what the model is actually doing, not just its score
#
# NOTE: Importances sum to 1.0 across all features.
print("\nFeature Importances:")
for name, importance in sorted(
    zip(iris.feature_names, model.feature_importances_),
    key=lambda x: x[1],
    reverse=True,
):
    print(f"  {name:20s} {importance:.4f}")


# ============================================================
# WHAT YOU JUST ACCOMPLISHED:
#   You loaded a dataset, trained a Random Forest, and evaluated it with
#   multiple metrics. This is the same workflow used in real ML projects —
#   the datasets are bigger and the models are more complex, but the
#   fundamental steps are identical.
#
# NEXT STEPS — Try these experiments to deepen your understanding:
#
#   1. CHANGE THE SPLIT:
#      Set test_size to 0.3 or 0.1. With a tiny test set, is accuracy
#      still reliable? With a large test set, does the model have enough
#      training data?
#
#   2. CHANGE THE NUMBER OF TREES:
#      Set n_estimators to 10 or 500. Does more trees = better accuracy?
#      At what point do you see diminishing returns?
#
#   3. TRY A DIFFERENT ALGORITHM:
#      Replace RandomForestClassifier with one of these and compare:
#        from sklearn.svm import SVC
#        from sklearn.neighbors import KNeighborsClassifier
#        from sklearn.linear_model import LogisticRegression
#
#   4. ADD CROSS-VALIDATION:
#      Instead of one train/test split, evaluate across 5 different splits:
#        from sklearn.model_selection import cross_val_score
#        scores = cross_val_score(model, X, y, cv=5)
#        print(f"Mean accuracy: {scores.mean():.2%}")
# ============================================================
