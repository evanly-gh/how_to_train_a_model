# How to Train a Model

A beginner-friendly project that walks through the complete machine learning pipeline — from raw data to a trained, evaluated model. If you've never trained a model before, this is a good place to start.

We use the **Iris dataset** (a small, clean dataset of flower measurements) and a **Random Forest classifier** (a reliable, easy-to-understand algorithm). The goal isn't to build something production-ready — it's to understand the fundamental steps that *every* ML project follows.

---

## What You Will Learn

By working through this project, you will understand:

- **What a dataset looks like** — rows of samples, columns of features, and a target label you're trying to predict.
- **Why you explore data before training** — Exploratory Data Analysis (EDA) helps you spot patterns, catch problems, and build intuition about which features matter.
- **How to split data into training and test sets** — and *why* this matters. If you test on the same data you trained on, you can't tell whether the model actually learned or just memorized.
- **What "training a model" actually means** — the model examines labeled examples and learns patterns that map features to labels.
- **How to evaluate a model** — accuracy alone isn't enough. You'll learn precision, recall, F1-score, and how to read a confusion matrix to understand *where* the model makes mistakes.
- **What feature importance is** — which inputs the model relied on most to make its decisions.

---

## Prerequisites

You need **Python 3.8+** installed on your machine. No prior machine learning experience is required — that's the whole point of this project.

If you've never used Python before, you should be comfortable with:
- Running commands in a terminal
- Basic Python syntax (variables, loops, print statements)

---

## Project Structure

| File | Purpose |
|------|---------|
| `train.py` | The main script. Loads data, trains a Random Forest model, and prints evaluation results. **Start here after EDA.** |
| `EDA.ipynb` | A Jupyter notebook that visualizes the dataset. **Run this first** to understand the data before training. |
| `requirements.txt` | Lists the Python packages this project depends on. You install these in the setup step below. |
| `.gitignore` | Tells Git which files to ignore (virtual environments, compiled files, etc.). You don't need to touch this. |

---

## Getting Started — Step by Step

Follow these steps in order. Each step builds on the previous one.

### Step 1: Clone the repository

This downloads the project to your computer.

```bash
git clone https://github.com/evanly-gh/how_to_train_a_model.git
cd how_to_train_a_model
```

### Step 2: Create a virtual environment

A virtual environment keeps this project's packages isolated from your system Python. This prevents version conflicts with other projects.

```bash
python -m venv venv
```

Activate it:
```bash
# macOS / Linux
source venv/bin/activate

# Windows (Command Prompt)
venv\Scripts\activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

You'll know it's active when you see `(venv)` at the beginning of your terminal prompt.

### Step 3: Install dependencies

This reads `requirements.txt` and installs the packages listed there.

```bash
pip install -r requirements.txt
```

**What's being installed and why:**
| Package | What it does |
|---------|-------------|
| `pandas` | Data manipulation — loads data into tables (DataFrames) you can filter, sort, and analyze |
| `scikit-learn` | The core ML library — provides datasets, model algorithms, train/test splitting, and evaluation metrics |
| `matplotlib` | Plotting library — creates charts and graphs |
| `seaborn` | Built on matplotlib — makes statistical visualizations easier and prettier |
| `jupyter` | Runs interactive notebooks (`.ipynb` files) where you can execute code cell by cell |

### Step 4: Explore the data (EDA)

**Do this before training.** In real ML projects, you never jump straight to training. You always look at the data first to understand what you're working with.

```bash
jupyter notebook EDA.ipynb
```

This opens a notebook in your browser. Run each cell top-to-bottom (Shift+Enter) and read the markdown explanations between code cells. Pay attention to:
- **Class balance** — are the species evenly represented? (Imbalanced classes can bias a model.)
- **Feature distributions** — do the species overlap in certain measurements?
- **Which features separate the species best** — this tells you what the model will likely rely on.

### Step 5: Train the model

Now that you understand the data, train the model and see how it performs:

```bash
python train.py
```

The script prints output for each phase of the pipeline. Read the output carefully — the comments in the code explain what each number means.

---

## The ML Pipeline — What's Happening in `train.py`

Every machine learning project follows roughly the same steps. Here's what `train.py` does and why each step matters.

### Phase 1: Data Preparation

**What:** Load the Iris dataset and split it into two groups — a training set (80%) and a test set (20%).

**Why this matters:** The model learns from the training set. The test set is held back as "unseen data" to check whether the model can generalize — i.e., make correct predictions on data it wasn't trained on. If you skip this step and evaluate on the training data, you'll get misleadingly high accuracy because the model has already seen those examples.

**Key concept — Features vs. Target:**
- **Features (X):** The input measurements — sepal length, sepal width, petal length, petal width. These are what the model uses to make predictions.
- **Target (y):** The label you're trying to predict — the species of the flower (setosa, versicolor, or virginica).

### Phase 2: Model Training

**What:** Create a Random Forest classifier and call `.fit()` to train it on the training data.

**Why Random Forest:** It's an *ensemble* method — it builds 100 decision trees, each looking at a random subset of the data, then takes a majority vote. This makes it resistant to overfitting (memorizing noise in the training data) and usually gives good results out of the box. It's a great first algorithm to learn because it works well without much tuning.

**Key concept — What `.fit()` does:** When you call `model.fit(X_train, y_train)`, the algorithm examines the training features and labels, finds patterns (decision boundaries), and stores them internally. After `.fit()`, the model is ready to make predictions on new data.

### Phase 3: Evaluation

**What:** Use the trained model to predict species on the test set, then compare predictions to the actual labels.

**Metrics you'll see:**

| Metric | What it tells you |
|--------|------------------|
| **Accuracy** | What percentage of predictions were correct overall. Simple but can be misleading with imbalanced classes. |
| **Precision** | "Of everything the model *predicted* as species X, how many actually were species X?" High precision = few false positives. |
| **Recall** | "Of all *actual* species X samples, how many did the model correctly identify?" High recall = few false negatives. |
| **F1-score** | The harmonic mean of precision and recall. Useful when you care about both equally. |
| **Confusion matrix** | A table showing exactly which species the model confused with each other. Diagonal = correct, off-diagonal = errors. |
| **Feature importances** | How much each feature contributed to the model's decisions. Higher = the model relied on that feature more. |

---

## About the Iris Dataset

The Iris dataset is a classic machine learning benchmark introduced by statistician Ronald Fisher in 1936. It's often the first dataset people use to learn ML because:

- It's **small** (150 samples) — easy to understand and fast to train on
- It's **clean** — no missing values, no noise, no preprocessing needed
- It has a **clear structure** — 3 species, 4 features, balanced classes (50 samples each)
- It's **built into scikit-learn** — no downloads or file management required

**The 4 features:**
| Feature | What it measures |
|---------|-----------------|
| Sepal length | Length of the outer leaf-like part of the flower (cm) |
| Sepal width | Width of the sepal (cm) |
| Petal length | Length of the inner colored part of the flower (cm) |
| Petal width | Width of the petal (cm) |

**The 3 species:** Setosa, Versicolor, Virginica

---

## Concepts to Remember

These are the key ideas from this project that apply to *any* ML project, not just this one:

1. **Always explore your data first.** EDA reveals patterns, problems, and insights that inform how you approach training.
2. **Always split into train and test sets.** Evaluating on training data gives you a false sense of accuracy.
3. **Accuracy isn't everything.** Look at precision, recall, and the confusion matrix to understand *how* the model is failing, not just *how often*.
4. **Feature importance tells you what the model learned.** If it doesn't match your domain knowledge, something might be wrong.
5. **Reproducibility matters.** Setting `random_state` ensures you (and others) get the same results every time.

---

## Next Steps

Once you're comfortable with this project, try these experiments to deepen your understanding:

### Change the model
Replace `RandomForestClassifier` with a different algorithm and compare accuracy:
```python
from sklearn.svm import SVC                          # Support Vector Machine
from sklearn.neighbors import KNeighborsClassifier    # K-Nearest Neighbors
from sklearn.linear_model import LogisticRegression   # Logistic Regression
```

### Tune hyperparameters
Modify these values in `train.py` and observe the effect:
- `n_estimators` — try 10, 50, 500. More trees = slower but potentially better.
- `max_depth` — add `max_depth=3` to limit tree depth. Does it help or hurt?
- `test_size` — try 0.1 (tiny test set) or 0.5 (half the data). How does it affect accuracy?

### Add cross-validation
Instead of a single train/test split, evaluate across multiple splits for a more reliable score:
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-validation accuracy: {scores.mean():.2%} (+/- {scores.std():.2%})")
```

### Try a harder dataset
The Iris dataset is intentionally easy. Try these built-in alternatives:
```python
from sklearn.datasets import load_wine          # 13 features, 3 classes
from sklearn.datasets import load_digits        # image classification (8x8 pixels)
from sklearn.datasets import load_breast_cancer # binary classification, 30 features
```
