# How to Train a Model

A beginner-friendly project that walks through the complete machine learning pipeline using the classic Iris dataset and a Random Forest classifier.

## What You Will Learn

- How to explore and visualize data (Exploratory Data Analysis)
- How to prepare data for machine learning (train/test split)
- How to train a classifier and evaluate its performance
- How to interpret accuracy, precision, recall, and confusion matrices

## Project Structure

| File | Description |
|------|-------------|
| `train.py` | ML pipeline: loads data, trains a Random Forest, prints evaluation metrics |
| `EDA.ipynb` | Jupyter notebook with visualizations and data exploration |
| `requirements.txt` | Python packages needed for this project |
| `README.md` | This guide |

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/evanly-gh/how_to_train_a_model.git
cd how_to_train_a_model
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate          # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Explore the data

```bash
jupyter notebook EDA.ipynb
```

### 5. Train the model

```bash
python train.py
```

## The ML Pipeline

1. **Data Preparation** -- Load the Iris dataset and split it into training (80%) and test (20%) sets. The model learns from the training set and is evaluated on the test set to check how well it generalizes.

2. **Model Training** -- Fit a Random Forest classifier, which is an ensemble of 100 decision trees that each see a random subset of the data and vote together on the final prediction.

3. **Evaluation** -- Measure accuracy (overall correctness), precision and recall (per-class performance), and review the confusion matrix to see exactly where the model gets things right or wrong.

## About the Iris Dataset

The Iris dataset is a classic machine learning benchmark introduced by statistician Ronald Fisher in 1936. It contains 150 samples of iris flowers from 3 species (setosa, versicolor, virginica), each described by 4 measurements: sepal length, sepal width, petal length, and petal width.

## Next Steps

- Try different classifiers: `SVC`, `KNeighborsClassifier`, `LogisticRegression`
- Experiment with hyperparameters (`n_estimators`, `max_depth`, `test_size`)
- Add cross-validation with `cross_val_score`
- Try a different dataset from `sklearn.datasets`
