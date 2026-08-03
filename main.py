import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================================================
# CREATE SCREENSHOT FOLDER
# ==========================================================

os.makedirs("screenshots", exist_ok=True)

# ==========================================================
# LOAD DATASET
# ==========================================================

data = pd.read_csv("dataset/dataset.txt", header=None)

data.columns = [
    "Variance",
    "Skewness",
    "Curtosis",
    "Entropy",
    "Class"
]

# ==========================================================
# DATA OVERVIEW
# ==========================================================

print("=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)
print(data.head())

print("\n")

print("=" * 60)
print("DATASET SHAPE")
print("=" * 60)
print(data.shape)

print("\n")

print("=" * 60)
print("COLUMN NAMES")
print("=" * 60)
print(data.columns)

print("\n")

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
data.info()

print("\n")

print("=" * 60)
print("MISSING VALUES")
print("=" * 60)
print(data.isnull().sum())

print("\n")

print("=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)
print(data.describe())

print("\n")

print("=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)
print(data["Class"].value_counts())

# ==========================================================
# GRAPH 1 : CLASS DISTRIBUTION
# ==========================================================

plt.figure(figsize=(6,4))

sns.countplot(
    data=data,
    x="Class",
    color="#4C72B0"
)

plt.title(
    "Distribution of Genuine and Counterfeit Banknotes",
    fontsize=14
)

plt.xlabel("Class (0 = Genuine, 1 = Counterfeit)")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    "screenshots/class_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================================================
# GRAPH 2 : HISTOGRAMS
# ==========================================================

data.hist(
    figsize=(12,8),
    bins=20,
    edgecolor="black"
)

plt.suptitle(
    "Distribution of Dataset Features",
    fontsize=16
)

plt.tight_layout(rect=[0,0,1,0.96])

plt.savefig(
    "screenshots/feature_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================================================
# GRAPH 3 : CORRELATION HEATMAP
# ==========================================================

plt.figure(figsize=(8,6))

sns.heatmap(
    data.corr(),
    annot=True,
    cmap="coolwarm",
    linewidths=0.5
)

plt.title(
    "Correlation Heatmap",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    "screenshots/heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================================================
# GRAPH 4 : PAIRPLOT
# ==========================================================

pair_plot = sns.pairplot(
    data,
    hue="Class",
    diag_kind="hist"
)

pair_plot.fig.suptitle(
    "Pair Plot of Banknote Features",
    y=1.02
)

pair_plot.savefig(
    "screenshots/pairplot.png",
    dpi=300
)

plt.show()
# ==========================================================
# DATA PREPROCESSING
# ==========================================================

X = data.drop("Class", axis=1)
y = data["Class"]

print("\n")
print("=" * 60)
print("FEATURES (X)")
print("=" * 60)
print(X.head())

print("\n")

print("=" * 60)
print("TARGET (y)")
print("=" * 60)
print(y.head())

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n")
print("=" * 60)
print("TRAINING DATA SHAPE")
print("=" * 60)
print("Features :", X_train.shape)
print("Labels   :", y_train.shape)

print("\n")

print("=" * 60)
print("TESTING DATA SHAPE")
print("=" * 60)
print("Features :", X_test.shape)
print("Labels   :", y_test.shape)

# ==========================================================
# LOGISTIC REGRESSION MODEL
# ==========================================================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\n")
print("=" * 60)
print("MODEL TRAINED SUCCESSFULLY")
print("=" * 60)

# ==========================================================
# PREDICTIONS
# ==========================================================

prediction = model.predict(X_test)

# ==========================================================
# MODEL ACCURACY
# ==========================================================

accuracy = accuracy_score(
    y_test,
    prediction
)

print("\n")
print("=" * 60)
print("MODEL ACCURACY")
print("=" * 60)

print(f"Accuracy : {accuracy*100:.2f}%")

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    y_test,
    prediction
)

print("\n")
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)
print(cm)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    linewidths=1,
    square=True,
    cbar=False
)

plt.title(
    "Confusion Matrix",
    fontsize=15
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()

plt.savefig(
    "screenshots/confusion_matrix_logistic.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        prediction
    )
)

# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n")
print("=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)

print(f"Dataset Size      : {len(data)} Samples")
print(f"Features          : {X.shape[1]}")
print(f"Training Samples  : {len(X_train)}")
print(f"Testing Samples   : {len(X_test)}")
print(f"Model             : Logistic Regression")
print(f"Accuracy          : {accuracy*100:.2f}%")

print("\nGenerated Screenshots")

print("----------------------------")

print("✓ class_distribution.png")
print("✓ feature_distribution.png")
print("✓ heatmap.png")
print("✓ pairplot.png")
print("✓ confusion_matrix_logistic.png")

print("\n")
print("=" * 60)
print("EDA & LOGISTIC REGRESSION COMPLETED")
print("=" * 60)