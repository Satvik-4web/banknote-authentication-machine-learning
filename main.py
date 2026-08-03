import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# ==========================
# LOAD DATASET
# ==========================

data = pd.read_csv("dataset/dataset.txt", header=None)

data.columns = [
    "Variance",
    "Skewness",
    "Curtosis",
    "Entropy",
    "Class"
]

# ==========================
# DATA OVERVIEW
# ==========================

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

# =====================================================
# GRAPH 1 : CLASS DISTRIBUTION
# =====================================================

plt.figure(figsize=(6,4))

sns.countplot(data=data, x="Class")

plt.title("Distribution of Real and Fake Bank Notes")
plt.xlabel("Class (0 = Genuine, 1 = Fake)")
plt.ylabel("Count")

plt.show()

# =====================================================
# GRAPH 2 : HISTOGRAMS
# =====================================================

data.hist(figsize=(12,8))

plt.suptitle("Feature Distributions", fontsize=16)

plt.tight_layout()

plt.show()

# =====================================================
# GRAPH 3 : CORRELATION HEATMAP
# =====================================================

plt.figure(figsize=(8,6))

sns.heatmap(
    data.corr(),
    annot=True,
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap")

plt.show()

# =====================================================
# GRAPH 4 : PAIRPLOT
# =====================================================

pair_plot = sns.pairplot(
    data,
    hue="Class",
    diag_kind="hist"
)

pair_plot.fig.suptitle(
    "Pair Plot of Bank Note Features",
    y=1.02
)

plt.show()
# =====================================================
# DATA PREPROCESSING
# =====================================================

X = data.drop("Class", axis=1)

y = data["Class"]

print("=" * 60)
print("FEATURES (X)")
print("=" * 60)
print(X.head())

print("\n")

print("=" * 60)
print("TARGET (y)")
print("=" * 60)
print(y.head())
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("=" * 60)
print("TRAINING DATA SHAPE")
print("=" * 60)
print(X_train.shape)
print(y_train.shape)

print("\n")

print("=" * 60)
print("TESTING DATA SHAPE")
print("=" * 60)
print(X_test.shape)
print(y_test.shape)
# =====================================================
# LOGISTIC REGRESSION MODEL
# =====================================================

model = LogisticRegression()
model.fit(X_train, y_train)
print("=" * 60)
print("MODEL TRAINED SUCCESSFULLY")
print("=" * 60)

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)

print("=" * 60)
print("MODEL ACCURACY")
print("=" * 60)
print(f"Accuracy : {accuracy * 100:.2f}%")
# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(y_test, prediction)

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)
print(cm)
# =====================================================
# CLASSIFICATION REPORT
# =====================================================

print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(classification_report(y_test, prediction))