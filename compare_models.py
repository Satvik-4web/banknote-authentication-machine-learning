import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================================================
# CREATE REQUIRED FOLDERS
# ==========================================================

os.makedirs("models", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)

# ==========================================================
# LOAD DATASET
# ==========================================================

data = pd.read_csv(
    "dataset/dataset.txt",
    header=None
)

data.columns = [
    "Variance",
    "Skewness",
    "Curtosis",
    "Entropy",
    "Class"
]

print("="*70)
print("BANKNOTE AUTHENTICATION DATASET")
print("="*70)

print("\nFirst Five Rows\n")
print(data.head())

print("\nDataset Shape")
print(data.shape)

print("\nMissing Values")
print(data.isnull().sum())

print("\nClass Distribution")
print(data["Class"].value_counts())

# ==========================================================
# CLASS DISTRIBUTION
# ==========================================================

plt.figure(figsize=(6,4))

sns.countplot(
    data=data,
    x="Class",
    color="#4C72B0",
)

plt.title(
    "Distribution of Genuine and Counterfeit Banknotes",
    fontsize=14
)

plt.xlabel("Class")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    "screenshots/class_distribution.png",
    dpi=300
)

plt.show()

# ==========================================================
# FEATURE DISTRIBUTION
# ==========================================================

data.hist(
    figsize=(12,8),
    bins=20
)

plt.suptitle(
    "Feature Distribution",
    fontsize=16
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig(
    "screenshots/feature_distribution.png",
    dpi=300
)

plt.show()

# ==========================================================
# HEATMAP
# ==========================================================

plt.figure(figsize=(8,6))

sns.heatmap(
    data.corr(),
    annot=True,
    cmap="coolwarm",
    linewidths=0.5,
    square=True
)

plt.title(
    "Correlation Heatmap",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    "screenshots/heatmap.png",
    dpi=300
)

plt.show()

# ==========================================================
# PAIRPLOT
# ==========================================================

pair = sns.pairplot(
    data,
    hue="Class",
    diag_kind="hist"
)

pair.fig.suptitle(
    "Pair Plot",
    y=1.02
)

pair.savefig(
    "screenshots/pairplot.png",
    dpi=300
)

plt.show()

# ==========================================================
# FEATURES & TARGET
# ==========================================================

X = data.drop(
    "Class",
    axis=1
)

y = data["Class"]

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
print("="*70)
print("TRAIN TEST SPLIT")
print("="*70)

print("Training :", X_train.shape)
print("Testing  :", X_test.shape)
# ==========================================================
# MACHINE LEARNING MODELS
# ==========================================================

models = {

    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        random_state=42
    ),

    "K-Nearest Neighbors": KNeighborsClassifier(),

    "Support Vector Machine": SVC()

}

# ==========================================================
# MODEL COMPARISON
# ==========================================================

results = {}

cv_results = {}

trained_models = {}

print("\n")
print("="*70)
print("TRAINING MACHINE LEARNING MODELS")
print("="*70)

for name, model in models.items():

    print(f"\n{name}")
    print("-"*50)

    # -------------------------------
    # Train Model
    # -------------------------------

    model.fit(
        X_train,
        y_train
    )

    # -------------------------------
    # Prediction
    # -------------------------------

    prediction = model.predict(
        X_test
    )

    # -------------------------------
    # Accuracy
    # -------------------------------

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    # -------------------------------
    # Cross Validation
    # -------------------------------

    cv = cross_val_score(
        model,
        X,
        y,
        cv=5
    )

    results[name] = accuracy

    cv_results[name] = cv.mean()

    trained_models[name] = model

    print(f"Accuracy              : {accuracy*100:.2f}%")
    print(f"Cross Validation Mean : {cv.mean()*100:.2f}%")
    print(f"Standard Deviation    : {cv.std()*100:.2f}%")

# ==========================================================
# SORT RESULTS
# ==========================================================

sorted_results = sorted(

    results.items(),

    key=lambda x: x[1],

    reverse=True

)

# ==========================================================
# FINAL RESULTS TABLE
# ==========================================================

print("\n")
print("="*90)
print("MODEL COMPARISON")
print("="*90)

print(
    f"{'Rank':<6}"
    f"{'Model':<30}"
    f"{'Accuracy':<18}"
    f"{'Cross Validation'}"
)

print("-"*90)

for rank, (model_name, accuracy) in enumerate(

        sorted_results,

        start=1):

    print(

        f"{rank:<6}"

        f"{model_name:<30}"

        f"{accuracy*100:<18.2f}"

        f"{cv_results[model_name]*100:.2f}"

    )

# ==========================================================
# BEST MODEL
# ==========================================================

best_accuracy = max(results.values())

best_models = [

    model

    for model, acc in results.items()

    if acc == best_accuracy

]

print("\n")
print("="*70)
print("BEST PERFORMING MODEL(S)")
print("="*70)

for model in best_models:

    print(

        f"{model} : {best_accuracy*100:.2f}%"

    )

# Select first best model for saving

best_model_name = best_models[0]

best_model = trained_models[best_model_name]

prediction = best_model.predict(
    X_test
)
# ==========================================================
# SAVE BEST MODEL
# ==========================================================

joblib.dump(
    best_model,
    "models/currency_model.pkl"
)

print("\n")
print("=" * 70)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 70)
print("Location : models/currency_model.pkl")

# ==========================================================
# SAVE MODEL COMPARISON TABLE
# ==========================================================

comparison = pd.DataFrame({

    "Model": list(results.keys()),

    "Accuracy (%)": [

        accuracy * 100

        for accuracy in results.values()

    ],

    "Cross Validation (%)": [

        cv * 100

        for cv in cv_results.values()

    ]

})

comparison = comparison.sort_values(

    by="Accuracy (%)",

    ascending=False

)

comparison.to_csv(

    "screenshots/model_comparison.csv",

    index=False

)

# ==========================================================
# ACCURACY COMPARISON GRAPH
# ==========================================================

names = comparison["Model"]

scores = comparison["Accuracy (%)"]

plt.figure(figsize=(10,6))

bars = plt.bar(

    names,

    scores

)

plt.title(

    "Accuracy Comparison of Machine Learning Models",

    fontsize=15,

    fontweight="bold"

)

plt.xlabel("Machine Learning Models")

plt.ylabel("Accuracy (%)")

plt.ylim(95,101)

plt.grid(

    axis="y",

    linestyle="--",

    alpha=0.3

)

for bar in bars:

    plt.text(

        bar.get_x()+bar.get_width()/2,

        bar.get_height()+0.08,

        f"{bar.get_height():.2f}%",

        ha="center",

        fontsize=10

    )

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(

    "screenshots/accuracy_comparison.png",

    dpi=300,

    bbox_inches="tight"

)

plt.show()

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(

    y_test,

    prediction

)

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

    f"Confusion Matrix ({best_model_name})",

    fontsize=14,

    fontweight="bold"

)

plt.xlabel("Predicted Label")

plt.ylabel("Actual Label")

plt.tight_layout()

plt.savefig(

    "screenshots/confusion_matrix.png",

    dpi=300,

    bbox_inches="tight"

)

plt.show()

# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(

    classification_report(

        y_test,

        prediction

    )

)

# ==========================================================
# FINAL PROJECT SUMMARY
# ==========================================================

print("\n")
print("=" * 70)
print("PROJECT SUMMARY")
print("=" * 70)

print(f"Dataset Size           : {len(data)}")
print(f"Number of Features     : {X.shape[1]}")
print(f"Models Compared        : {len(models)}")
print(f"Best Model             : {best_model_name}")
print(f"Test Accuracy          : {results[best_model_name]*100:.2f}%")
print(f"Cross Validation Score : {cv_results[best_model_name]*100:.2f}%")

print("\nGenerated Files")

print("-------------------------")

print("✓ models/currency_model.pkl")
print("✓ screenshots/class_distribution.png")
print("✓ screenshots/feature_distribution.png")
print("✓ screenshots/heatmap.png")
print("✓ screenshots/pairplot.png")
print("✓ screenshots/accuracy_comparison.png")
print("✓ screenshots/confusion_matrix.png")
print("✓ screenshots/model_comparison.csv")

print("\n")
print("=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)