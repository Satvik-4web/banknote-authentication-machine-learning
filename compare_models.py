import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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

# ============================================================
# LOAD DATASET
# ============================================================

data = pd.read_csv("dataset/dataset.txt", header=None)

data.columns = [
    "Variance",
    "Skewness",
    "Curtosis",
    "Entropy",
    "Class"
]

print("=" * 70)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 70)
print(data.head())

# ============================================================
# FEATURES & TARGET
# ============================================================

X = data.drop("Class", axis=1)
y = data["Class"]

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n")
print("=" * 70)
print("TRAIN TEST SPLIT")
print("=" * 70)

print("Training Features :", X_train.shape)
print("Training Labels   :", y_train.shape)
print("Testing Features  :", X_test.shape)
print("Testing Labels    :", y_test.shape)

# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "Random Forest": RandomForestClassifier(random_state=42),

    "KNN": KNeighborsClassifier(),

    "SVM": SVC()

}

results = {}
trained_models = {}

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

# ============================================================
# TRAIN ALL MODELS
# ============================================================

for name, model in models.items():

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=5
    )

    results[name] = accuracy
    trained_models[name] = model

    print(f"Accuracy              : {accuracy*100:.2f}%")
    print(f"Cross Validation Mean : {cv_scores.mean()*100:.2f}%")
    print(f"Standard Deviation    : {cv_scores.std()*100:.2f}%")

# ============================================================
# FINAL RANKING
# ============================================================

print("\n")
print("=" * 70)
print("FINAL MODEL RANKING")
print("=" * 70)

sorted_results = sorted(
    results.items(),
    key=lambda x: x[1],
    reverse=True
)

for rank, (model_name, accuracy) in enumerate(
        sorted_results,
        start=1):

    print(f"{rank}. {model_name:<25} {accuracy*100:.2f}%")

# ============================================================
# BEST MODEL
# ============================================================

best_model_name = sorted_results[0][0]
best_model = trained_models[best_model_name]

print("\n")
print("=" * 70)
print("BEST MODEL")
print("=" * 70)

print(f"Model Name : {best_model_name}")
print(f"Accuracy   : {results[best_model_name]*100:.2f}%")

# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    best_model,
    "models/currency_model.pkl"
)

print("\nModel saved successfully!")
print("Location : models/currency_model.pkl")

# ============================================================
# ACCURACY COMPARISON GRAPH
# ============================================================

plt.figure(figsize=(10,6))

bars = plt.bar(
    results.keys(),
    [i*100 for i in results.values()]
)

plt.title("Accuracy Comparison of Machine Learning Models")

plt.xlabel("Machine Learning Models")

plt.ylabel("Accuracy (%)")

plt.ylim(95,101)

plt.xticks(rotation=20)

for bar in bars:

    height = bar.get_height()

    plt.text(
        bar.get_x()+bar.get_width()/2,
        height+0.1,
        f"{height:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "screenshots/accuracy_comparison.png",
    dpi=300
)

plt.show()

# ============================================================
# CONFUSION MATRIX
# ============================================================

prediction = best_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    prediction
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    f"{best_model_name} Confusion Matrix"
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "screenshots/confusion_matrix.png",
    dpi=300
)

plt.show()

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

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

# ============================================================
# PROJECT SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("PROJECT SUMMARY")
print("=" * 70)

print(f"Dataset Size        : {len(data)} Samples")
print(f"Number of Features  : {X.shape[1]}")
print(f"Models Compared     : {len(models)}")
print(f"Best Model          : {best_model_name}")
print(f"Final Accuracy      : {results[best_model_name]*100:.2f}%")
print(f"Saved Model         : models/currency_model.pkl")
print(f"Accuracy Graph      : screenshots/accuracy_comparison.png")
print(f"Confusion Matrix    : screenshots/confusion_matrix.png")

print("\n")
print("=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 70)