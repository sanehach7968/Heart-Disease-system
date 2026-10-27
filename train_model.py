
import os
import json
import joblib
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

# Project folders
os.makedirs("artifacts", exist_ok=True)
os.makedirs("artifacts/plots", exist_ok=True)

# Download sample data if it is not available locally
DATA_URL = (
    "https://raw.githubusercontent.com/"
    "plotly/datasets/master/heart.csv"
)

if not os.path.exists("heart.csv"):
    print("Downloading sample heart disease dataset...")
    urllib.request.urlretrieve(DATA_URL, "heart.csv")

df = pd.read_csv("heart.csv")
df.columns = df.columns.str.strip().str.lower()

# Support common alternative column names
df = df.rename(columns={
    "num": "target",
    "condition": "target",
    "resting_bp": "trestbps",
    "resting_blood_pressure": "trestbps",
    "serum_cholesterol": "chol",
    "maximum_heart_rate": "thalach",
    "fasting_blood_sugar": "fbs",
    "exercise_induced_angina": "exang",
    "st_depression": "oldpeak",
    "number_of_major_vessels": "ca"
})

# Required features in the common Cleveland-style dataset
features = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]

if "target" not in df.columns:
    raise ValueError("Dataset must contain a target column.")

missing = [c for c in features if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing columns: {missing}. "
        "Use the expected heart disease dataset."
    )

# Convert columns to numeric and treat missing markers as NaN
for col in features + ["target"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.replace("?", np.nan)
df = df.dropna(subset=["target"])

# Remove rows with invalid target values
df = df[df["target"].isin([0, 1, 2, 3, 4])].copy()

# Standard Cleveland dataset: 0 = no disease, 1-4 = disease
df["target"] = (df["target"] > 0).astype(int)

X = df[features]
y = df["target"]

if y.nunique() != 2 or y.value_counts().min() < 2:
    raise ValueError("Dataset must contain both target classes.")

print("\nDataset shape:", df.shape)
print("\nTarget distribution:")
print(y.value_counts())

# Split before fitting preprocessing to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Models: impute missing values and scale numeric features
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(probability=True, random_state=42),
    "Naive Bayes": GaussianNB()
}

results = []
roc_data = {}

for name, classifier in models.items():
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": auc
    })

    roc_data[name] = roc_curve(y_test, probabilities)

    safe_name = name.lower().replace(" ", "_")
    joblib.dump(model, f"artifacts/{safe_name}.joblib")

    # Save each model's confusion matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        f"artifacts/plots/{safe_name}_confusion_matrix.png"
    )
    plt.close()

    print(f"\n{name}")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

# Compare evaluation metrics
results_df = pd.DataFrame(results)
results_df.to_csv("artifacts/model_comparison.csv", index=False)

print("\nMODEL COMPARISON")
print(results_df.round(3).to_string(index=False))

# Save feature order and target information
with open("artifacts/metadata.json", "w") as f:
    json.dump({
        "features": features,
        "target": "target",
        "positive_class": 1,
        "dataset_rows": int(len(df)),
        "test_size": 0.20,
        "random_state": 42
    }, f, indent=4)

# Model comparison graph
results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
].plot(kind="bar", figsize=(11, 6))

plt.title("Heart Disease Model Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("artifacts/plots/model_comparison.png")
plt.close()

# Correlation heatmap
plt.figure(figsize=(12, 9))
sns.heatmap(
    df[features + ["target"]].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("artifacts/plots/correlation_heatmap.png")
plt.close()

# ROC curves
plt.figure(figsize=(8, 6))
for name, (fpr, tpr, thresholds) in roc_data.items():
    auc = results_df.loc[
        results_df["Model"] == name, "ROC AUC"
    ].iloc[0]
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("artifacts/plots/roc_comparison.png")
plt.close()

print("\nTraining completed successfully.")
print("Models and reports saved in the artifacts folder.")
