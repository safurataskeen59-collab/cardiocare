# ============================================================
# CARDIOCARE - CARDIOVASCULAR DISEASE PREDICTION
# Data analysis, visualization, model comparison, and artifact saving
# ============================================================

import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# -------------------- PROJECT PATHS --------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "data cardio"  # Add ".csv" here if your file has that extension.
MODEL_DIR = BASE_DIR / "models"
PLOT_DIR = BASE_DIR / "plots"
RESULTS_DIR = BASE_DIR / "results"

for folder in (MODEL_DIR, PLOT_DIR, RESULTS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "age_years", "gender", "height", "weight", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active", "BMI"
]
TARGET = "cardio"


def save_plot(filename):
    """Save the current Matplotlib figure in the plots directory."""
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 65)
    print("CARDIOCARE - CARDIOVASCULAR DISEASE PREDICTION")
    print("=" * 65)

    # -------------------- LOAD AND VALIDATE DATA --------------------
    print("\nLoading dataset:", DATA_PATH)
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n"
            "Place the dataset in the data folder and name it 'data cardio', "
            "or update DATA_PATH near the top of analysis.py to match the filename."
        )

    df = pd.read_csv(DATA_PATH, sep=";")
    print("Original dataset shape:", df.shape)
    print("\nFirst five rows:\n", df.head())
    print("\nColumns:\n", df.columns.tolist())
    print("\nMissing values before cleaning:\n", df.isnull().sum())
    print("Duplicate rows before cleaning:", df.duplicated().sum())

    required = {
        "age", "gender", "height", "weight", "ap_hi", "ap_lo",
        "cholesterol", "gluc", "smoke", "alco", "active", TARGET
    }
    missing_columns = sorted(required - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    # Convert required columns to numeric; invalid values become NaN and are removed.
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    rows_before = len(df)
    df = df.drop_duplicates()
    df = df.dropna(subset=list(required))

    # Keep records with valid target and category encodings.
    df = df[df[TARGET].isin([0, 1])]
    df = df[df["gender"].isin([1, 2])]
    for col in ["smoke", "alco", "active"]:
        
        df = df[df[col].isin([0, 1])]

    # Validate height, weight, and blood pressure values.
    df = df[df["height"].between(100, 250)]
    df = df[df["weight"].between(30, 250)]
    df = df[df["ap_hi"].between(60, 250)]
    df = df[df["ap_lo"].between(40, 200)]
    df = df[df["ap_hi"] >= df["ap_lo"]]
    df = df[df["cholesterol"].isin([1, 2, 3])]
    df = df[df["gluc"].isin([1, 2, 3])]

    # Convert age from days into years and calculate BMI.
    df["age_years"] = (df["age"] / 365.25).round(1)
    df = df[df["age_years"].between(1, 120)]
    df["BMI"] = df["weight"] / ((df["height"] / 100) ** 2)
    df["BMI"] = df["BMI"].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES + [TARGET])

    print("\nRows removed during cleaning:", rows_before - len(df))
    print("Cleaned dataset shape:", df.shape)
    print("\nMissing values after cleaning:\n", df[FEATURES + [TARGET]].isnull().sum())

    if df.empty:
        raise ValueError("No valid rows remain after cleaning. Check your dataset.")

    # -------------------- EXPLORATORY DATA ANALYSIS --------------------
    print("\nCreating visualizations...")

    # 1. Target distribution
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x=TARGET)
    plt.title("Cardiovascular Disease Distribution")
    plt.xlabel("Cardio (0 = No, 1 = Yes)")
    plt.ylabel("Number of people")
    save_plot("01_target_distribution.png")

    # 2. Age distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="age_years", bins=30, kde=True)
    plt.title("Age Distribution")
    save_plot("02_age_distribution.png")

    # 3. Gender distribution by target
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="gender", hue=TARGET)
    plt.title("Gender Distribution by Cardiovascular Disease")
    save_plot("03_gender_distribution.png")

    # 4. Correlation heatmap
    plt.figure(figsize=(12, 9))
    sns.heatmap(df[FEATURES + [TARGET]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    save_plot("04_correlation_heatmap.png")

    # 5. Feature distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, col in zip(axes.flat, ["age_years", "height", "weight", "ap_hi", "ap_lo", "BMI"]):
        sns.histplot(data=df, x=col, bins=25, ax=ax)
        ax.set_title(f"{col} Distribution")
    save_plot("05_feature_distributions.png")

    # 6. Blood pressure distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="ap_hi", color="red", label="Systolic", alpha=0.5)
    sns.histplot(data=df, x="ap_lo", color="blue", label="Diastolic", alpha=0.5)
    plt.legend()
    plt.title("Blood Pressure Distribution")
    save_plot("06_blood_pressure_distribution.png")

    # 7. BMI distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="BMI", bins=30, kde=True)
    plt.title("Body Mass Index Distribution")
    save_plot("07_bmi_distribution.png")

    # 8. Cholesterol distribution
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="cholesterol")
    plt.title("Cholesterol Levels (1 = Normal, 2 = Above Normal, 3 = High)")
    save_plot("08_cholesterol_distribution.png")

    # 9. Glucose distribution
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="gluc")
    plt.title("Glucose Levels")
    save_plot("09_glucose_distribution.png")

    # 10. Smoking distribution
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="smoke")
    plt.title("Smoking Status")
    save_plot("10_smoking_distribution.png")

    # 11. Activity distribution
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="active")
    plt.title("Physical Activity Status")
    save_plot("11_activity_distribution.png")

    # 12. Alcohol distribution
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="alco")
    plt.title("Alcohol Consumption")
    save_plot("12_alcohol_distribution.png")

    # 13. Age by target
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=TARGET, y="age_years")
    plt.title("Age by Cardiovascular Disease Status")
    save_plot("13_age_by_target.png")

    # 14. BMI by target
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=TARGET, y="BMI")
    plt.title("BMI by Cardiovascular Disease Status")
    save_plot("14_bmi_by_target.png")

    print("EDA visualizations saved in:", PLOT_DIR)

    # -------------------- MODEL TRAINING --------------------
    print("\nPreparing data for machine learning...")
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=42))
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"
        ),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(probability=True, random_state=42))
        ])
    }

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    fitted_models = {}

    print("\nComparing five machine learning models...")
    for name, model in models.items():
        print(f"Training and evaluating: {name}")
        model.fit(X_train, y_train)
        fitted_models[name] = model

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1 Score": f1_score(y_test, predictions, zero_division=0),
            "ROC AUC": roc_auc_score(y_test, probabilities)
        })

    results_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False)
    results_df.to_csv(RESULTS_DIR / "model_results.csv", index=False)
    print("\nTest-set model comparison:\n", results_df.to_string(index=False))

    # Cross-validation comparison
    print("\nRunning 5-fold cross-validation...")
    cv_rows = []
    for name, model in models.items():
        scores = cross_validate(
            model, X, y, cv=cv, scoring=scoring, n_jobs=-1,
            error_score="raise"
        )
        cv_rows.append({
            "Model": name,
            "Accuracy": scores["test_accuracy"].mean(),
            "Precision": scores["test_precision"].mean(),
            "Recall": scores["test_recall"].mean(),
            "F1 Score": scores["test_f1"].mean(),
            "ROC AUC": scores["test_roc_auc"].mean()
        })

    cv_df = pd.DataFrame(cv_rows).sort_values("F1 Score", ascending=False)
    cv_df.to_csv(RESULTS_DIR / "cross_validation_results.csv", index=False)
    print("\nCross-validation results:\n", cv_df.to_string(index=False))

    # Select the best model using cross-validation F1 score.
    best_name = cv_df.iloc[0]["Model"]
    best_model = models[best_name]
    best_model.fit(X_train, y_train)
    best_predictions = best_model.predict(X_test)
    best_probabilities = best_model.predict_proba(X_test)[:, 1]

    print("\nBest model:", best_name)
    print("\nClassification report:\n",
          classification_report(y_test, best_predictions, zero_division=0))

    report = classification_report(
        y_test, best_predictions, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(
        RESULTS_DIR / "classification_report.csv", index=True
    )

    # Confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        confusion_matrix(y_test, best_predictions),
        annot=True, fmt="d", cmap="Blues",
        xticklabels=["No disease (0)", "Disease (1)"],
        yticklabels=["No disease (0)", "Disease (1)"]
    )
    plt.title(f"Confusion Matrix - {best_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    save_plot("15_confusion_matrix.png")

    # ROC curves for all trained models
    plt.figure(figsize=(8, 6))
    for name, model in fitted_models.items():
        probabilities = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        auc_value = roc_auc_score(y_test, probabilities)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_value:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Model Comparison")
    plt.legend(loc="lower right", fontsize=8)
    save_plot("16_roc_curves.png")

    # Model comparison plot
    comparison = results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score"]]
    comparison.plot(kind="bar", figsize=(12, 6))
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=25, ha="right")
    save_plot("17_model_comparison.png")

    # Feature importance from Random Forest
    rf_model = fitted_models["Random Forest"]
    importance = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": rf_model.feature_importances_
    }).sort_values("Importance", ascending=False)
    importance.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=importance, x="Importance", y="Feature")
    plt.title("Random Forest Feature Importance")
    save_plot("18_feature_importance.png")

    # -------------------- SAVE MODEL AND METADATA --------------------
    joblib.dump(best_model, MODEL_DIR / "best_cardio_model.pkl")
    joblib.dump(FEATURES, MODEL_DIR / "features.pkl")
    joblib.dump({
        "best_model": best_name,
        "features": FEATURES,
        "test_accuracy": float(accuracy_score(y_test, best_predictions)),
        "test_f1": float(f1_score(y_test, best_predictions, zero_division=0))
    }, MODEL_DIR / "model_metadata.pkl")

    print("\nSaved model and features in:", MODEL_DIR)
    print("Saved results in:", RESULTS_DIR)
    print("Saved plots in:", PLOT_DIR)
    print("\nAnalysis completed successfully!")
    print("Note: This educational model is not a medical diagnosis tool.")


if __name__ == "__main__":
    main()
