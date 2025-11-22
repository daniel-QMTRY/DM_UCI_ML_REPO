# src/03_diabetes_analysis_plots.py  (DIABETES READMISSION – "wow" analysis)

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "diabetes_uci_merged_binary.csv"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def train_model():
    """Train logistic regression on diabetes readmission and return model + splits + preds."""
    print("DATA_PATH =", DATA_PATH)
    print("Exists?   =", DATA_PATH.exists())
    
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Target = 30-day readmission flag (0/1)
    y = df["readmit_30d"]
    X = df.drop(columns=["readmit_30d"])

    # Train/test split (stratified for imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Categorical vs numeric cols
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_pipe = Pipeline(steps=[("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", numeric_pipe, num_cols),
        ]
    )

    clf = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    n_jobs=-1,
                    class_weight="balanced",  # up-weight readmit=1
                    solver="lbfgs",
                ),
            ),
        ]
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # For ROC we need predicted probabilities for class 1
    y_proba = clf.predict_proba(X_test)[:, 1]

    return clf, X_train, X_test, y_train, y_test, y_pred, y_proba


def plot_confusion_matrix(y_test, y_pred):
    labels = [0, 1]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=["No readmit", "Readmit 30d"],
        yticklabels=["No readmit", "Readmit 30d"],
    )
    plt.title("Diabetes – Normalized Confusion Matrix (Readmit 30d)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    out_path = FIG_DIR / "diabetes_confusion_matrix.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix → {out_path}")


def plot_roc_curve(y_test, y_proba):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall for readmit=1)")
    plt.title("Diabetes – ROC Curve for 30-Day Readmission")
    plt.legend(loc="lower right")
    plt.tight_layout()
    out_path = FIG_DIR / "diabetes_roc_curve.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved ROC curve → {out_path}")


def plot_positive_class_metrics(y_test, y_pred):
    """Bar chart of precision/recall/F1 for the readmit=1 class."""
    report = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    pos = report["1"]  # metrics for class "1"

    metrics = ["precision", "recall", "f1-score"]
    values = [pos[m] for m in metrics]

    plt.figure(figsize=(5, 4))
    sns.barplot(x=metrics, y=values)
    plt.ylim(0, 1.0)
    plt.title("Diabetes – Metrics for Readmit 30d (Class 1)")
    plt.ylabel("Score")
    plt.tight_layout()
    out_path = FIG_DIR / "diabetes_readmit1_metrics.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved positive-class metrics → {out_path}")


def main():
    clf, X_train, X_test, y_train, y_test, y_pred, y_proba = train_model()

    print("=== Diabetes 30-day readmission – detailed classification report ===")
    print(classification_report(y_test, y_pred, digits=3))

    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_proba)
    plot_positive_class_metrics(y_test, y_pred)


if __name__ == "__main__":
    main()
