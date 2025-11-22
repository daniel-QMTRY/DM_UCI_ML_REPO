# src/03_obesity_analysis_plots.py  (OBESITY LEVELS – "wow" analysis)

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "obesity_merged.csv"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def train_model():
    """Train logistic regression on obesity dataset and return model + splits + preds."""
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Target column = obesity_level (created in 01_fetch_and_save.py)
    y = df["obesity_level"]
    X = df.drop(columns=["obesity_level"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

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
                    class_weight="balanced",
                    multi_class="auto",
                ),
            ),
        ]
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return clf, X_train, X_test, y_train, y_test, y_pred


def plot_confusion_matrix(y_test, y_pred):
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Obesity Levels – Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    out_path = FIG_DIR / "obesity_confusion_matrix.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix → {out_path}")


def plot_f1_scores(y_test, y_pred):
    report_dict = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    rows = {
        k: v
        for k, v in report_dict.items()
        if k not in ["accuracy", "macro avg", "weighted avg"]
    }

    df_scores = (
        pd.DataFrame(rows)
        .T.reset_index()
        .rename(columns={"index": "obesity_level"})
    )

    plt.figure(figsize=(8, 4))
    sns.barplot(data=df_scores, x="obesity_level", y="f1-score")
    plt.title("Obesity Levels – F1-score by Class")
    plt.ylabel("F1-score")
    plt.xlabel("Obesity Level")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    out_path = FIG_DIR / "obesity_f1_by_class.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved F1 scores plot → {out_path}")


def main():
    clf, X_train, X_test, y_train, y_test, y_pred = train_model()

    print("=== Obesity levels – detailed classification report ===")
    print(classification_report(y_test, y_pred, digits=3))

    plot_confusion_matrix(y_test, y_pred)
    plot_f1_scores(y_test, y_pred)


if __name__ == "__main__":
    main()
