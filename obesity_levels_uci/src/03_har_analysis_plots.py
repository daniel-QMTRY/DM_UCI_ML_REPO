# src/03_har_analysis_plots.py  (HAR – "wow" analysis)

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "har_merged.csv"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def train_model():
    """Train logistic regression on HAR and return model + splits + preds."""
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Target: activity label
    y = df["activity"]

    # Features: drop IDs
    X = df.drop(columns=["subject_id", "activity_id", "activity"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
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
    plt.title("HAR – Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    out_path = FIG_DIR / "har_confusion_matrix.png"
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
        .rename(columns={"index": "activity"})
    )

    plt.figure(figsize=(8, 4))
    sns.barplot(data=df_scores, x="activity", y="f1-score")
    plt.title("HAR – F1-score by Activity")
    plt.ylabel("F1-score")
    plt.xlabel("Activity")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    out_path = FIG_DIR / "har_f1_by_activity.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved F1 scores plot → {out_path}")


def plot_pca_embedding(X, y):
    """2D PCA of a subset of samples, colored by activity."""
    max_points = 3000
    if len(X) > max_points:
        rs = np.random.RandomState(42)
        idx = rs.choice(len(X), size=max_points, replace=False)
        X_sample = X.iloc[idx]
        y_sample = y.iloc[idx]
    else:
        X_sample = X
        y_sample = y

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)

    pca = PCA(n_components=2, random_state=42)
    X_embedded = pca.fit_transform(X_scaled)

    embed_df = pd.DataFrame(
        {
            "PC1": X_embedded[:, 0],
            "PC2": X_embedded[:, 1],
            "activity": y_sample.values,
        }
    )

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=embed_df,
        x="PC1",
        y="PC2",
        hue="activity",
        s=15,
        alpha=0.7,
        edgecolor=None,
    )
    plt.title("HAR – 2D PCA Embedding of Activities")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.tight_layout()
    out_path = FIG_DIR / "har_pca_embedding.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved PCA embedding → {out_path}")


def main():
    clf, X_train, X_test, y_train, y_test, y_pred = train_model()

    print("=== HAR – detailed classification report ===")
    print(classification_report(y_test, y_pred, digits=3))

    plot_confusion_matrix(y_test, y_pred)
    plot_f1_scores(y_test, y_pred)

    X_full = pd.concat([X_train, X_test], ignore_index=True)
    y_full = pd.concat([y_train, y_test], ignore_index=True)
    plot_pca_embedding(X_full, y_full)


if __name__ == "__main__":
    main()
