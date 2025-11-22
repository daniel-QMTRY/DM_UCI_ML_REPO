# src/02_baseline_model.py  (HAR SMARTPHONES BASELINE – v1)

from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "har_merged.csv"


def main():
    # 1) Load merged HAR data
    #    -> columns include: subject_id, 0..560 feature cols, activity_id, activity
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # ----- THIS is your "target" -----
    # prediction column = activity (e.g., WALKING, SITTING, etc.)
    y = df["activity"]

    # features = all numeric sensor columns, but drop ID-ish columns
    X = df.drop(columns=["subject_id", "activity_id", "activity"])

    # 2) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,   # keep class proportions
    )

    # all remaining columns are numeric features
    num_cols = X.columns.tolist()

    # 3) Scale features + logistic regression
    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    n_jobs=-1,
                    class_weight="balanced",  # activities that are rarer get more weight
                    multi_class="auto",
                ),
            ),
        ]
    )

    # 4) Train + evaluate
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("=== HAR – smartphone activity recognition (logistic regression) ===")
    print(classification_report(y_test, y_pred, digits=3))


if __name__ == "__main__":
    main()
