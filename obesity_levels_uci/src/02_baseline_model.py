# src/02_baseline_model.py  (OBESITY LEVELS BASELINE – v1)

from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "obesity_merged.csv"


def main():
    # 1) Load merged data
    #    -> this file was created by 01_fetch_and_save.py
    #       and contains all features + an "obesity_level" column
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # ----- THIS is your "target" -----
    # prediction column = obesity_level
    y = df["obesity_level"]

    # features = everything else
    X = df.drop(columns=["obesity_level"])

    # 2) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,   # keep class proportions roughly the same
    )

    # 3) Work out which columns are categorical vs numeric
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    numeric_pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", numeric_pipe, num_cols),
        ]
    )

    # 4) Multiclass logistic regression baseline
    clf = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    n_jobs=-1,
                    class_weight="balanced",  # handle class imbalance
                    multi_class="auto",
                ),
            ),
        ]
    )

    # 5) Train + evaluate
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("=== Obesity levels – logistic regression (balanced) ===")
    print(classification_report(y_test, y_pred, digits=3))


if __name__ == "__main__":
    main()
