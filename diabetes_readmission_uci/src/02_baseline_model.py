# src/02_baseline_model.py  (DIABETES READMISSION BASELINE – v2)

from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "diabetes_uci_merged_binary.csv"


def main():
    # 1) Load merged data (low_memory=False to avoid DtypeWarning)
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Target + features
    y = df["readmit_30d"]
    X = df.drop(columns=["readmit_30d"])

    # 2) Basic train/test split (stratified for imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 3) Column types
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # Preprocessing: one-hot for categoricals, scaling for numerics
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

    # 4) Baseline model: logistic regression with class_weight to fight imbalance
    clf = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,        # more iterations to avoid convergence warning
                    n_jobs=-1,
                    class_weight="balanced",  # up-weight the minority (readmit=1)
                ),
            ),
        ]
    )

    # 5) Train + evaluate
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("=== Diabetes 30-day readmission – logistic regression (balanced) ===")
    print(classification_report(y_test, y_pred, digits=3))


if __name__ == "__main__":
    main()
