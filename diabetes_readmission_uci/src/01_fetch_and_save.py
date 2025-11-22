# src/01_fetch_and_save.py  (OBESITY PROJECT)

from ucimlrepo import fetch_ucirepo
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # Fetch the obesity dataset from UCI (ID 544)
    obesity = fetch_ucirepo(id=544)

    # Features and targets
    X = obesity.data.features.copy()
    y = obesity.data.targets.copy()

    # The class label column is called "NObesity"
    if "NObesity" not in y.columns:
        raise ValueError(f"Expected 'NObesity' in targets, but found {y.columns.tolist()}")

    target = y["NObesity"]
    target.name = "NObesity"

    # Save raw-style CSVs
    X.to_csv(DATA_DIR / "obesity_features_raw.csv", index=False)
    y.to_csv(DATA_DIR / "obesity_targets_raw.csv", index=False)

    # Save merged version (features + label)
    df = X.copy()
    df["NObesity"] = target
    df.to_csv(DATA_DIR / "obesity_merged.csv", index=False)

    print("Saved:")
    print(" - data/obesity_features_raw.csv")
    print(" - data/obesity_targets_raw.csv")
    print(" - data/obesity_merged.csv")

    # Quick peek at the target
    print("\nTarget sample:")
    print(y.head())


if __name__ == "__main__":
    main()
