# src/01_fetch_and_save.py  (HAR SMARTPHONES PROJECT)

from pathlib import Path
from urllib.request import urlretrieve
import zipfile
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# UCI HAR dataset (official UCI zip)
HAR_ZIP_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/"
    "UCI%20HAR%20Dataset.zip"
)
HAR_ZIP_PATH = DATA_DIR / "UCI_HAR_Dataset.zip"
EXTRACTED_DIR = DATA_DIR / "UCI HAR Dataset"


def download_and_extract():
    """Download the HAR zip once and extract it under data/."""
    if not HAR_ZIP_PATH.exists():
        print(f"Downloading HAR dataset to {HAR_ZIP_PATH} ...")
        urlretrieve(HAR_ZIP_URL, HAR_ZIP_PATH)
        print("Download complete.")
    else:
        print("HAR zip already exists, skipping download.")

    if not EXTRACTED_DIR.exists():
        print(f"Extracting into {DATA_DIR} ...")
        with zipfile.ZipFile(HAR_ZIP_PATH, "r") as zf:
            zf.extractall(DATA_DIR)
        print("Extraction complete.")
    else:
        print("Extracted folder already exists, skipping extraction.")


def load_split(split: str):
    """
    Load one split (train or test) from the extracted UCI HAR dataset.

    Returns:
        X: DataFrame of features (561 columns)
        y: DataFrame with 'activity_id'
        subjects: DataFrame with 'subject_id'
    """
    split_dir = EXTRACTED_DIR / split
    X_path = split_dir / f"X_{split}.txt"
    y_path = split_dir / f"y_{split}.txt"
    subject_path = split_dir / f"subject_{split}.txt"

    # Features are whitespace-delimited, no header
    X = pd.read_csv(X_path, delim_whitespace=True, header=None)
    y = pd.read_csv(y_path, header=None, names=["activity_id"])
    subjects = pd.read_csv(subject_path, header=None, names=["subject_id"])

    return X, y, subjects


def main():
    # 1) Get the raw UCI HAR files under data/
    download_and_extract()

    # 2) Load train and test splits, then stack them
    X_train, y_train, subj_train = load_split("train")
    X_test, y_test, subj_test = load_split("test")

    X = pd.concat([X_train, X_test], ignore_index=True)
    y = pd.concat([y_train, y_test], ignore_index=True)
    subjects = pd.concat([subj_train, subj_test], ignore_index=True)

    # 3) Map activity IDs (1..6) to labels (WALKING, SITTING, etc.)
    activity_labels_path = EXTRACTED_DIR / "activity_labels.txt"
    activity_labels = pd.read_csv(
        activity_labels_path,
        delim_whitespace=True,
        header=None,
        names=["activity_id", "activity"],
    )
    y = y.merge(activity_labels, on="activity_id", how="left")

    # 4) Save raw-style CSVs (features only, targets only)
    X.to_csv(DATA_DIR / "har_features_raw.csv", index=False)
    y.to_csv(DATA_DIR / "har_targets_raw.csv", index=False)

    # 5) Save merged CSV: subject + features + activity columns
    df = subjects.copy()
    df = pd.concat([df, X], axis=1)
    df["activity_id"] = y["activity_id"]
    df["activity"] = y["activity"]

    df.to_csv(DATA_DIR / "har_merged.csv", index=False)

    print("Saved:")
    print(" - data/har_features_raw.csv")
    print(" - data/har_targets_raw.csv")
    print(" - data/har_merged.csv")
    print("\nSample targets:")
    print(y.head())


if __name__ == "__main__":
    main()
