from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "ProdTaken"

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Convert empty/blank text cells into NaN.
    df = df.replace(r"^\s*$", np.nan, regex=True)

    # Remove accidental leading/trailing spaces from text values.
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Known label cleanup commonly found in the provided tourism data.
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    if "MaritalStatus" in df.columns:
        df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplicate rows removed: {before - len(df)}")

    # CustomerID is a unique identifier and has no useful business signal.
    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])
        print("Removed unnecessary identifier: CustomerID")

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' is missing.")

    if df[TARGET].isna().any():
        raise ValueError("Target column contains missing values.")

    df[TARGET] = df[TARGET].astype(int)
    invalid = set(df[TARGET].unique()) - {0, 1}
    if invalid:
        raise ValueError(f"Unexpected target values: {invalid}")

    return df

def fill_missing_from_training(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()
    summary = {"numeric_medians": {}, "categorical_modes": {}}

    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

    # Derive imputation values from TRAINING data only to avoid leakage.
    for col in num_cols:
        median = float(X_train[col].median())
        X_train[col] = X_train[col].fillna(median)
        X_test[col] = X_test[col].fillna(median)
        summary["numeric_medians"][col] = median

    for col in cat_cols:
        modes = X_train[col].mode(dropna=True)
        fill_value = modes.iloc[0] if not modes.empty else "Unknown"
        X_train[col] = X_train[col].fillna(fill_value)
        X_test[col] = X_test[col].fillna(fill_value)
        summary["categorical_modes"][col] = str(fill_value)

    return X_train, X_test, summary

def main(input_path: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_path)
    df = clean_dataset(raw)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    X_train, X_test, imputation = fill_missing_from_training(X_train, X_test)

    train_df = X_train.copy()
    train_df[TARGET] = y_train.to_numpy()

    test_df = X_test.copy()
    test_df[TARGET] = y_test.to_numpy()

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    summary_path = output_dir / "preparation_summary.json"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    summary = {
        "original_shape": list(raw.shape),
        "clean_shape_before_split": list(df.shape),
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "split": "80% train / 20% test",
        "target": TARGET,
        "train_missing_cells": int(train_df.isna().sum().sum()),
        "test_missing_cells": int(test_df.isna().sum().sum()),
        "train_target_distribution": {
            str(k): int(v) for k, v in train_df[TARGET].value_counts().sort_index().items()
        },
        "test_target_distribution": {
            str(k): int(v) for k, v in test_df[TARGET].value_counts().sort_index().items()
        },
        "imputation_from_training_only": imputation,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 78)
    print("DATA PREPARATION SUMMARY")
    print("=" * 78)
    print("Split             : 80% train / 20% test")
    print(f"Train shape       : {train_df.shape}")
    print(f"Test shape        : {test_df.shape}")
    print(f"Train missing     : {train_df.isna().sum().sum()}")
    print(f"Test missing      : {test_df.isna().sum().sum()}")
    print("\nTrain target %:")
    print((train_df[TARGET].value_counts(normalize=True).sort_index() * 100).round(2))
    print("\nTest target %:")
    print((test_df[TARGET].value_counts(normalize=True).sort_index() * 100).round(2))
    print("=" * 78)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/tourism.csv")
    parser.add_argument("--output-dir", default="artifacts/data")
    args = parser.parse_args()
    main(args.input, args.output_dir)
