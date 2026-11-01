from pathlib import Path
import argparse
import pandas as pd

EXPECTED_COLUMNS = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
]

def main(path: str):
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    missing_cols = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
    extra_cols = sorted(set(df.columns) - set(EXPECTED_COLUMNS))

    print("=" * 78)
    print("DATA REGISTRATION / VALIDATION SUMMARY")
    print("=" * 78)
    print(f"Path              : {csv_path.resolve()}")
    print(f"Rows              : {df.shape[0]}")
    print(f"Columns           : {df.shape[1]}")
    print(f"Duplicate rows    : {df.duplicated().sum()}")
    print(f"Total missing     : {int(df.isna().sum().sum())}")
    print(f"Missing columns   : {missing_cols if missing_cols else 'None'}")
    print(f"Unexpected columns: {extra_cols if extra_cols else 'None'}")

    if missing_cols:
        raise ValueError("Validation failed. Missing columns: " + ", ".join(missing_cols))

    print("\nTarget distribution:")
    print(df["ProdTaken"].value_counts(dropna=False).sort_index())

    print("\nMissing-value summary:")
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    print(miss if not miss.empty else "No missing values")

    print("\nVALIDATION STATUS: PASSED")
    print("=" * 78)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/tourism.csv")
    args = parser.parse_args()
    main(args.input)
