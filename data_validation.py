# === import libraries === 
import os
import pandas as pd
import matplotlib.pyplot as plt

# === global constants ===
## path to data folder
DATA_PATH = "data/"

## required columns
ID_COLS = ["LOG_ID","MRN"]
OUTCOME_COL = ["hypoxemia"]
LAB_COLS = ["crp", "co2", "glucose", "hematocrit", "hemoglobin",
           "leukocytes", "potassium", "sodium", "pH", "lactate"]
DEMO_COLS = ["age", "sex", "bmi", "height", "weight"]
REQUIRED_COLS = ID_COLS + OUTCOME_COL + LAB_COLS + DEMO_COLS

## expected column groups by type
EXPECTED_TYPE_COLS = {"string": ["LOG_ID", "MRN"], 
                      "numeric": LAB_COLS + [col for col in DEMO_COLS if col != "sex"], 
                      "binary": ["hypoxemia", "sex"]}

def check_duplicate_keys(df: pd.DataFrame, key_cols: list[str]) -> None:
    dup_n = df.duplicated(subset=key_cols).sum()

    if dup_n == 0:
        print("No duplicated keys found.")
    else:
        print(f"Found {dup_n} duplicated keys.")

def check_required_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing_cols = [col for col in required_cols if col not in df.columns]

    if len(missing_cols) == 0:
        print("All required columns are present.")
    else:
        print("Missing required columns:")
        print(missing_cols)

def check_expected_column_types(df: pd.DataFrame, expected_type_cols: dict[str, list[str]]) -> None:
    """
    Checks whether columns match their expected type categories:
    - string: should be object/string dtype
    - numeric: should be numeric dtype
    - binary: should contain only 0/1 (ignoring NaN)
    """
    print("Checking expected column types...\n")

    print("Checking string column types...\n")
    for col in expected_type_cols.get("string", []):
        if col not in df.columns:
            print(f"{col}: column not found")
        elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            print(f"{col}: string")
        else:
            print(f"{col}: NOT string, found dtype = {df[col].dtype}")

    print("Checking numeric column types...\n")
    for col in expected_type_cols.get("numeric", []):
        if col not in df.columns:
            print(f"{col}: column not found")
        elif pd.api.types.is_numeric_dtype(df[col]):
            print(f"{col}: numeric")
        else:
            print(f"{col}: NOT numeric, found dtype = {df[col].dtype}")

    print("Checking numeric column types...\n")
    for col in expected_type_cols.get("binary", []):
        if col not in df.columns:
            print(f"{col}: column not found")
        else:
            non_missing_values = set(df[col].dropna().unique())
            if non_missing_values.issubset({0, 1}):
                print(f"{col}: binary")
            else:
                print(f"{col}: NOT binary, found values = {sorted(non_missing_values)}")

def check_implausible_values(df: pd.DataFrame, expected_type_cols: dict, val_dir: str = "validation/") -> None:
    """
    Screens for implausible values.

    - For numeric columns: creates boxplots and saves them to validation/image/.
    - For binary columns: computes count distributions including missing values
      and saves them to validation/table/binary_distribution.csv.
    """

    image_dir = val_dir + "image/"
    table_dir = val_dir + "table/"

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    numeric_cols = expected_type_cols.get("numeric", [])
    binary_cols = expected_type_cols.get("binary", [])

    print("Creating boxplots for numeric columns...\n")
    for col in numeric_cols:
        if col not in df.columns:
            print(f"{col}: column not found")
            continue
        plt.figure(figsize=(4, 4))
        plt.boxplot(df[col].dropna())
        plt.title(f"Boxplot of {col}")
        plt.ylabel(col)
        plt.tight_layout()

        file_path = image_dir + f"boxplot_{col}.png"
        plt.savefig(file_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"{col}: saved boxplot to {file_path}")

    print("\nChecking distributions for binary columns...\n")
    existing_binary_cols = [col for col in binary_cols if col in df.columns]
    missing_binary_cols = [col for col in binary_cols if col not in df.columns]

    for col in missing_binary_cols:
        print(f"{col}: column not found")

    if existing_binary_cols:
        binary_dist = (
            df[existing_binary_cols]
            .apply(lambda x: x.value_counts(dropna=False))
            .T
            .fillna(0)
            .astype(int)
        )

        output_path = table_dir + "binary_distribution.csv"
        binary_dist.to_csv(output_path)
        print(f"\nSaved binary variable distribution to {output_path}")

def check_missingness(df: pd.DataFrame, val_dir: str = "validation/") -> None:
    """
    Checks missingness for all columns and saves the summary table to
    validation/table/missingness_summary.csv.
    """
    table_dir = val_dir + "table/"
    os.makedirs(table_dir, exist_ok=True)

    missing_summary = pd.DataFrame({
        "n_missing": df.isna().sum(),
        "pct_missing": df.isna().mean() * 100
    })

    output_path = table_dir + "missingness_summary.csv"
    missing_summary.to_csv(output_path)

    print(f"Saved missingness summary to {output_path}")

# === main ===
def main():
    # --- define file paths ---
    input_file = "restructured_data.csv"

    input_path = DATA_PATH + input_file
    output_path = "validation/"

    # --- read data ---
    try:
        df = pd.read_csv(input_path)
        print(f"Successfully loaded {input_file}.")
    except FileNotFoundError:
        print(f"{input_file} not found at {DATA_PATH}")
        return

    ## check duplicate encounter keys
    check_duplicate_keys(df, ID_COLS)

    ## verify required columns are present
    check_required_columns(df, REQUIRED_COLS)

    ## check expected data types
    check_expected_column_types(df, EXPECTED_TYPE_COLS)

    ## screen for implausible values
    check_implausible_values(df, EXPECTED_TYPE_COLS)

    ## miss value checks
    check_missingness(df)

if __name__ == "__main__":
    main()


