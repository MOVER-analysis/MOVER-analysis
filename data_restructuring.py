# === import libraries === 
import pandas as pd

# === global constants ===
## Path to data folder
DATA_PATH = "data/"

## lab name mapping
LAB_NAME_MAP = {
    "C reactive protein": "crp",
    "Carbon dioxide": "co2",
    "Glucose": "glucose",
    "Hematocrit": "hematocrit",
    "Hemoglobin": "hemoglobin",
    "Leukocytes^^corrected for nucleated erythrocytes": "leukocytes",
    "Potassium": "potassium",
    "Sodium": "sodium",
    "pH": "pH",
    "Lactate": "lactate"
}

## columns to rename
COL_NAME_MAP = {
    "AGE": "age",
    "HEIGHT": "height",
    "WEIGHT": "weight"
}

## columns to drop before reshaping data
DROP_COLS = ["Lab Code",
             "Measurement Units",
             "Collection Datetime"]

## columns to keep in the output dataset
ID_COLS = ["LOG_ID", "MRN"]
OUTCOME_COLS = ["hypoxemia"]
DEMO_COLS = ["age", "sex", "bmi", "height", "weight"]

# === functions ===
def pivot_wider(df: pd.DataFrame, 
                id_cols: list[str] | str, 
                names_from: str, 
                values_from: str
               ) -> pd.DataFrame:
    if isinstance(id_cols, str):
        id_cols = [id_cols]
        
    value_wide = df.pivot(
        index=id_cols,
        columns=names_from,
        values=values_from
    ).reset_index()

    df_merged = df.merge(value_wide, on=id_cols, how="left")

    df_wide = (
        df_merged
            .drop(columns=[
                names_from,
                values_from
            ])
            .drop_duplicates()
    )
    return df_wide

# === main ===
def main():
    # --- define file paths ---
    input_file = "feature_engineered_data.csv"
    output_file = "restructured_data.csv"

    input_path = DATA_PATH + input_file
    output_path = DATA_PATH + output_file

    # --- read data ---
    try:
        df_input = pd.read_csv(input_path)
        print(f"Successfully loaded {input_file}.")
    except FileNotFoundError:
        print(f"{input_file} not found at {DATA_PATH}")
        return

    # work on a copy of the input data
    df = df_input.copy()

    # rename selected columns
    df = df.rename(columns=COL_NAME_MAP)
    print("Selected columns renamed successfully.")
    
    # --- data restructuring ---
    # drop long-format columns before pivoting
    df = df.drop(columns=DROP_COLS, errors="ignore")

    # rename lab tests
    df["Lab Name"] = df["Lab Name"].map(LAB_NAME_MAP)
    print("Lab tests renamed successfully.")

    # reshape the data wider
    df_wide = pivot_wider(df, ID_COLS, "Lab Name", "Observation Value")
    print("Data reshaped to wide format successfully.")

    # get the names of the new wide-format lab columns
    wide_cols = df["Lab Name"].dropna().unique().tolist()

    # keep only required columns that exist in the wide dataframe
    keep_cols = ID_COLS + OUTCOME_COLS + wide_cols + DEMO_COLS
    keep_cols = [col for col in keep_cols if col in df_wide.columns]
    print("Final columns selected successfully.")

    df_restructured = df_wide[keep_cols]

    # --- export to csv ---
    df_restructured.to_csv(output_path, index = False)
    print(f"Restructured data saved to {output_file}.")

if __name__ == "__main__":
    main()

    

    