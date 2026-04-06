import pandas as pd

# === functions ===
def pivot_wider(df: pd.DataFrame, 
                id_cols: list[str] | str, 
                names_from: str, 
                values_from: str
               ) -> pd.DataFrame:
    """
    Reshapes a dataframe from long format to wide format using specified ID columns,
    column names, and values.
    
    Returns a dataframe in wide format with one row per unique ID combination.
    Arguments:
        df: input dataframe in long format
        id_cols: column name or list of column names that uniquely identify each row
        names_from: column whose unique values will become new column names
        values_from: column whose values will fill the new wide-format columns

    Returns:
        A dataframe reshaped to wide format.
    """
    if isinstance(id_cols, str):
        id_cols = [id_cols]
    
    df2 = df.groupby(id_cols + [names_from], as_index=False)[values_from].first()    
    value_wide = df2.pivot(
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
    print("Data reshaped to wide format successfully.")
    return df_wide

# === main ===
def main():
    # --- define file paths ---
    ## Path to data folder
    DATA_PATH = "data/"
    
    input_file = "feature_engineered_data.csv"
    output_file = "restructured_data.csv"

    input_path = DATA_PATH + input_file
    output_path = DATA_PATH + output_file

    # --- read data ---
    df = setup.load_data(input_path)

    # rename selected columns
    ## columns to rename
    COL_NAME_MAP = {
        "AGE": "age",
        "HEIGHT": "height",
        "WEIGHT": "weight"
    }
    df = df.rename(columns=COL_NAME_MAP)
    print("Selected columns renamed successfully.")
    
    # --- data restructuring ---
    # drop long-format columns before pivoting
    ## columns to drop before reshaping data
    DROP_COLS = [
        "Lab Code",
        "Measurement Units",
        "Collection Datetime"
    ]
    df = df.drop(columns=DROP_COLS, errors="ignore")

    # rename lab tests
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
    df["Lab Name"] = df["Lab Name"].map(LAB_NAME_MAP)
    print("Lab tests renamed successfully.")

    # reshape the data wider
    ID_COLS = ["LOG_ID", "MRN"]
    df_wide = pivot_wider(df, ID_COLS, "Lab Name", "Observation Value")

    # get the names of the new wide-format lab columns
    wide_cols = df["Lab Name"].dropna().unique().tolist()

    # keep only required columns that exist in the wide dataframe
    ## columns to keep in the output dataset
    OUTCOME_COLS = ["hypoxemia"]
    DEMO_COLS = ["age", "sex", "bmi", "height", "weight"]

    keep_cols = ID_COLS + OUTCOME_COLS + wide_cols + DEMO_COLS
    keep_cols = [col for col in keep_cols if col in df_wide.columns]
    print("Final columns selected successfully.")

    df_restructured = df_wide[keep_cols]

    # --- export to csv ---
    df_restructured.to_csv(output_path, index = False)
    print(f"Restructured data saved to {output_file}.")

if __name__ == "__main__":
    main()

    

    