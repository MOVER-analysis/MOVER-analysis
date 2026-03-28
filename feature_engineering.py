# === functions ===
def create_binary_cols(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Creates binary columns based on specified target values, new column names, and matching rules.
    Returns df with the new binary columns added.

    config format:
        {source_column: (target_value, new_column_name, match_type)}

    match_type:
        - "in": code as 1 if target value is contained in the source value
        - "exact": code as 1 if source value exactly matches the target value
    """
    for source_col, (target_value, new_col, match_type) in config.items():
        target_value = str(target_value).lower()

        if match_type == "in":
            df[new_col] = (
                df[source_col]
                .astype(str)
                .str.lower()
                .str.contains(target_value, na=False)
                .astype(int)
            )

        elif match_type == "exact":
            source_series = df[source_col]

            df[new_col] = source_series.apply(
                lambda x: (
                    pd.NA if pd.isna(x)
                    else int(str(x).lower() == target_value)
                )
            )

        else:
            raise ValueError(
                f"Invalid match_type '{match_type}' for column '{source_col}'. "
                "Use 'in' or 'exact'."
            )

    print("Binary columns created successfully.")
    return df

def calculate_bmi(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Calculates BMI using height and weight columns.
    Returns df with the BMI column added.

    config format:
        {
            "height_col": (height column name, height unit),
            "weight_col": (weight column name, weight unit),
            "new_col": output BMI column name
        }
    """
    # Extract column names and units from the config
    height_col, height_unit = config["height_col"]
    weight_col, weight_unit = config["weight_col"]
    new_col = config["new_col"]

    # Convert weight to kilograms if needed
    height_unit = height_unit.lower()
    if height_unit in ["m", "meter", "meters"]:
        height = df[height_col]
    elif height_unit in ["cm", "centimeter", "centimeters"]:
        height = df[height_col] / 100
    else:
        raise ValueError(f"Unsupported height unit: {height_unit}")

    # Convert weight to kilograms if needed
    weight_unit = weight_unit.lower()
    if weight_unit in ["lb", "pounds", "pound"]:
        weight = df[weight_col] * 0.45359237
    elif weight_unit in ["kg", "kilograms", "kilogram"]:
        weight = df[weight_col]
    else:
        raise ValueError(f"Unsupported weight unit: {weight_unit}")

    # Calculate BMI as weight (kg) divided by height squared (m^2)
    df[new_col] = weight / (height ** 2)

    print(f"BMI column created successfully: '{new_col}'.")
    return df

# === main ===

def main():
    # --- define file paths ---
    ## Path to data folder
    DATA_PATH = "data/"
    
    input_file = "cleaned_data.csv"
    output_file = "feature_engineered_data.csv"

    input_path = DATA_PATH + input_file
    output_path = DATA_PATH + output_file

    # --- read data ---
    df = load_data(input_path)

    # --- feature engineering ---
    ## config for creating binary indicator columns
    BINARY_COL_CONFIG = {
        "SMRTDTA_ELEM_VALUE": ("hypoxemia", "hypoxemia", "in"),
        "SEX": ("female", "sex", "exact")
    }
    df = create_binary_cols(df, BINARY_COL_CONFIG)

    ## config for calculating BMI from height and weight
    BMI_CONFIG = {
        "height_col": ("HEIGHT", "m"),
        "weight_col": ("WEIGHT", "kg"),
        "new_col": "bmi"
    }
    df = calculate_bmi(df, BMI_CONFIG)

    # --- export to csv ---
    df.to_csv(output_path, index = False)
    print(f"Feature-engineered data saved to {output_file}.")

if __name__ == "__main__":
    main()
    
