# === import libraries === 
import pandas as pd

# === global constants ===
## Path to data folder
DATA_PATH = "data/"

## config for creating binary indicator columns
BINARY_COL_CONFIG = {
    "SMRTDTA_ELEM_VALUE": ("hypoxemia", "hypoxemia"),
    "SEX": ("male", "sex")
}

## config for calculating BMI from height and weight
BMI_CONFIG = {
    "height_col": ("HEIGHT", "m"),
    "weight_col": ("WEIGHT", "kg"),
    "new_col": "bmi"
}

# === functions ===
def create_binary_cols(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Creates binary columns based on specified target values and new column names.
    Returns df with the new binary columns added.

    config format:
        {source column: (target value to code as 1, new column name)}
    """
    for source_col, (target_value, new_col) in config.items():
        df[new_col] = (
            df[source_col]
            .astype(str)
            .str.lower()
            .str.contains(str(target_value).lower(), na=False)
            .astype(int)
        )
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

    return df

# === main ===

def main():
    # --- define file paths ---
    input_file = "cleaned_data.csv"
    output_file = "feature_engineered_data.csv"

    input_path = DATA_PATH + input_file
    output_path = DATA_PATH + output_file

    # --- read data ---
    try:
        df = pd.read_csv(input_path)
        print(f"Successfully loaded {input_file}.")
    except FileNotFoundError:
        print(f"{input_file} not found at {DATA_PATH}")
        return

    # --- feature engineering ---
    df = create_binary_cols(df, BINARY_COL_CONFIG)
    print("Binary columns created successfully.")
    
    df = calculate_bmi(df, BMI_CONFIG)
    print("BMI calculated successfully.")

    # --- export to csv ---
    df.to_csv(output_path, index = False)
    print(f"Feature-engineered data saved to {output_file}.")

if __name__ == "__main__":
    main()
    
