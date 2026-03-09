# functions to write:
# 1. removing invalid LOG_ID
# 2. ffill for height and weight (partitioning by MRN, sort by date)
# 3. regression imputation

# === Import libraries === 
import pandas as pd
import numpy as np

# for plotting missingness pattern
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns

# for regression imputation
import statsmodels.formula.api as smf

# for function definition and calls
from typing import List, Tuple, Callable

# === Global Constants === 
# Path to data folder
RAW_DATA_PATH = "raw/"

# Relevant variables
POSTOP_COLS = ["LOG_ID", "MRN", "SMRTDTA_ELEM_VALUE"]
INFO_COLS = ["LOG_ID", "MRN", "BIRTH_DATE", "HEIGHT", "WEIGHT", "SEX", "AN_START_DATETIME"]
LABS_COLS = ["LOG_ID", "MRN", "Lab Code", "Lab Name", "Observation Value", "Measurement Units", "Collection Datetime"]

# === Helper Functions for Data Preprocessing ===

def remove_invalid_log_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes df rows with LOG_ID values corresponding to multiple MRN values.
    """
    # Make sure both LOG_ID and MRN exist in df
    required_cols = ["LOG_ID", "MRN"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    # Raise an error if either of LOG_ID or MRN is missing
    if missing_cols:
        raise KeyError(f"Dataframe is missing the required columns: {missing_cols}.")

    # Group by log id and count MRN
    log_mrn_count = df.groupby("LOG_ID")["MRN"].nunique()
    
    # Identify log ids with multiple MRN
    ids_to_remove = log_mrn_count[log_mrn_count > 1].index 
    
    # Remove invalid log ids
    df_cleaned = df[~df["LOG_ID"].isin(ids_to_remove)].reset_index(drop = True)

    # Logging output
    removed_count = len(ids_to_remove)
    print(f"Removed {removed_count} invalid LOG_ID values. {df_cleaned['LOG_ID'].nunique()} unique LOG_IDs remain.")
    
    return df_cleaned


def partition_by_completeness(df: pd.DataFrame,
                              target: str,
                              predictors: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
    """
    Partitions df into a training set (complete) and a test set (missing target).
    Returns a tuple of train set, test set, and test set index.
    """

    # Ensure target and predictors are present in the dataframe
    if not set(predictors).issubset(df.columns) or target not in df.columns:
        raise KeyError(f"{target} or {predictors} not found in dataframe.")

    # Training Set: target and all predictors are not null
    train_mask = df[target].notna() & df[predictors].notna().all(axis = 1)
    train_df = df.loc[train_mask].copy()

    # Test Set: target is null, all predictors are not null
    test_mask = df[target].isna() & df[predictors].notna().all(axis=1)
    test_df = df.loc[test_mask].copy()
    test_index = test_df.index

    # Ensure train and test sets are not empty
    if train_df.empty:
        raise ValueError(f"No complete cases available to train model for {target}.")
    
    if test_df.empty:
        print(f"No missing values found for {target}.")

    return train_df, test_df, test_index
    

def run_lmm_imputation(df: pd.DataFrame,
                       target: str,
                       predictors: List[str],
                       group_col: str) -> pd.DataFrame:
    """
    Applies regression imputation to fill missing values in target using predictors,
    using a linear mixed model with predictors as fixed effect and group_col as random intercept.
    """
    df_filled = df.copy()
    
    try:
        # Partition data into train (complete) and test (missing) sets
        train, test, test_idx = partition_by_completeness(df_filled, target, predictors) 
        
        if test.empty:
            print(f"No missing values found for {target}.")
            return df_filled
    
        # Define regression formula
        formula = f"{target} ~ {' + '.join(predictors)}"
    
        # Fit LMM
        model = smf.mixedlm(formula, train, groups = train[group_col]).fit()
    
        # Predict
        predictions = model.predict(test)
        
        # 5. Inject back into the original dataframe
        df_filled.loc[test_idx, target] = predictions
        print(f"-> Successfully imputed {len(predictions)} values.")
        
    except (ValueError, KeyError) as e:
        print(f"Imputation skipped for {target}: {e}")
    except Exception as e:
        print(f"Unexpected error during {target} imputation: {e}")

    # Imputation summary
    remaining_nulls = df_filled[target].isna().sum()
    print(f"-> Final check: {remaining_nulls} null values remain in '{target}'.")

    return df_filled

    


# === Functions for Data Cleaning ===

def clean_complications(postop: pd.DataFrame, cols_to_keep: List[str] = POSTOP_COLS) -> pd.DataFrame:
    
    """
    Cleans postop (patient_postoperative_copmlications dataframe) by:
    - Selecting and retaining only cols_to_keep
    - Removing invalid LOG_ID values (those correspond to multiple MRNs)

    Returns a df containing cols_to_keep and all encounters with valid LOG_IDs.
    """

    print("Starting cleaning patient postoperative complications data ...")

    # Raise an error if any of cols_to_keep is missing from postop df
    missing_cols = [col for col in cols_to_keep if col not in postop.columns]
    if missing_cols:
        raise KeyError(f"The following required columns are missing: {missing_cols}")

    # Select relevant columns
    # use .copy() to make sure postop is a df not a slice
    postop = postop[cols_to_keep].copy()

    # Drop invalid LOG_IDs
    postop_cleaned = remove_invalid_log_ids(postop)

    print("Cleaning complete.")

    return postop_cleaned
    

def clean_information(info: pd.DataFrame, 
                      cols_to_keep: List[str] = INFO_COLS,
                      date_format: str = "%m/%d/%y %H:%M") -> pd.DataFrame:
    """
    Cleans info (patient_information dataframe) by:
    - Selecting and retaining only cols_to_keep
    - Removing invalid LOG_ID values (those correspond to multiple MRNs)
    - Converting variables to desired type
    - Treating missing values

    INFO_COLS = ["LOG_ID", "MRN", "BIRTH_DATE", "HEIGHT", "WEIGHT", "SEX", "AN_START_DATETIME"]

    date_format: format of AN_START_DATETIME

    Returns a cleaned patient information containing cols_to_keep and all encounters with valid LOG_IDs.
    """

    print("Starting cleaning patient information data ...")

    # --- PRE-PROCESSING ---

    # Raise an error if any of cols_to_keep is missing from info df
    missing_cols = [col for col in cols_to_keep if col not in info.columns]
    if missing_cols:
        raise KeyError(f"The following required columns are missing: {missing_cols}")

    # Select relevant columns
    info = info[cols_to_keep].copy()
    
    # Drop invalid LOG_IDs
    info_cleaned = remove_invalid_log_ids(info)

    # --- CONVERT VARIABLES ---
    
    # Convert AN_START_DATETIME to DateTime
    info_cleaned["AN_START_DATETIME"] = pd.to_datetime(info_cleaned["AN_START_DATETIME"], 
                                                       format = date_format, 
                                                       errors = "coerce")

    # Convert HEIGHT from feet and inches to meters
    height_df = info_cleaned["HEIGHT"].str.split("' ", expand = True) # split string by '
    feet = pd.to_numeric(height_df[0], errors = "coerce")
    inches = pd.to_numeric(height_df[1], errors = "coerce")
    info_cleaned["HEIGHT"] = 0.0254 * (feet * 12 + inches)

    # Convert WEIGHT from ounces to kg
    info_cleaned["WEIGHT"] = 0.0283 * info_cleaned["WEIGHT"]

    # Rename BIRTH_DATE to AGE
    info_cleaned.rename(columns = {"BIRTH_DATE": "AGE"}, inplace = True)

    # --- HANDLE MISSING VALUES ---

    print("Handling missing values in patient_information data...")

    # Check and report number of missing values
    na_count_sex = info_cleaned[info_cleaned["SEX"] == "Unknown"].shape[0]
    print(f"Missing value count in SEX: {na_count_sex}")

    na_count_time = info_cleaned[info_cleaned["AN_START_DATETIME"].isna()].shape[0]
    print(f"Missing value count in AN_START_DATETIME: {na_count_time}")

    na_count_height = info_cleaned[info_cleaned["HEIGHT"].isna()].shape[0]
    print(f"Missing value count in HEIGHT: {na_count_height}")

    na_count_weight = info_cleaned[info_cleaned["WEIGHT"].isna()].shape[0]
    print(f"Missing value count in WEIGHT: {na_count_weight}")

    # Remove "Unknown" value in SEX variable
    info_cleaned = info_cleaned[info_cleaned["SEX"] != "Unknown"].reset_index(drop = True)

    # Remove missing AN_START_DATETIME
    info_cleaned = info_cleaned.dropna(subset = ["AN_START_DATETIME"]).reset_index(drop = True)

    # Impute HEIGHT and WEIGHT
    # Step 1: try forward fill, grouped by MRN and sorted by AN_START_DATETIME
    info_sorted = info_cleaned.sort_values(by = ["MRN", "AN_START_DATETIME"])
    info_sorted['HEIGHT'] = info_sorted.groupby("MRN")["HEIGHT"].ffill()
    info_sorted['WEIGHT'] = info_sorted.groupby("MRN")["WEIGHT"].ffill()
    info_sorted = info_sorted.reset_index(drop = True)

    # Step 2: try regression imputation
    info_filled = run_lmm_imputation(df = info_sorted,
                                    target = "HEIGHT",
                                    predictors = ["SEX", "WEIGHT"],
                                    group_col = "MRN")

    info_filled = run_lmm_imputation(df = info_sorted,
                                    target = "WEIGHT",
                                    predictors = ["SEX", "HEIGHT"],
                                    group_col = "MRN")

    # Step 3: drop remaining NAs (rows still missing both HEIGHT and WEIGHT)
    info_filled = info_filled.dropna(subset = ["HEIGHT", "WEIGHT"], how = "all").reset_index(drop = True)

    print(f"Cleaning complete. Final Row Count: {len(info_filled)}")

    return info_filled
    

    
    
    

# def clean_labs(df: pd.DataFrame, cols_to_retain: List[str] = LABS_COLS) -> pd.DataFrame:



# === Main ===

def main():
    # --- Define file paths ---
    input_files = ["patient_information.csv", "patient_post_op_complications.csv", "patient_labs.csv"]
    raw_files = {} # store .csv files into a dictionary
    output_file = "cleaned_data.csv"

    print("--- Loading data --- ")

    # --- Load raw data files ---
    # try:
    #     for file in input_files:
    #         file_path = RAW_DATA_PATH + file
    #         print(f"Reading {file_path}...")

    #         # read in .csv
    #         df = pd.read_csv(file_path)

    #         # store in a dictionary with the key being the file name
    #         key = file.replace(".csv", "")
    #         raw_files[key] = df
    #         print(f"Successfully loaded {file}")
    # # Raise an error if file not found
    # except FileNotFoundError as e:
    #     print(f"\n[Error] A required file is missing: ")
    #     print(f"{e}")
    #     return
    # # Raise an error if an unexpected error occurred
    # except Exception as e:
    #     print(f"\n[Error] An unexpected error occurred while reading files: ")
    #     print(f"{e}")
    #     return

    

    # --- Extract dataframes ---
    # postop_raw = raw_files["patient_post_op_complications"]
    # info_raw = raw_files["patient_information"]
    # labs_raw = raw_files["patient_labs"]

    info_raw = pd.read_csv("raw/patient_information.csv")
    
    print("\nAll data ready for processing")


    # postop = clean_complications(postop_raw)
    info = clean_information(info_raw)
    


if __name__ == "__main__":
    main()
            
    
    