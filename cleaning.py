# functions to write:
# 1. removing invalid LOG_ID
# 2. ffill for height and weight (partitioning by MRN, sort by date)
# 3. regression imputation

# === Import libraries === 
import pandas as pd
import numpy as np

# for plotting missingness pattern
# import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns

# for regression imputation
import statsmodels.formula.api as smf

# for function definition and calls
from typing import List, Tuple, Callable

# === Global Constants === 
# Path to data folder
RAW_DATA_PATH = "raw/"
OUTPUT_DATA_PATH = "data/"

# Relevant variables
POSTOP_COLS = ["LOG_ID", "MRN", "SMRTDTA_ELEM_VALUE"]
INFO_COLS = ["LOG_ID", "MRN", "BIRTH_DATE", "HEIGHT", "WEIGHT", "SEX", "AN_START_DATETIME"]
LABS_COLS = ["LOG_ID", "MRN", "Lab Code", "Lab Name", "Observation Value", "Measurement Units", "Collection Datetime"]

# Lab tests
LAB_TESTS = ['Leukocytes', 'pH', 'Hematocrit', 'C reactive protein', 'Lactate']

# === Helper Functions ===

# For preprocessing

def remove_invalid_ids(df: pd.DataFrame, 
                           id1: str = "LOG_ID",
                           id2: str = "MRN") -> pd.DataFrame:
    """
    Removes df rows with id1 values corresponding to multiple id2 values.
    By default id1 = LOG_ID, id2 = MRN.
    """
    # Make sure both id1 and id2 exist in df
    required_cols = [id1, id2]
    missing_cols = [col for col in required_cols if col not in df.columns]

    # Raise an error if either of id1 or id2 is missing
    if missing_cols:
        raise KeyError(f"Dataframe is missing the required columns: {missing_cols}.")

    # Group by id1 and count id2
    id_pair_count = df.groupby(id1)[id2].nunique()
    
    # Identify id1 with multiple id2
    ids_to_remove = id_pair_count[id_pair_count > 1].index 
    
    # Remove invalid log ids
    df_cleaned = df[~df[id1].isin(ids_to_remove)].reset_index(drop = True)

    # Logging output
    removed_count = len(ids_to_remove)
    print(f"Removed {removed_count} invalid {id1} values. {df_cleaned[id1].nunique()} unique {id1} remain.")
    
    return df_cleaned

def pre_process(df: pd.DataFrame, cols_to_keep: List[str]) -> pd.DataFrame:
    """
    Pre-process df by 
    - Selecting and retaining only cols_to_keep
    - Removing invalid LOG_ID values (those correspond to multiple MRNs)

    Return cleaned df
    """
    # Raise an error if any of cols_to_keep is missing from df
    missing_cols = [col for col in cols_to_keep if col not in df.columns]
    if missing_cols:
        raise KeyError(f"The following required columns are missing: {missing_cols}")

    # Select relevant columns
    # use .copy() to make sure df is a dataframe not a slice
    df = df[cols_to_keep].copy()

    # Drop invalid LOG_IDs
    df_cleaned = remove_invalid_ids(df)
    
    return df_cleaned

# For missing value imputation

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
        
        # Inject back into the original dataframe
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

# For filtering lab tests

def find_lab_name_code_pair(df: pd.DataFrame,
                            most_common_tests: List[str],
                            pre_defined_tests: List[str] = LAB_TESTS) -> pd.DataFrame:
    """
    Find the most common lab name-code pair for each test in most_common_tests and pre_defined_tests

    LAB_TESTS = ['Leukocytes', 'pH', 'Hematocrit', 'C reactive protein', 'Lactate']
    """
    target_tests = pre_defined_tests + most_common_tests
    selected_pairs = []

    for test in target_tests:
        # Case-sensitive filtering for pH
        if test == "pH":
            test_matches = df[df["Lab Name"].str.contains(r'\bpH\b', case = True, na = False)]
        
        # Filtering for Lactate (exclude D-lactate)
        elif test == "Lactate":
            test_matches = df[
                df["Lab Name"].str.contains("Lactate", case = False, na = False) & 
                ~df["Lab Name"].str.contains("D-lactate", case = False, na = False)
            ]
        
        # Case-insensitive filtering for all other tests
        else:
            test_matches = df[df["Lab Name"].str.contains(test, case = False, na = False)]
    
        if not test_matches.empty:
            # Choose the most common Lab Name value
            most_common_name = test_matches["Lab Name"].value_counts().idxmax()
            
            # Find the most common Lab Code associated with that specific Lab Name
            name_subset = test_matches[test_matches["Lab Name"] == most_common_name]
            most_common_code = name_subset["Lab Code"].value_counts().idxmax()
            
            selected_pairs.append({"Lab Name": most_common_name, "Lab Code": most_common_code})

    # convert to a df
    tests_df = pd.DataFrame(selected_pairs)
    
    return tests_df

# === Functions for Data Cleaning ===

def clean_complications(postop: pd.DataFrame, cols_to_keep: List[str] = POSTOP_COLS) -> pd.DataFrame:
    
    """
    Cleans postop (patient_postoperative_complications dataframe) by:
    - Selecting and retaining only cols_to_keep
    - Removing invalid LOG_ID values (those correspond to multiple MRNs)

    Returns a df containing cols_to_keep and all encounters with valid LOG_IDs.
    """

    print("Starting cleaning patient postoperative complications data ...")

    postop_cleaned = pre_process(postop, cols_to_keep)

    print(f"Cleaning complete. Final Row Count: {len(postop_cleaned)}")
    # should be 203939

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

    info_cleaned = pre_process(info, cols_to_keep)

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
    print("Drop remaining NAs in HEIGHT and WEIGHT")
    info_filled = info_filled.dropna(subset = ["HEIGHT", "WEIGHT"], how = "all").reset_index(drop = True)

    print(f"Cleaning complete. Final Row Count: {len(info_filled)}")
    # Note: should be 57026

    return info_filled
    

def clean_labs(labs: pd.DataFrame, 
               cols_to_keep: List[str] = LABS_COLS,
               date_format: str = "%Y-%m-%d %H:%M:%S") -> pd.DataFrame:
    """
    Cleans info (patient_information dataframe) by:
    - Selecting and retaining only cols_to_keep
    - Removing invalid LOG_ID values (those correspond to multiple MRNs)
    - Converting variables to desired type
    - Treating missing values
    - Filtering for desired lab tests

    LABS_COLS = ["LOG_ID", "MRN", "Lab Code", "Lab Name", "Observation Value", "Measurement Units", "Collection Datetime"]
    LAB_TESTS = ['Leukocytes', 'pH', 'Hematocrit', 'C reactive protein', 'Lactate']

    date_format: format of Collection Datetime

    Returns a cleaned lab test data containing cols_to_keep and all encounters with valid LOG_IDs.
    """

    print("Starting cleaning patient labs data ...")
    
    # --- PRE-PROCESSING ---

    labs_cleaned = pre_process(labs, cols_to_keep)

    # --- CONVERT VARIABLES ---
    # Convert Collection Datetime to DateTime
    labs_cleaned["Collection Datetime"] = pd.to_datetime(labs_cleaned["Collection Datetime"],
                                                         format = date_format,
                                                         errors = "coerce")

    # --- HANDLE MISSING VALUES ---
    # Drop rows with Observation Value = 9999999.0
    labs_cleaned = labs_cleaned[labs_cleaned["Observation Value"] != 9999999.0]

    # --- FILTER LAB TESTS ---
    # Find the 5 most common lab tests excluding those in LAB_TESTS
    mask = labs_cleaned["Lab Name"].str.contains('|'.join(LAB_TESTS), case = False, na = False)
    other_tests_df = labs_cleaned[~mask]
    top5_tests = other_tests_df["Lab Name"].value_counts().head(5).index.tolist()
    print(f"Most common tests {top5_tests}")

    # Choose most common lab name-code pair for each test
    tests_df = find_lab_name_code_pair(df = labs_cleaned, most_common_tests = top5_tests)
    print(f"Selected pairs: {tests_df}")

    # Filter to include only rows matching the selected name-code pair
    keys = ["Lab Name", "Lab Code"]
    i1 = labs_cleaned.set_index(keys).index
    i2 = tests_df.set_index(keys).index
    
    # filter to include only the 10 selected name-code pairs
    labs_filtered = labs_cleaned[i1.isin(i2)].reset_index(drop=True)

    print(f"Cleaning complete. Final Row Count: {len(labs_filtered)}")
    # should be 5052813

    return labs_filtered



# === Main ===

def main():
    # --- Define file paths ---
    input_files = ["patient_information.csv", "patient_post_op_complications.csv", "patient_labs.csv"]
    raw_files = {} # store .csv files into a dictionary
    output_file = "cleaned_data.csv"

    print("--- Loading data --- ")

    # --- Load raw data files ---
    try:
        for file in input_files:
            file_path = RAW_DATA_PATH + file
            print(f"Reading {file_path}...")

            # read in .csv
            df = pd.read_csv(file_path)

            # store in a dictionary with the key being the file name
            key = file.replace(".csv", "")
            raw_files[key] = df
            print(f"Successfully loaded {file}")
            
    # Raise an error if file not found
    except FileNotFoundError as e:
        print(f"\n[Error] A required file is missing: ")
        print(f"{e}")
        return
        
    # Raise an error if an unexpected error occurred
    except Exception as e:
        print(f"\n[Error] An unexpected error occurred while reading files: ")
        print(f"{e}")
        return

    # --- Extract dataframes ---
    postop_raw = raw_files["patient_post_op_complications"]
    info_raw = raw_files["patient_information"]
    labs_raw = raw_files["patient_labs"]
    
    print("\nAll data ready for processing")

    # --- Clean data ---
    postop = clean_complications(postop_raw)
    info = clean_information(info_raw)
    labs = clean_labs(labs_raw)

    # --- Merge data ---
    print(" --- Merging dataframes --- ")
    merged_df = postop.merge(info, on = ["LOG_ID", "MRN"], how = "inner")
    merged_df = merged_df.merge(labs, on = ["LOG_ID", "MRN"], how = "inner")
    nlog = merged_df["LOG_ID"].nunique()
    print(f"Merging complete. \nRow Count: {len(merged_df)}. \nUnique LOG ID: {nlog}")

    # -- Final filtering of lab tests ---
    print(" --- Filtering lab tests taken before anesthesia --- ")
    final_df = merged_df[merged_df['Collection Datetime'] <= merged_df['AN_START_DATETIME']].reset_index(drop = True)

    #group by LOG_ID, MRN and Lab Name
    # find indices corresponding to the latest Collection Datetime for each group
    latest_indices = final_df.groupby(['LOG_ID', 'MRN', 'Lab Name'])['Collection Datetime'].idxmax()
    final_df_filtered = final_df.loc[latest_indices].reset_index(drop = True)

    print(f"Filtering complete. Final Row Count: {len(final_df_filtered)}")
    # should be 173721

    # Export to csv
    print("--- Export data ---")
    output_path = OUTPUT_DATA_PATH + output_file
    final_df_filtered.to_csv(output_path, index = False)
    print(f"File saved at {output_path}")


if __name__ == "__main__":
    main()
            
    
    