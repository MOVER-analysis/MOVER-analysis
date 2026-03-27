# === Import libraries === 
import pandas as pd
import numpy as np
import re

# for regression imputation
import statsmodels.formula.api as smf

# for function definition and calls
from typing import List, Tuple, Callable, Dict, Any, Optional


# === Global Constants === 
# Path to data folder
RAW_DATA_PATH = "raw/"
OUTPUT_DATA_PATH = "data/"

# === Pre-processing Functions === 
def validate_col_existence(df: pd.DataFrame, columns: List[str]):
    """
    Check if columns exist in df. 
    Raise KeyError if any are missing.
    """

    missing_cols = [col for col in columns if col not in df.columns]

    if missing_cols:
        raise KeyError(f"Dataframe is missing the required columns: {missing_cols}")



def pre_process(df: pd.DataFrame, col_mapping: dict) -> pd.DataFrame:
    """
    Pre-process df by 
    - Selecting and retaining only cols_to_keep
    - Dropping rows where either encounter_id or patient_id is null
    - Removing invalid encounter_id values (those correspond to multiple patient_id)

    Example col_mapping:
        {
            encounter_id: "LOG_ID",
            patient_id: "MRN",
            ...
        }
    
    Return cleaned df
    """
        
    # extract IDs from mapping
    encounter_id = col_mapping.get("encounter_id")
    patient_id = col_mapping.get("patient_id")

    # make sure the columns we need to keep are in the df
    keep_cols = list(col_mapping.values())
    validate_col_existence(df, keep_cols)

    # include only the selected columns
    df_cleaned = df[keep_cols].reset_index(drop=True)

    # drop rows with missing encounter / patient ids
    df_cleaned = df_cleaned.dropna(subset = [encounter_id, patient_id])

    # drop invalid encounter IDs
    if encounter_id and patient_id:
        # count unique patients per encounter
        id_counts = df_cleaned.groupby(encounter_id)[patient_id].nunique()
        
        # find encounters associated with more than 1 patient
        invalid_enc_ids = id_counts[id_counts > 1].index
        df_cleaned = df_cleaned[~df_cleaned[encounter_id].isin(invalid_enc_ids)].reset_index(drop=True)
        removed_count = len(invalid_enc_ids)
        print(f"Removed {removed_count} invalid {encounter_id}s.")
        print(f"{df_cleaned[encounter_id].nunique()} unique {encounter_id}s remain.")

    else:
        # if encounter id or patient id not provided
        print("Skipping ID integrity check: encounter ID or patient ID missing from mapping.")
    
    return df_cleaned

# === Handling Missing Value === 

def partition_by_completeness(df: pd.DataFrame,
                              target: str,
                              predictors: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
    """
    Partitions df into a training set (complete) and a test set (missing target).
    Returns a tuple of train set, test set, and test set index.
    """

    # Ensure target and predictors are present in the dataframe
    cols = [target] + predictors
    validate_col_existence(df, cols)

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

    return train_df, test_df, test_index

def run_lmm_imputation(df: pd.DataFrame,
                       target: str,
                       predictors: List[str],
                       group_col: str) -> pd.DataFrame:
    """
    Applies regression imputation to fill missing values in target using predictors,
    using a linear mixed model with predictors as fixed effect and group_col as random intercept.
    """

    # Ensure target, predictors and group_col are present in the dataframe
    cols = [target] + predictors + [group_col]
    validate_col_existence(df, cols)
    
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
        print(f"Successfully imputed {len(predictions)} values.")
        
    except (ValueError, KeyError) as e:
        print(f"Imputation skipped for {target}: {e}")
    except Exception as e:
        print(f"Unexpected error during {target} imputation: {e}")

    # Imputation summary
    # remaining_nulls = df_filled[target].isna().sum()
    # print(f"{remaining_nulls} null values remain in '{target}'.")

    return df_filled


def handle_missingness(df: pd.DataFrame, 
                       config: Dict[str, Any],
                       timestamp: Optional[str] = None,
                       group: Optional[str] = None) -> pd.DataFrame:
    """
    Drop or impute missing values in df based on config.

    Config is a dictionary detailing how the missing values in specified columns should be treated.
    
    Example:
     config = {
               "drop": {"sex": ["UNKNOWN"],
                        "timestamp": None # Just drop actual NaNs},
               "ffill": ["height", "weight"],
               "lmm_impute": {"height": ["sex", "weight"],
                              "weight": ["sex", "height"]}
              }
    
    """
    df_cleaned = df.copy()

    # Drop missing values in specified cols
    if "drop" in config:
        for col, custom_val in config["drop"].items():
            # replace custom_val (if specified) with np.NaN
            if custom_val:
                df_cleaned[col] = df_cleaned[col].replace(custom_val, np.nan)
            # drop missings
            df_cleaned = df_cleaned.dropna(subset=[col])

    # Forward fill
    if "ffill" in config:
        # sanity check: if timestamp is provided / exist in df
        if not timestamp or timestamp not in df_cleaned.columns:
            raise ValueError(f"Timestamp column '{timestamp}' is missing or invalid for ffill.")

        sort_cols = [group, timestamp] if group else [timestamp]
        df_cleaned = df_cleaned.sort_values(by=sort_cols)

        for col in config["ffill"]:
            if group:
                df_cleaned[col] = df_cleaned.groupby(group)[col].ffill()
            else:
                df_cleaned[col] = df_cleaned[col].ffill()

    # LMM
    if "lmm_impute" in config:
        # sanity check: group (random intercept) must be provided for lmm
        if not group:
            raise ValueError("Skipping lmm imputation: group column not provided.")

        for target, predictors in config["lmm_impute"].items():
            df_cleaned = run_lmm_imputation(
                df=df_cleaned,
                target=target,
                predictors=predictors,
                group_col=group
            )
            
        lmm_targets = list(config["lmm_impute"].keys())
        print(f"Removing rows still missing all targets: {lmm_targets}")
        df_cleaned = df_cleaned.dropna(subset=lmm_targets, how="all")
                
    return df_cleaned.reset_index(drop=True)

def translate_missing_config(missing_config: Dict[str, Any],
                             col_mapping: Dict[str, str]) -> Dict[str, Any]:

    """
    Translates logical keys in missing_config to actual column names 
    in the dataset using col_mapping.
    """
    translated = {}

    # Handle "drop": dictionary mapping {column: [values_to_drop]}
    if "drop" in missing_config:
        translated["drop"] = {
            col_mapping.get(key, key): value for key, value in missing_config["drop"].items()
        }

    # Handle "ffill": list of column keys
    if "ffill" in missing_config:
        translated["ffill"] = [
            col_mapping.get(key, key) for key in missing_config["ffill"]
        ]

    # Handle "lmm_impute": dictionary {target: [predictors]}
    if "lmm_impute" in missing_config:
        translated["lmm_impute"] = {
            col_mapping.get(target, target): [col_mapping.get(pred, pred) for pred in preds]
            for target, preds in missing_config["lmm_impute"].items()
        }

    return translated


# === Data Cleaning ===

def clean_complications(postop: pd.DataFrame, 
                        col_mapping: dict,
                        missing_config: Dict[str, Any] | None = None) -> pd.DataFrame:
    
    """
    Cleans postop (patient_postoperative_complications dataframe) by:
    - Selecting and retaining only the values in col_mapping
    - Removing invalid encounter_id values (those correspond to multiple patient_ids)
    - Treating missing values
    
    using col_mapping as the source of truth for column names.

    col_mapping example: {"encounter_id": "LOG_ID", 
                          "patient_id": "MRN", 
                          "response": "SMRTDTA_ELEM_VALUE"}

    Handles missing values as specified in missing_config.

    missing_config example: {"drop": {"sex": ["Unknown"]}}
    
    Returns a df containing the variables in col_mapping and all encounters with valid encounter_id.
    """

    print("Start cleaning patient postoperative complications data...")


    # extract mapped column names
    patient_id = col_mapping.get("patient_id")
    response = col_mapping.get("response")
    timestamp = col_mapping.get("timestamp")

    # pre-processing
    postop_cleaned = pre_process(postop, col_mapping)

    # handle missingness
    postop_final = postop_cleaned
    
    if missing_config:
        print("Handling missing values in patient postoperative complications data...")
        
        translated_cfg = translate_missing_config(missing_config, col_mapping)
        postop_final = handle_missingness(df=postop_cleaned,
                                          config=translated_cfg,
                                          group=patient_id,
                                          timestamp=timestamp)


        # print summary
        for col in postop_final.columns:
            col_na = postop_final[col].isna().sum()
            print(f"Remaining NAs in {col}: {col_na}")
    else:
        print("Skipped handling missing values. missing_config not provided.")

    print(f"Cleaning complete for post-operative complications data.")
    print(f"Remaining {len(postop_final)} rows.")
    # should be 203939
    
    return postop_final


def clean_information(info: pd.DataFrame,
                      col_mapping: Dict[str, str], 
                      date_format: str | None = None, 
                      convert_hw: bool = True,
                      missing_config: Dict[str, Any] | None = None) -> pd.DataFrame:

    """
    Cleans info (patient_information dataframe) by:
    - Selecting and retaining only the values in col_mapping
    - Removing invalid encounter_id values (those correspond to multiple patient_ids)
    - Treating missing values
    
    using col_mapping as the source of truth for column names.

    col_mapping example: {"encounter_id": "LOG_ID", 
                          "patient_id": "MRN", 
                          "timestamp": "AN_START_DATETIME",
                          "height": "HEIGHT",
                          "weight": "WEIGHT",
                          "sex": "SEX",
                          "age": "BIRTH_DATE"}
    
    If a timestamp variable is provided, then date_format detailing the format of timestamp 
    should also be provided (ex. "%m/%d/%y %H:%M")

    convert_hw = True if height (in feet and inches) and weight (in ounces) are provided 
    and need to be converted to meters and kg, respectively.

    Handles missing values as specified in missing_config.

    missing_config example: {
                                "drop": {"sex": ["Unknown"],
                                         "timestamp": None},
                                "ffill": ["height", "weight"],
                                "lmm_impute": {"height": ["sex", "weight"],
                                               "weight": ["sex", "height"]}
                            }
    
    """
    
    print("Start cleaning patient information data...")
    
    # extract mapped column names
    patient_id = col_mapping.get("patient_id")
    timestamp = col_mapping.get("timestamp")
    height = col_mapping.get("height")
    weight = col_mapping.get("weight")
    sex = col_mapping.get("sex")

    # pre-processing
    info_cleaned = pre_process(info, col_mapping)

    # convert timestamp to datetime 
    # and drop encounters with missing timestamp
    if timestamp:
        info_cleaned[timestamp] = pd.to_datetime(
            info_cleaned[timestamp], format=date_format, errors="coerce"
        )

    # convert units
    if convert_hw:
        # convert height if provided
        if height:
            height_split = info_cleaned[height].astype(str).str.split("' ", expand=True)
            if height_split.shape[1] == 2:
                feet = pd.to_numeric(height_split[0], errors="coerce")
                inches = pd.to_numeric(height_split[1], errors="coerce")
                info_cleaned[height] = 0.0254 * (feet * 12 + inches)
        
        # convert weight if provided
        if weight:
            info_cleaned[weight] = 0.0283 * info_cleaned[weight]

    # handle missingness
    if missing_config:
        print("Handling missing values in patient information data...")

        # translating missing_config
        translated_cfg = translate_missing_config(missing_config, col_mapping)
        
        info_final = handle_missingness(df=info_cleaned, 
                                        config=translated_cfg,
                                        group=patient_id, 
                                        timestamp=timestamp)


        # print summary
        for col in info_final.columns:
            col_na = info_final[col].isna().sum()
            print(f"Remaining NAs in {col}: {col_na}")
    else:
        print("Skipped handling missing values. missing_config not provided.")

    print(f"Cleaning complete for patient information data.")
    print(f"Remaining {len(info_final)} rows.")
    # Note: should be 57026
    
    return info_final


def clean_labs(labs: pd.DataFrame, 
               col_mapping: dict, 
               date_format: Optional[str] = None,
               missing_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Cleans labs (patient_labs dataframe) by:
    - Selecting and retaining only the values in col_mapping
    - Removing invalid encounter_id values (those correspond to multiple patient_ids)
    - Treating missing values
    using col_mapping as the source of truth for column names.

    
    Example:
    col_mapping = {"encounter_id": "LOG_ID",
                   "patient_id": "MRN",
                   "code": "Lab Code",
                   "name": "Lab Name",
                   "value": "Observation Value",
                   "unit": "Measurement Units",
                   "timestamp": "Collection Datetime"}

    If a timestamp variable is provided, then date_format detailing the format of timestamp 
    should also be provided (ex. "%Y-%m-%d %H:%M:%S").

    Handles missing values as specified in missing_config.

    missing_config example: {"drop": {"Observation Value": [9999999.0],
                                      "Collection Datetime": None}
                            }
    """

    print("Start cleaning patient labs data...")

    # extract cols that will be used later 
    encounter_id = col_mapping.get("encounter_id")
    patient_id = col_mapping.get("patient_id")
    code = col_mapping.get("code")
    name = col_mapping.get("name")
    value = col_mapping.get("value")
    unit = col_mapping.get("unit")
    timestamp = col_mapping.get("timestamp")
    
    # pre-processing
    labs_cleaned = pre_process(labs, col_mapping)

    # convert timestamp to datetime 
    # and drop encounters with missing timestamp
    if timestamp:
        labs_cleaned[timestamp] = pd.to_datetime(
            labs_cleaned[timestamp], format=date_format, errors="coerce"
        )

    # handle missing values
    if missing_config:
        print("Handling missing values in patient labs data...")
        
        translated_cfg = translate_missing_config(missing_config, col_mapping)
        labs_final = handle_missingness(df=labs_cleaned, 
                                        config=translated_cfg,
                                        group=patient_id, 
                                        timestamp=timestamp)


        # print summary
        for col in labs_final.columns:
            col_na = labs_final[col].isna().sum()
            print(f"Remaining NAs in {col}: {col_na}")
    else:
        print("Skipped handling missing values. missing_config not provided.")
        labs_final = labs_cleaned
        
    print(f"Cleaning complete for patient labs data.")
    print(f"Remaining {len(labs_final)} rows.")
    
    return labs_final

# filtering lab test results

def find_lab_name_code_pair(df: pd.DataFrame,
                            test_list: List[str],
                            name_col: str,
                            code_col: str,
                            special_configs: Optional[Dict[str, Dict[str, Any]]] = None
                           ) -> pd.DataFrame:
    """
    Finds name-code pairs for each test in test_list with a default search strategy 
    (case-insensitive, include all relevant results).
    
    If provided, special_configs overrides the default search strategy.

    Supported keys within each config:
    1. pat: Specific string/regex to search for (defaults to the test name).
    2. case: Boolean for case sensitivity (defaults to False).
    3. regex: Boolean to enable regex special characters (defaults to False).
    4. exclude: String to ignore if found in the Lab Name (defaults to None).
    
    Example special_configs:
    {
        "pH": {"pat": r"\bpH\b", "case": True},
        "Lactate": {"pat": "Lactate", "case": False, "exclude": "D-lactate"}
    }

    Returns a dataframe of selected pairs.
    """

    # Make sure name_col and code_col exist in df
    cols = [name_col, code_col]
    validate_col_existence(df, cols)
    
    special_configs = special_configs or {}
    selected_pairs = []

    for test in test_list:
        # Get custom settings or use defaults
        cfg = special_configs.get(test, {})
        search_pat = cfg.get("pat", test)
        is_case = cfg.get("case", False)
        is_regex = cfg.get("regex", False)
        exclusion = cfg.get("exclude", None)

        # Filter df
        mask = df[name_col].str.contains(search_pat, 
                                         case = is_case, 
                                         regex = is_regex, 
                                         na = False)
        if exclusion:
            mask &= ~df[name_col].str.contains(exclusion, case = False, na = False)

        test_matches = df[mask]
        counts = test_matches[name_col].value_counts()

        if not counts.empty:
            # Find the most common lab name for the test
            most_common_name = counts.idxmax()

            # Find the most common lab code
            name_subset = test_matches[test_matches[name_col] == most_common_name]
            most_common_code = name_subset[code_col].value_counts().idxmax()

            selected_pairs.append({
                name_col: most_common_name,
                code_col: most_common_code
            })
    

    return pd.DataFrame(selected_pairs)


def filter_labs(labs: pd.DataFrame,
                col_mapping: Dict[str, str],
                predefined_tests: List[str],
                special_configs: Optional[Dict[str, Dict[str, Any]]] = None,
                use_common_tests: bool = True) -> pd.DataFrame:

    """
    Filters a cleaned lab DataFrame for specific tests using a name-code pairing strategy.
    
    1. Identifies the most common Name-Code pairs for each desired test.
    2. Optionally adds the top 5 most frequent tests not in the predefined list.
    3. Filters the dataframe to include only rows matching those exact pairs.

    Example col_mapping:
    {
        "code": "Lab Code",
        "name": "Lab Name"
    }
    
    Example special_configs:
    {
        "pH": {"pat": r"\bpH\b", "case": True},
        "Lactate": {"pat": "Lactate", "case": False, "exclude": "D-lactate"}
    }
    """
    print("Filtering lab tests results...")
    
    # extract columns
    name = col_mapping.get("name")
    code = col_mapping.get("code")

    if not name or not code:
        raise KeyError("col_mapping must contain lab 'name' and 'code' for filtering.")

    # validate existence
    validate_col_existence(df=labs, columns=[name, code])

    tests_to_filter = predefined_tests.copy()
    
    # top 5 tests excluding those in predefined tests
    if use_common_tests:
        pattern = '|'.join([re.escape(t) for t in predefined_tests])
        is_predefined = labs[name].str.contains(pattern, case=False, na=False)
        other_tests = labs[~is_predefined]
        
        if not other_tests.empty:
            top5_tests = other_tests[name].value_counts().head(5).index.tolist()
            print(f"Adding top 5 common tests: {top5_tests}")
            tests_to_filter = predefined_tests + top5_tests

    # find name-code pairs
    selected_pairs_df = find_lab_name_code_pair(
        df=labs,
        test_list=tests_to_filter,
        special_configs=special_configs,
        name_col=name,
        code_col=code
    )

    if selected_pairs_df.empty:
        print("No matching lab pairs found for the provided tests list.")
        return labs

    print(f"Selected {len(selected_pairs_df)} Lab Name-Code pairs for filtering.")
    print(selected_pairs_df)

    # filter
    keys = [name, code]
    
    # Create the reference index from the selected pairs
    valid_pairs_index = selected_pairs_df.set_index(keys).index
    
    # Filter the main dataframe
    labs_filtered = labs[labs.set_index(keys).index.isin(valid_pairs_index)].copy()

    print(f"Filtering complete for patient labs data.")
    print(f"Remaining {len(labs_filtered)} rows.")
    # should be 5052812

    return labs_filtered.reset_index(drop=True)
    
# === For testing ===
# Delete the following parts after moving everything to main.py

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

    # --- Extract dataframes ---
    postop_raw = raw_files["patient_post_op_complications"]
    info_raw = raw_files["patient_information"]
    info_raw.rename(columns = {"BIRTH_DATE": "AGE"}, inplace = True)
    labs_raw = raw_files["patient_labs"]
    
    print("\nAll data ready for processing")


    # --- Define column mapping configurations ---

    postop_mapping = {"encounter_id": "LOG_ID",
                      "patient_id": "MRN", 
                      "response": "SMRTDTA_ELEM_VALUE"}

    info_mapping = {"encounter_id": "LOG_ID", 
                    "patient_id": "MRN", 
                    "age": "AGE", 
                    "height": "HEIGHT", 
                    "weight": "WEIGHT", 
                    "sex": "SEX", 
                    "timestamp": "AN_START_DATETIME"}

    labs_mapping = {"encounter_id": "LOG_ID", 
                    "patient_id": "MRN", 
                    "code": "Lab Code",
                    "name": "Lab Name",
                    "value": "Observation Value",
                    "unit": "Measurement Units",
                    "timestamp": "Collection Datetime"}

    # --- Define missing value configurations ---

    info_missing_config = {
        "drop": {"sex": ["Unknown"],
                 "timestamp": None},
        "ffill": ["height", "weight"],
        "lmm_impute": {"height": ["sex", "weight"],
                       "weight": ["sex", "height"]}
        }
    
    

    labs_missing_config = {
        "drop": {"value": [9999999.0],
                 "timestamp": None}
        }

    # --- Define constants for filtering lab tests ---

    predefined_tests = ['Leukocytes', 'pH', 'Hematocrit', 'C reactive protein', 'Lactate']
    
    lab_search_config = {"pH": {"pat": r"\bpH\b", "case": True, "regex": True},
                         "Lactate": {"exclude": "D-Lactate"}
                         }

    # --- Clean datasets ---

    postop = clean_complications(postop = postop_raw,
                                 col_mapping = postop_mapping)

    info = clean_information(info = info_raw,
                            col_mapping = info_mapping,
                            date_format = "%m/%d/%y %H:%M",
                            convert_hw = True,
                            missing_config = info_missing_config)

    labs = clean_labs(labs = labs_raw,
                      col_mapping = labs_mapping,
                      date_format = "%Y-%m-%d %H:%M:%S",
                      missing_config = labs_missing_config)

    # --- Filter tests results ---
    labs_filtered = filter_labs(labs = labs,
                                col_mapping = labs_mapping,
                                predefined_tests = predefined_tests,
                                special_configs = lab_search_config)

    # --- Merge logic optimization ---
    print("Pre-filtering labs to latest result before anesthesia...")
    
    # filter labs
    timing_info = info[['LOG_ID', 'MRN', 'AN_START_DATETIME']]
    labs_filtered = labs_filtered.merge(timing_info, on=['LOG_ID', 'MRN'], how='inner')
    
    # Filter for labs before anesthesia
    labs_final = labs_filtered[
        labs_filtered['Collection Datetime'] <= labs_filtered['AN_START_DATETIME']
    ]
    
    # Get latest indices
    latest_indices = labs_final.groupby(['LOG_ID', 'MRN', 'Lab Name'])['Collection Datetime'].idxmax()
    labs_final_subset = labs_final.loc[latest_indices].copy()
    
    # Drop the AN_START_DATETIME from the subset before the final merge 
    # to avoid AN_START_DATETIME_x / _y
    labs_final_subset = labs_final_subset.drop(columns=['AN_START_DATETIME'])
    
    # Final Merge
    print("Performing final merge...")
    merged_patient_info = postop.merge(info, on=["LOG_ID", "MRN"], how="inner")
    
    # Merging with filtered labs (this will expand the DF to one row per LOG_ID + Lab Test)
    final_df_filtered = merged_patient_info.merge(labs_final_subset, on=["LOG_ID", "MRN"], how="inner")
    
    print(f"Filtering complete. Final Row Count: {len(final_df_filtered)}")
    # 695752

    # Export to csv
    print("--- Export data ---")
    output_path = OUTPUT_DATA_PATH + output_file
    final_df_filtered.to_csv(output_path, index = False)
    print(f"File saved at {output_path}")



if __name__ == "__main__":
    main()
    
   