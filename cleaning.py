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
from typing import List, Tuple, Callable, Dict, Any, Optional


# === Global Constants === 
# Path to data folder
RAW_DATA_PATH = "raw/"
OUTPUT_DATA_PATH = "data/"

# === Pre-processing Functions === 
def validate_col_existence(df: pd.DataFrame, columns = List[str]):
    """
    Check if columns exist in df. 
    Raise KeyError if any are missing.
    """

    missing_cols = [col for col in columns if col not in df.columns]

    if missing_cols:
        raise KeyError(f"Dataframe is missing the required columns: {missing_cols}")


# def pre_process(df: pd.DataFrame, cols_to_keep: List[str], 
#                 encounter_id, patient_id) -> pd.DataFrame:
#     """
#     Pre-process df by 
#     - Selecting and retaining only cols_to_keep
#     - Removing invalid encounter_id values (those correspond to multiple patient_id)
    
#     Return cleaned df
#     """

#     # Make sure encounter_id, patient_id and all columns in cols_to_keep exist in df
#     cols = [encounter_id, patient_id] + cols_to_keep
#     validate_col_existence(df, cols)

#     # Select relevant columns
#     # use reset_index to make sure df is a dataframe not a slice
#     df = df[cols_to_keep].reset_index(drop = True)

#     # Drop invalid encounter_id
#     # Group by id1 and count patient_id
#     id_pair_count = df.groupby(encounter_id)[patient_id].nunique()
    
#     # Identify encounter_id with multiple patient_id
#     ids_to_remove = id_pair_count[id_pair_count > 1].index 
    
#     # Remove invalid encounter_id values
#     df_cleaned = df[~df[encounter_id].isin(ids_to_remove)].reset_index(drop = True)

#     # Logging output
#     removed_count = len(ids_to_remove)
#     print(f"Removed {removed_count} invalid {encounter_id} values from {df}. {df_cleaned[encounter_id].nunique()} unique {encounter_id} remain.")
    
    
#     return df_cleaned


def pre_process(df: pd.DataFrame, col_mapping: dict) -> pd.DataFrame:
    """
    Pre-process df by 
    - Selecting and retaining only cols_to_keep
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
            if col not in df_cleaned.columns:
                raise KeyError(f"'{col}' not found in dataframe.")
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
            if col in df_cleaned.columns:
                if group:
                    df_cleaned[col] = df_cleaned.groupby(group)[col].ffill()
                else:
                    df_cleaned[col] = df_cleaned[col].ffill()
            else:
                raise KeyError(f"Skipping forward fill for {col}: variable not found in dataset.")

    # LMM
    if "lmm_impute" in config:
        # sanity check: group (random intercept) must be provided for lmm
        if not group:
            raise ValueError("Skipping lmm imputation: group column not provided.")

        for target, predictors in config["lmm_impute"].items():
            # Check target and predictors existence
            if target not in df_cleaned.columns:
                raise KeyError(f"Target column '{target}' not found.")
            
            missing_preds = [p for p in predictors if p not in df_cleaned.columns]
            if missing_preds:
                raise KeyError(f"Predictors {missing_preds} not found for target '{target}'.")
            
            if df_cleaned[target].isna().any():
                df_cleaned = run_lmm_imputation(
                    df=df_cleaned,
                    target=target,
                    predictors=predictors,
                    group_col=group
                )
            else:
                print(f"Skipping lmm imputation for {target}: no missing values found.")

            
            lmm_targets = list(config["lmm_impute"].keys())
            print(f"Removing rows still missing all targets: {lmm_targets}")
            df_cleaned = df_cleaned.dropna(subset=lmm_targets, how="all")
                
    return df_cleaned.reset_index(drop=True)

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

        if not test_matches.empty:
            # Find the most common lab name for the test
            most_common_name = test_matches[name_col].value_counts().idxmax()

            # Find the most common lab code
            name_subset = test_matches[test_matches[name_col] == most_common_name]
            most_common_code = name_subset[code_col].value_counts().idxmax()

            selected_pairs.append({
                name_col: most_common_name,
                code_col: most_common_code
            })
    

    return pd.DataFrame(selected_pairs)


# === Data Cleaning ===

def clean_complications(postop: pd.DataFrame, 
                        col_mapping: dict) -> pd.DataFrame:
    
    """
    Cleans postop (patient_postoperative_complications dataframe) by:
    - Selecting and retaining only the values in col_mapping
    - Removing invalid encounter_id values (those correspond to multiple patient_ids)

    col_mapping example: {"encounter_id": "LOG_ID", 
                          "patient_id": "MRN", 
                          "response": "SMRTDTA_ELEM_VALUE"}

    Returns a df containing the variables in col_mapping and all encounters with valid encounter_id.
    """

    print("Start cleaning patient postoperative complications data...")

    # pre-processing
    postop_cleaned = pre_process(postop, col_mapping)

    print(f"Cleaning complete for post-operative complications data.")
    print(f"Remaining {len(postop_cleaned)} rows.")
    # should be 203939

    return postop_cleaned


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

    missing_config example: {"drop": {"sex": ["UNKNOWN"],
                                      "timestamp": None # Just drop actual NaNs},
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
        info_cleaned = info_cleaned.dropna(subset=[timestamp]).reset_index(drop=True)

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
        
        info_final = handle_missingness(df=info_cleaned, 
                                        group=patient_id, 
                                        timestamp=timestamp, 
                                        config=missing_config)


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

# def clean_information(info: pd.DataFrame,
#                       col_mapping: dict,
#                       date_format: Optional[str] = None,
#                       convert_hw: bool = True,
#                       impute_hw: bool = True) -> pd.DataFrame:
#     """
#     Cleans info (patient_information dataframe) by:
#     - Selecting and retaining only the values in col_mapping
#     - Removing invalid encounter_id values (those correspond to multiple patient_ids)
#     - Treating missing values
    
#     using col_mapping as the source of truth for column names.

#     col_mapping example: {"encounter_id": "LOG_ID", 
#                           "patient_id": "MRN", 
#                           "timestamp": "AN_START_DATETIME",
#                           "height": "HEIGHT",
#                           "weight": "WEIGHT",
#                           "sex": "SEX",
#                           "age": "BIRTH_DATE"}
    
#     If a timestamp variable is provided, then date_format detailing the format of timestamp 
#     should also be provided (ex. "%m/%d/%y %H:%M")

#     convert_hw = True if height (in feet and inches) and weight (in ounces) are provided 
#     and need to be converted to meters and kg, respectively.

#     impute_hw = True if the missing values in height and weight need to be imputed.
#     """

#     print("Start cleaning patient information data...")
    
#     # extract cols that will be used later 
#     # (returns None if col_mapping does not have the key
    
#     patient_id = col_mapping.get("patient_id")
#     timestamp = col_mapping.get("timestamp")
#     height = col_mapping.get("height")
#     weight = col_mapping.get("weight")
#     sex = col_mapping.get("sex")

#     # pre-processing
#     info_cleaned = pre_process(info, col_mapping)

#     # convert timestamp to datetime 
#     # and drop encounters with missing timestamp
#     if timestamp:
#         info_cleaned[timestamp] = pd.to_datetime(
#             info_cleaned[timestamp], format=date_format, errors="coerce"
#         )
#         info_cleaned = info_cleaned.dropna(subset=[timestamp]).reset_index(drop=True)

#     # convert units
#     if convert_hw:
#         # convert height if provided
#         if height:
#             height_split = info_cleaned[height].astype(str).str.split("' ", expand=True)
#             if height_split.shape[1] == 2:
#                 feet = pd.to_numeric(height_split[0], errors="coerce")
#                 inches = pd.to_numeric(height_split[1], errors="coerce")
#                 info_cleaned[height] = 0.0254 * (feet * 12 + inches)
        
#         # convert weight if provided
#         if weight:
#             info_cleaned[weight] = 0.0283 * info_cleaned[weight]

#     # drop missing values (NaN and "Unknown") in sex
#     if sex:
#         info_cleaned = info_cleaned[
#             info_cleaned[sex].notna() & 
#             (info_cleaned[sex].astype(str).str.upper() != "UNKNOWN")
#         ].reset_index(drop=True)

#     # Impute height and weight
#     if impute_hw and height and weight:
#         print("Imputing height and weight using forward fill ...")
        
#         # forward fill, grouped by patient_id and sorted by timestamp
#         info_sorted = info_cleaned.sort_values(by = [patient_id, timestamp])
#         for col in [height, weight]:
#             if col:
#                 info_sorted[col] = info_sorted.groupby(patient_id)[col].ffill()

#         # check if any NAs remaining
#         has_h_na = info_sorted[height].isna().any()
#         has_w_na = info_sorted[weight].isna().any()

#         # setup prdictors for LMM imputation
#         # set sex as a predictor only if provided in the mapping
#         preds = [sex] if sex else []

#         if has_h_na:
#             print("Running LMM imputation for remaining missing height...")
#             info_sorted = run_lmm_imputation(
#                     df=info_sorted, 
#                     target=height, 
#                     predictors=preds + [weight], 
#                     group_col=patient_id
#                 )
#         if has_w_na:
#             print("Running LMM imputation for remaining missing weight...")
#             info_sorted = run_lmm_imputation(
#                     df=info_sorted, 
#                     target=weight, 
#                     predictors=preds + [height], 
#                     group_col=patient_id
#                 )
        
#         # drop rows still missing both height and weight
#         print("Removing rows still missing both height and weight after imputation...")
#         info_filled = info_sorted.dropna(subset=[height, weight], how="all")

#         # print imputation summary
#         h_na = info_filled[height].isna().sum()
#         w_na = info_filled[weight].isna().sum()
#         print(f"Remaining NAs in {height}: {h_na}")
#         print(f"Remaining NAs in {weight}: {w_na}")

#     else:
#         status = "disabled" if not impute_hw else "skipped (missing H/W)"
#         info_filled = info_cleaned
#         print(f"Height / Weight imputation {status}.")

#     print(f"Cleaning complete for patient information data.")
#     print(f"Remaining {len(info_filled)} rows.")
#     # Note: should be 57026

#     return info_filled


def clean_labs(labs: pd.DataFrame, 
               col_mapping: dict, 
               predefined_tests: List[str],
               use_common_tests: bool = True, 
               date_format: Optional[str] = None) -> pd.DataFrame:
    """
    Cleans labs (patient_labs dataframe) by:
    - Selecting and retaining only the values in col_mapping
    - Removing invalid encounter_id values (those correspond to multiple patient_ids)
    - Treating missing values
    - Filtering desired lab tests
    using col_mapping as the source of truth for column names.

    
    Example:
    col_mapping = {"encounter_id": "LOG_ID",
                   "patient_id": "MRN",
                   "code": "Lab Code",
                   "name": "Lab Name",
                   "value": "Observation Value",
                   "unit": "Measurement Units",
                   "timestamp": "Collection Datetime"}

    predefined_tests = ['Leukocytes', 'pH', 'Hematocrit', 'C reactive protein', 'Lactate']

    if use_common_tests = True, then the 5 most common tests in labs df will also be included,
    in addition to the tests in predefined_tests.
    
    If a timestamp variable is provided, then date_format detailing the format of timestamp 
    should also be provided (ex. "%Y-%m-%d %H:%M:%S").
    """

    print("Start cleaning patient labs data...")

    # extract cols that will be used later 
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
        labs_cleaned = labs_cleaned.dropna(subset=[timestamp]).reset_index(drop=True)


    # drop encoutners with missing lab value
    if value:
        labs_cleaned = labs_cleaned[
            labs_cleaned[value].notna() & 
            (labs_cleaned[value]!= 9999999.0)
        ].reset_index(drop=True)

    

    return labs_filtered




    
# === For testing ===

def cleaning():
    input_files = "patient_information.csv"
    info_raw = pd.read_csv(RAW_DATA_PATH + input_files)

    info_raw.rename(columns = {"BIRTH_DATE": "AGE"}, inplace = True)
    info_mapping = {"encounter_id": "LOG_ID",
                   "patient_id": "MRN",
                   "timestamp": "AN_START_DATETIME",
                   "height": "HEIGHT",
                   "weight": "WEIGHT",
                   "sex": "SEX",
                   "age": "AGE"}

    info_missing_config = {"drop": {"SEX": ["Unknown"],
                                    "AN_START_DATETIME": None},
                           "ffill": ["HEIGHT", "WEIGHT"],
                           "lmm_impute": {"HEIGHT": ["SEX", "WEIGHT"],
                                          "WEIGHT": ["SEX", "HEIGHT"]}
                            }

    info = clean_information(info = info_raw, 
                             col_mapping = info_mapping, 
                             date_format = "%m/%d/%y %H:%M",
                             convert_hw = True,
                             missing_config = info_missing_config)

    ##### TEST ERROR HANDLING


if __name__ == "__main__":
    cleaning()
    
    
    # === Note: in main.py, need to define the following ===
    
    # For pre_process and the cleaning functions: 
    # define the subset of columns to keep in each dataset
        # postop_mapping = {encounter_id: "LOG_ID",
        #                   patient_id: "MRN", 
        #                   y: "SMRTDTA_ELEM_VALUE"}
    
        # info_mapping = {encounter_id: "LOG_ID", 
        #                 patient_id: "MRN", 
        #                 age: "BIRTH_DATE", 
        #                 height: "HEIGHT", 
        #                 weight: "WEIGHT", 
        #                 sex: "SEX", 
        #                 timestamp: "AN_START_DATETIME"}
    
        # labs_mapping = {"encounter_id": "LOG_ID", 
        #                 "patient_id": "MRN", 
        #                 "code": "Lab Code",
        #                 "name": "Lab Name",
        #                 "value": "Observation Value",
        #                 "unit": "Measurement Units",
        #                 "timestamp": "Collection Datetime"}

    
    # For find_lab_name_code_pair: 
        # Define the list of lab tests to be included:
        #     ['Leukocytes', 'pH', 'Hematocrit', 'C reactive protein', 'Lactate'] + top 5
    
        # Define the dictionary for special search cases: 
        # special_search = {
        #     "pH": {"pat": r"\bpH\b", "case": True, "regex": True},
        #     "Lactate": {"exclude": "D-Lactate"}
        # }
    
    # For clean_information: 
        # rename BIRTH_DATE to AGE outside of the function
    