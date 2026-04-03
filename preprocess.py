# === Import libraries === 
import pandas as pd
import numpy as np
import re

# for regression imputation
import statsmodels.formula.api as smf
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# for function definition and calls
from typing import List, Tuple, Callable, Dict, Any, Optional

# === Pre-processing Functions === 
def validate_col_existence(df: pd.DataFrame, columns: List[str]):
    """
    Check if columns exist in df. 
    Raise KeyError if any are missing.
    """

    missing_cols = [col for col in columns if col not in df.columns]

    if missing_cols:
        raise KeyError(f"Dataframe is missing the required columns: {missing_cols}")

def pre_process(df: pd.DataFrame, 
                id_cols: List[str], 
                feature_cols: List[str],
                timestamp_idx: Optional[List[int]] = None,
                timestamp_formats: Optional[List[Optional[str]]] = None) -> pd.DataFrame:
    """
    Pre-process df by 
    - Retaining only id_cols and feature_cols
    - Dropping rows where either encounter_id or patient_id is null
    - Removing invalid encounter_id values (those correspond to multiple patient_id)
    - If timestamp is provided in feature_cols, convert timestamp to datetime and 
      drop rows with missing timestamp

    Arguments:
    - df: the dataframe to be processed
    - id_cols: [encounter_id, patient_id]
    - features_cols: list of features to be included

    Optional Args:
    - timestamp_idx: the list of indic of the timestamp variable(s) within feature_cols
    - timestamp_format: the list of strftime format (e.g., "%Y-%m-%d %H:%M:%S"), 
                        must match the length of timestamp_idx if provided
    """

    # --- Validate there's exactly two id columns --- 
    if len(id_cols)!= 2:
        raise ValueError(f"Expected 2 ID columns, got {len(id_cols)}")

    encounter_id = id_cols[0]
    patient_id = id_cols[1]

    # --- Make sure all columns exist in df ---
    cols_to_keep = id_cols + feature_cols
    validate_col_existence(df, cols_to_keep)
    
    # --- Subset df to include only id_cols + feature_cols ---
    df_subset = df[cols_to_keep].copy().reset_index(drop=True)

    # --- Drop rows with missing or invalid ids --- 
    df_subset = df_subset.dropna(subset = id_cols)

    # Count unique patients per encounter
    id_counts = df_subset.groupby(encounter_id)[patient_id].nunique()

    # Find encounters associated with more than one patient id
    invalid_enc_ids = id_counts[id_counts > 1].index
    df_cleaned = df_subset[~df_subset[encounter_id].isin(invalid_enc_ids)].reset_index(drop=True)
    remaining_row = len(df_cleaned)
    
    removed_count = len(invalid_enc_ids)
    print(f"Removed {removed_count} invalid {encounter_id}s.")
    print(f"{df_cleaned[encounter_id].nunique()} unique {encounter_id}s.")

    # --- (Optional) Convert timestamp variables to datetime ---
    if timestamp_idx is not None:
        formats = timestamp_formats if timestamp_formats else [None] * len(timestamp_idx)
        
        # Sanity check
        if len(timestamp_idx) != len(formats):
            raise ValueError("timestamp_indices and timestamp_formats must have the same length.")

        for ts_idx, ts_fmt in zip(timestamp_idx, formats):
            if ts_idx >= len(feature_cols):
                raise IndexError(f"timestamp_idx {timestamp_idx} is out of range for feature_cols (length {len(feature_cols)})")
                
            # Map the feature_cols index to the df index (shift by 2 for IDs)
            df_time_idx = ts_idx + len(id_cols)
            timestamp_name = df_cleaned.columns[df_time_idx]
            
            # Convert
            df_cleaned.iloc[:, df_time_idx] = pd.to_datetime(df_cleaned.iloc[:, df_time_idx],
                                                             format=ts_fmt,
                                                             errors='coerce')

            # Drop rows with missing timestamp
            rows_before_drop = len(df_cleaned)
            df_cleaned = df_cleaned.dropna(subset=[timestamp_name]).reset_index(drop=True)
            removed_count = rows_before_drop - len(df_cleaned)
            print(f"Converted '{timestamp_name}' to datetime. Dropped {removed_count} rows with missing {timestamp_name}")

        final_row_count = len(df_cleaned)
        print(f"Completed pre-processing. {final_row_count} rows remain.")

    # Generate a column index map based on df_cleaned
    col_map = {name: i for i, name in enumerate(df_cleaned.columns)}

    print(col_map)

    return df_cleaned, col_map


# === Unit Conversion === 

def convert_hw_units(df: pd.DataFrame, 
                     col_map: Dict[str, int], 
                     height_key: Optional[str] = None, 
                     weight_key: Optional[str] = None) -> pd.DataFrame:
    """
    Converts height (feet' inches) to meters and weight (ounces) to kilograms 
    using logical keys and a column mapping.

    Args:
        df: The dataframe to process.
        col_map: Dictionary mapping logical names to physical column indices.
        height_key: The logical key for the height column (e.g., 'height').
        weight_key: The logical key for the weight column (e.g., 'weight').

    Returns:
        DataFrame with converted units and numeric dtypes.
    """
    df_converted = df.copy()
    
    # --- Convert Height ---
    if height_key and height_key in col_map:
        try:
            # get the actual name from the map
            height_col = df_converted.columns[col_map[height_key]]
            
            # split the string column
            height_split = df_converted[height_col].astype(str).str.split("' ", expand=True)

            if height_split.shape[1] == 2:
                feet = pd.to_numeric(height_split[0], errors="coerce")
                inches = pd.to_numeric(height_split[1], errors="coerce")
                
                # calculate height in m
                df_converted[height_col] = 0.0254 * (feet * 12 + inches)
                df_converted[height_col] = pd.to_numeric(df_converted[height_col], errors='coerce')
            else:
                # If it's not in the expected format, at least ensure it's numeric
                df_converted[height_col] = pd.to_numeric(df_converted[height_col], errors="coerce")
                warnings.warn(f"Height column '{height_col}' not in 'ft' in' format; cast to numeric only.")
                
        except Exception as e:
            warnings.warn(f"Height conversion failed for key '{height_key}': {e}")
        
    # --- Convert Weight ---
    if weight_key and weight_key in col_map:
        try:
            # get the actual name from the map
            weight_col = df_converted.columns[col_map[weight_key]]
            
            # convert to numeric and calculate weight in kg
            raw_weight = pd.to_numeric(df_converted[weight_col], errors="coerce")
            df_converted[weight_col] = 0.0283495 * raw_weight
            
        except Exception as e:
            warnings.warn(f"Weight conversion failed for key '{weight_key}': {e}")

    return df_converted


# === Handling Missing Values ===

def drop_missing(df: pd.DataFrame, 
                 config: Dict[str, Optional[List[Any]]], 
                 col_map: Dict[str, int]) -> pd.DataFrame:
    """
    Drops rows containing np.NaN or specified "missing" placeholders (such as "Unknown") 
    using column indices resolved from col_map (name-to-index mapping).

    Arguments:
    - df: The DataFrame to clean.
    - config: {"column_name": [list of values to treat as NaN] or None }
    - col_map: {"column_name": integer_column_index}
    """
    df_cleaned = df.copy()

    for col_name, missing_values in config.items():
        # Get the column index
        if col_name not in col_map:
            print(f"Warning: {col_name} not found in column map. Skipping.")
            continue

        # col name -> col index -> actual col name: safety guard just in case the
        # keys in col_map is not up to date with the actual column names in df
        col_idx = col_map[col_name]
        actual_col_label = df_cleaned.columns[col_idx]

        # log original row count
        rows_before = len(df_cleaned)

        # Replacing the specified values with np.nan
        if missing_values is not None:
            df_cleaned[actual_col_label] = df_cleaned[actual_col_label].replace(missing_values, np.nan)

        # Drop NaNs 
        # first need to retrieve the actual variable name
       
        df_cleaned = df_cleaned.dropna(subset=[actual_col_label])

        # Pring log
        dropped_count = rows_before - len(df_cleaned)
        print(f"Dropped {dropped_count} rows due to missingness in '{col_name}'.")
        print(f"Remaining NaNs in '{col_name}': {df_cleaned[col_name].isna().sum()}")

    return df_cleaned.reset_index(drop=True)

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


def impute_missing(df: pd.DataFrame, 
                   config: Dict[str, Any], 
                   col_map: Dict[str, int]) -> pd.DataFrame:
    """
    Imputes missing values via ffill or linear mixed models).
    
    Arguments:
    - df: The DataFrame to be imputed.
    - col_map: Mapping of {logical_name: physical_index}.
    - config: {
        'ffill': {'target_key': {'timestamp': 'time_key', 'group': 'grp_key'} },
        'lmm': {
            'target_key': {
                'predictors': ['age', 'weight'],
                'group': 'patient_id'
            }
        }
      }
    """
    df_imputed = df.copy()

    # --- Forward Fill ---
    if 'ffill' in config:
        for target_key, params in config['ffill'].items():
            try:
                # get column names as they actually appear in df
                target_label = df_imputed.columns[col_map[target_key]]
                ts_label = df_imputed.columns[col_map[params.get('timestamp')]] if params.get('timestamp') in col_map else None
                grp_label = df_imputed.columns[col_map[params.get('group')]] if params.get('group') in col_map else None

                # sort df by group and timestamp col, if provided
                sort_cols = [col for col in [grp_label, ts_label] if col]
                if sort_cols:
                    df_imputed = df_imputed.sort_values(by=sort_cols)

                # forward fill (grouped or global)
                if grp_label:
                    df_imputed[target_label] = df_imputed.groupby(grp_label)[target_label].ffill()
                else:
                    df_imputed[target_label] = df_imputed[target_label].ffill()
                
                print(f"Forward fill complete for '{target_key}'. Remaining NaNs: {df_imputed[target_label].isna().sum()}")
            
            except KeyError as e:
                print(f"Skipping ffill for '{target_key}': Key {e} not in col_map.")

    # --- LMM Imputation ---
    if 'lmm' in config:
        for target_key, settings in config['lmm'].items():
            print(f"Start imputing {target_key}")
            try:
                pred_keys = settings.get('predictors', [])
                grp_key = settings.get('group')

                # get column names as they actually appear in df
                target_label = df_imputed.columns[col_map[target_key]]
                pred_labels = [df_imputed.columns[col_map[p]] for p in pred_keys]
                grp_label = df_imputed.columns[col_map[grp_key]] if grp_key in col_map else None

                if not grp_label:
                    raise ValueError(f"LMM requires a 'group' column. None found for '{target_key}'.")

                # partition data
                search_cols = pred_labels + [grp_label]
                train_df, test_df, test_idx = partition_by_completeness(df=df_imputed,
                                                                        target=target_label,
                                                                        predictors=search_cols)

                if not train_df.empty and not test_df.empty:
                    # construct model formula and fit model
                    formula = f"{target_label} ~ {' + '.join(pred_labels)}"
                    model = smf.mixedlm(formula, data=train_df, groups=train_df[grp_label]).fit()
                    
                    # predict on missing rows
                    predictions = model.predict(test_df[pred_labels])
                    df_imputed.loc[test_idx, target_label] = predictions
                    
                    print(f"LMM Imputation complete for '{target_key}'.")
                else:
                    print(f"Skipping LMM for '{target_key}': Insufficient data.")

                print(f"Remaining NaNs in '{target_key}': {df_imputed[target_label].isna().sum()}")

            except Exception as e:
                print(f"LMM error on '{target_key}': {e}")

    return df_imputed.reset_index(drop=True)

# === Filtering lab tests ===
def find_lab_name_code_pair(df: pd.DataFrame,
                            test_list: List[str],
                            col_map: Dict[str, int],
                            name_key: str,
                            code_key: str,
                            special_configs: Optional[Dict[str, Dict[str, Any]]] = None
                           ) -> pd.DataFrame:
    """
    Finds the most frequent name-code pairs for tests in test_list.
    Uses col_map to extract actual column names for name_key and code_key as they appear in df.

    Args:
    - df: the DataFrame to be processed
    - test_list: a list of lab test names to be included in df
    - col_map: a column name-to-index mapping for df
    - name_key: The logical key for the Lab Name column in col_map.
    - code_key: The logical key for the Lab Code column in col_map.
    - special_configs: Optional dictionary to specify search logic.
        Example: {
                    "pH": {"pat": r"\bpH\b", "case": True},
                    "Lactate": {"pat": "Lactate", "case": False, "exclude": "D-lactate"}
                }

    Returns:
    A DataFrame containing the most frequent 'name_col' and 'code_col' pair for each test found.
    """
    # Get actual column names as they appear in df
    
    print("Filtering lab tests results...")
    
    try:
        name_col = df.columns[col_map[name_key]]
        code_col = df.columns[col_map[code_key]]
    except (KeyError, IndexError):
        print(f"Error: {name_key} or {code_key} not found in col_map/dataframe.")
        return pd.DataFrame()

    special_configs = special_configs or {}
    selected_pairs = []

    for test in test_list:
        cfg = special_configs.get(test, {})
        search_pat = cfg.get("pat", test)
        is_case = cfg.get("case", False)
        is_regex = cfg.get("regex", False)
        exclusion = cfg.get("exclude", None)

        # Filter for current test pattern
        mask = df[name_col].str.contains(search_pat, case=is_case, regex=is_regex, na=False)
        if exclusion:
            mask &= ~df[name_col].str.contains(exclusion, case=False, na=False)

        test_matches = df[mask]
        
        # Get most common Name-Code combination
        if not test_matches.empty:
            best_pair = test_matches.groupby([name_col, code_col]).size().idxmax()
            
            selected_pairs.append({
                name_col: best_pair[0],
                code_col: best_pair[1]
            })
    pairs_df = pd.DataFrame(selected_pairs)
    print(f"Selected pairs: {pairs_df}")

    return pd.DataFrame(pairs_df)


# for testing
# def main():
#     labs_raw = pd.read_csv("raw/patient_labs.csv")
#     id_cols = ["LOG_ID", "MRN"]
#     feature_cols = ["Lab Code", "Lab Name", "Observation Value", "Measurement Units", "Collection Datetime"]

#     labs, labs_col_map = pre_process(labs_raw, id_cols, feature_cols,
#                                     timestamp_idx = [4],
#                                     timestamp_formats = ["%Y-%m-%d %H:%M:%S"])

#     labs_drop_config = {
#         "Observation Value": [9999999.0]
#     }

#     predefined_tests = ['Leukocytes', 'pH', 'Hematocrit', 'C reactive protein', 'Lactate']
#     lab_search_config = {"pH": {"pat": r"\bpH\b", "case": True, "regex": True},
#                          "Lactate": {"exclude": "D-Lactate"}
#                          }

#     pairs_df = find_lab_name_code_pair(df=labs, test_list = predefined_tests,
#                                       col_map=labs_col_map, name_key = "Lab Name", code_key = "Lab Code",
#                                       special_configs = lab_search_config)
    
#     # info_raw = pd.read_csv("raw/patient_information.csv")
#     # info_raw.rename(columns = {"BIRTH_DATE": "AGE"}, inplace = True)
#     # id_cols = ["LOG_ID", "MRN"]
#     # feature_cols = ["AGE", "HEIGHT", "WEIGHT", "SEX", "AN_START_DATETIME"]
#     # info, info_col_map = pre_process(info_raw, id_cols, feature_cols,
#     #                                  timestamp_idx = [4],
#     #                                  timestamp_formats = ["%m/%d/%y %H:%M"])

#     # info_impute_config = {
#     #     "ffill": {
#     #         "HEIGHT": {"timestamp": "AN_START_DATETIME", "group": "MRN"},
#     #         "WEIGHT": {"timestamp": "AN_START_DATETIME", "group": "MRN"},
#     #     },
#     #     "lmm": {
#     #         "HEIGHT": {"predictors": ["SEX", "WEIGHT"], "group": "MRN"},
#     #         "WEIGHT": {"predictors": ["SEX", "HEIGHT"], "group": "MRN"}
#     #     }
#     # }

#     # info_drop_config = {
#     #     "SEX": ["Unknown"],
#     #     "HEIGHT": None,
#     #     "WEIGHT": None # dropping remaining NAs in height/weight after imputation
#     # }


#     # # convert hw
#     # info = convert_hw_units(df=info, 
#     #                         col_map = info_col_map,
#     #                         height_key = "HEIGHT",
#     #                         weight_key = "WEIGHT")
    
#     # # We have one row where SEX == "Unknown"
#     # # But AN_START_DATETIME for that row is also missing, so that row is already dropped in pre_process
#     # # So a msg "Dropped 0 rows due to missingness in 'SEX'" will pop up and this is expected.
#     # info_cleaned = impute_missing(info,
#     #                               config = info_impute_config,
#     #                               col_map = info_col_map)

#     # info_cleaned = drop_missing(info_cleaned,
#     #                             config = info_drop_config,
#     #                             col_map = info_col_map)



# if __name__ == "__main__":
#     main()

    
    