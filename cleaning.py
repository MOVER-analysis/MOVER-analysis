# === Import libraries === 
import pandas as pd
import numpy as np
import re

# for regression imputation
import statsmodels.formula.api as smf

# for function definition and calls
from typing import List, Tuple, Callable, Dict, Any, Optional

# import preprocessing functions
import preprocess as prep

# === Data Cleaning ===
def clean_df(df: pd.DataFrame, 
             col_map: Dict[str, int], 
             height_key: Optional[str] = None, 
             weight_key: Optional[str] = None,
             impute_config: Optional[Dict[str, Any]] = None,
             drop_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Orchestrates the cleaning pipeline: 
    1. Unit conversion
    2. Imputing and/or dropping missing values
    
    Returns the cleaned DataFrame.
    """
    df_working = df.copy()
    
    # Convert height and weight
    if height_key or weight_key:
        print("Converting units...")
        df_working = prep.convert_hw_units(df=df_working, 
                                           col_map=col_map,
                                           height_key=height_key, 
                                           weight_key=weight_key)

    # Impute missing values
    if impute_config:
        print("Performing imputation...")
        df_working = prep.impute_missing(df=df_working, 
                                         config=impute_config, 
                                         col_map=col_map)

    # Drop missing value
    if drop_config:
        print("Dropping remaining missings...")
        df_working = prep.drop_missing(df=df_working,
                                       config=drop_config,
                                       col_map=col_map)

    # Reporting
    final_rows = len(df_working)
    print("--- Cleaning Complete ---")
    print(f"Rows Retained: {final_rows}")
    
    return df_working.reset_index(drop=True)

def merge_info_labs(info_df: pd.DataFrame, info_map: Dict[str, int], 
                    labs_df: pd.DataFrame, labs_map: Dict[str, int],
                    name_key: str, code_key: str,
                    predefined_tests: List[str],
                    special_configs: Optional[Dict[str, Dict[str, Any]]] = None,
                    use_common_tests: bool = True) -> pd.DataFrame:
    """
    Standardized Merge: Filters labs and joins with info, accounting for 
    different physical column names for IDs and timestamps.
    """
    
    # --- Filtering labs ---
    labs_subset_map = {"encounter_id": labs_map["encounter_id"],
                       "patient_id": labs_map["patient_id"],
                       "name": labs_map[name_key],
                       "code": labs_map[code_key]}
    
    labs_filtered = prep.filter_labs(df=labs_df,
                                     subset_map=labs_subset_map,
                                     predefined_tests=predefined_tests,
                                     special_configs=special_configs,
                                     use_common_tests=use_common_tests)

    # --- Get column names for both dataframes and merge ---
    # for info
    info_enc = info_df.columns[info_map["encounter_id"]]
    info_pat = info_df.columns[info_map["patient_id"]]
    info_ts  = info_df.columns[info_map["timestamp"]]

    # for labs
    labs_enc = labs_df.columns[labs_map["encounter_id"]]
    labs_pat = labs_df.columns[labs_map["patient_id"]]
    labs_ts  = labs_df.columns[labs_map["timestamp"]]
    lab_name = labs_df.columns[labs_map[name_key]]

    print(f"Merging Info df with Labs df ...")

    merged_df = info_df.merge(labs_filtered, 
                              left_on=[info_enc, info_pat],
                              right_on=[labs_enc, labs_pat],
                              how='inner')

    # cleanup column names
    if info_enc != labs_enc:
        merged_df = merged_df.drop(columns=[labs_enc])
    if info_pat != labs_pat:
        merged_df = merged_df.drop(columns=[labs_pat])

    # --- Filter for the latest pre-operative lab test ---
    merged_df = merged_df[merged_df[labs_ts] <= merged_df[info_ts]].copy()
    idx = merged_df.groupby([info_enc, info_pat, lab_name])[labs_ts].idxmax()
    merged_df = merged_df.loc[idx]

    # --- Drop duplicates ---
    final_df = merged_df.drop_duplicates().reset_index(drop=True)
    print(f"Merge complete. {len(final_df)} latest labs retained.")
    
    return final_df

# testing
def main():
    # read in data
    print("--- Loading data files ---")
    info_raw = pd.read_csv("raw/patient_information.csv")
    info_raw.rename(columns = {"BIRTH_DATE": "AGE"}, inplace = True)
    labs_raw = pd.read_csv("raw/patient_labs.csv")
    postop_raw = pd.read_csv("raw/patient_post_op_complications.csv")

    # define constants for variable names
    encounter_id = "LOG_ID"
    patient_id = "MRN"

    # for information
    age = "AGE"
    height = "HEIGHT"
    weight = "WEIGHT"
    sex = "SEX"
    info_ts = "AN_START_DATETIME"

    # for labs df and lab tests
    lab_code = "Lab Code"
    lab_name = "Lab Name"
    value = "Observation Value"
    unit = "Measurement Units"
    labs_ts = "Collection Datetime"

    leukocytes = "Leukocytes"
    ph = "pH"
    hematocrit = "Hematocrit"
    crp = "C reactive protein"
    lactate = "Lactate"

    # for complications
    complications = "SMRTDTA_ELEM_VALUE"

    # define cols to keep
    id_cols = [encounter_id, patient_id]
    
    info_features = [age, height, weight, sex, info_ts]
    info_ts_idx = info_features.index(info_ts)
    
    labs_features = [lab_code, lab_name, value, unit, labs_ts]
    labs_ts_idx = labs_features.index(labs_ts)

    postop_features = [complications]

    # define configurations
    info_impute_config = {
        "ffill": {
            height: {"timestamp": info_ts, "group": patient_id},
            weight: {"timestamp": info_ts, "group": patient_id},
        },
        "lmm": {
            height: {"predictors": [sex, weight], "group": patient_id},
            weight: {"predictors": [sex, height], "group": patient_id}
        }
    }

    info_drop_config = {
        sex: ["Unknown"],
        height: None,
        weight: None # dropping remaining NAs in height/weight after imputation
    }
    
    labs_drop_config = {value: [9999999.0]}

    predefined_tests = [leukocytes, ph, hematocrit, crp, lactate]
    lab_search_config = {ph: {"pat": r"\bpH\b", "case": True, "regex": True},
                         lactate: {"exclude": "D-Lactate"}}

    # preprocess
    print("--- Pre-processing ---")
    print("Preprocessing information data...")
    info, info_map = prep.pre_process(df=info_raw, 
                                      id_cols=id_cols, 
                                      feature_cols=info_features,
                                      timestamp_idx=info_ts_idx,
                                      timestamp_format="%m/%d/%y %H:%M")
    print("Preprocessing labs data...")
    labs, labs_map = prep.pre_process(df=labs_raw, 
                                      id_cols=id_cols, 
                                      feature_cols=labs_features,
                                      timestamp_idx=labs_ts_idx,
                                      timestamp_format="%Y-%m-%d %H:%M:%S")
    print("Preprocessing postoperative complications data...")
    postop, postop_map = prep.pre_process(df=postop_raw,
                                         id_cols=id_cols,
                                         feature_cols=postop_features)

    # cleanup
    print("--- Clean patient informations data ---")
    info_clean = clean_df(df=info,
                          col_map=info_map,
                          height_key=height,
                          weight_key=weight,
                          impute_config=info_impute_config,
                          drop_config=info_drop_config)
    
    print("--- Clean patient labs data ---")
    labs_clean = clean_df(df=labs,
                         col_map=labs_map,
                         drop_config=labs_drop_config)

    # merge and filter lab tests
    print("--- Filtering lab tests ---")
    info_labs = merge_info_labs(info_df=info_clean, 
                                info_map=info_map,
                                labs_df=labs_clean, 
                                labs_map=labs_map,
                                name_key=lab_name, 
                                code_key=lab_code,
                                predefined_tests=predefined_tests,
                                special_configs=lab_search_config)

    # merge with postoperative complications
    print("--- Merging with complications data ---")
    df_final = postop.merge(info_labs, on=id_cols, how="inner")
    df_final = df_final.drop_duplicates().reset_index(drop=True)

    # --- 5. Export to CSV ---
    output_filename = "data/cleaned_data.csv"
    df_final.to_csv(output_filename, index=False)

    print(f"Cleaning complete. Final row count: {len(df_final)}")
    
if __name__ == "__main__":
    main()
    
    
    
   
