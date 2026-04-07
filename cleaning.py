# === Import libraries === 
import pandas as pd
import numpy as np
import re

# for regression imputation
import statsmodels.formula.api as smf

# for function definition and calls
from typing import List, Tuple, Callable, Dict, Any, Optional

# import preprocessing functions
from supporting_scripts import preprocess as prep

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
    
    
   
