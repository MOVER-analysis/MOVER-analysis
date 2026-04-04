# Function
def main():

    # Running the pipeline if the required files and .py scripts are present
    try:
        # Importing other .py scripts
        from supporting_scripts import setup
        from supporting_scripts import preprocess
        from supporting_scripts import cleaning

        # Importing library
        import pandas as pd

        # Loading config file
        config = setup.load_yaml()

        # Storing file constants
        RAW_DATA_PATH = config["data"]["raw_data_path"]
        RAW_INFO_NAME = RAW_DATA_PATH + "patient_information.csv"
        RAW_LABS_NAME = RAW_DATA_PATH + "patient_labs.csv"
        RAW_POSTOP_NAME = RAW_DATA_PATH + "patient_post_op_complications.csv"

        OUTPUT_DATA_PATH = config["data"]["output_data_path"]
        OUTPUT_DATA_NAME = OUTPUT_DATA_PATH + "final_data.csv"

        # Creating an output folder
        setup.create_output_folder(OUTPUT_DATA_PATH)
        
        # Printing a start message
        print("––––––––STARTING PIPELINE––––––––")

        # Preprocessing
        print("––––––––PREPROCESSING DATA––––––––")

        ## Loading data files
        info_raw = setup.load_data(RAW_INFO_NAME)
        info_raw.rename(columns = {"BIRTH_DATE": "AGE"}, inplace = True)
        labs_raw = setup.load_data(RAW_LABS_NAME)
        postop_raw = setup.load_data(RAW_POSTOP_NAME)

        ## Defining constants for variable names

        ### ID variables
        encounter_id = "LOG_ID"
        patient_id = "MRN"

        ### Information variables
        age = "AGE"
        height = "HEIGHT"
        weight = "WEIGHT"
        sex = "SEX"
        info_ts = "AN_START_DATETIME"

        ### Laboratory variables
        lab_code = "Lab Code"
        lab_name = "Lab Name"
        value = "Observation Value"
        unit = "Measurement Units"
        labs_ts = "Collection Datetime"

        ### Predefined laboratory tests
        leukocytes = "Leukocytes"
        ph = "pH"
        hematocrit = "Hematocrit"
        crp = "C reactive protein"
        lactate = "Lactate"

        ### Complication variable
        complications = "SMRTDTA_ELEM_VALUE"

        ## Defining columns to keep
        id_cols = [encounter_id, patient_id]

        info_features = [age, height, weight, sex, info_ts]
        info_ts_idx = info_features.index(info_ts)

        labs_features = [lab_code, lab_name, value, unit, labs_ts]
        labs_ts_idx = labs_features.index(labs_ts)

        postop_features = [complications]

        ## Defining configurations
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

        ## Preprocessing
        print("Preprocessing information data...")
        info, info_map = preprocess.pre_process(df=info_raw,
                                                id_cols=id_cols,
                                                feature_cols=info_features,
                                                timestamp_idx=info_ts_idx,
                                                timestamp_format="%m/%d/%y %H:%M")

        print("Preprocessing laboratory data...")
        labs, labs_map = preprocess.pre_process(df=labs_raw,
                                                id_cols=id_cols,
                                                feature_cols=labs_features,
                                                timestamp_idx=labs_ts_idx,
                                                timestamp_format="%Y-%m-%d %H:%M:%S")

        print("Preprocessing postoperative complications data...")
        postop, postop_map = preprocess.pre_process(df=postop_raw,
                                                    id_cols=id_cols,
                                                    feature_cols=postop_features)

        print("Preprocessing complete.")
        
        # Cleaning
        print("––––––––CLEANING DATA––––––––")

        ## Cleaning
        print("Cleaning information data...")
        info_clean = cleaning.clean_df(df=info,
                                       col_map=info_map,
                                       height_key=height,
                                       weight_key=weight,
                                       impute_config=info_impute_config,
                                       drop_config=info_drop_config)

        print("Cleaning laboratory data...")
        labs_clean = cleaning.clean_df(df=labs,
                                       col_map=labs_map,
                                       drop_config=labs_drop_config)

        ## Merging and filtering laboratory tests
        print("Filtering laboratory tests...")
        info_labs = cleaning.merge_info_labs(info_df=info_clean,
                                             info_map=info_map,
                                             labs_df=labs_clean,
                                             labs_map=labs_map,
                                             name_key=lab_name,
                                             code_key=lab_code,
                                             predefined_tests=predefined_tests,
                                             special_configs=lab_search_config)

        ## Merging with postoperative complications
        print("Merging with complications data...")
        df_final = postop.merge(info_labs, on=id_cols, how="inner")
        df_final = df_final.drop_duplicates().reset_index(drop=True)

        ## Export to .csv
        df_final.to_csv(OUTPUT_DATA_NAME, index=False)

        print(f"Cleaning complete. Final row count: {len(df_final)}")
    except ModuleNotFoundError as e:
        print(f"Module not found: {e}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()