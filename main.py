# Function
def main():

    # Running the pipeline if the required files and .py scripts are present
    try:
        # Importing other .py scripts
        from supporting_scripts import setup
        from supporting_scripts import preprocess
        from supporting_scripts import cleaning
        from supporting_scripts import feature_engineering
        from supporting_scripts import data_restructuring
        from supporting_scripts import data_validation
        from supporting_scripts import cohort_split
        from supporting_scripts import create_plots
        from supporting_scripts import log_transform
        from supporting_scripts import analysis

        # Importing libraries
        import pandas as pd
        from sklearn.model_selection import train_test_split
        
        # Printing a start message
        print("––––––––STARTING PIPELINE––––––––")

        # Loading configuration file
        config = setup.load_yaml()

        # Storing constants

        ## Files
        RAW_INFO_NAME = config["data"]["raw_data"]["raw_info_path"]
        RAW_LABS_NAME = config["data"]["raw_data"]["raw_labs_path"]
        RAW_POSTOP_NAME = config["data"]["raw_data"]["raw_postop_complications_path"]

        OUTPUT_DATA_PATH = config["output"]["output_data_path"]
        OUTPUT_VALIDATION_PATH = config["output"]["validation_data_path"]
        OUTPUT_PLOTS_PATH = config["output"]["plots_folder"]
        OUTPUT_MODEL_PATH = config["output"]["model_folder"]
        
        OUTPUT_DATA_NAME = config["data"]["output_data"]["final_data_path"]
        OUTPUT_TRAIN_NAME = config["data"]["output_data"]["training_set_path"]
        OUTPUT_TEST_NAME = config["data"]["output_data"]["test_set_path"]
        
        OUTPUT_FULL_MODEL_RESULTS_FILE = config["output"]["full_model_results_file"]
        OUTPUT_FULL_MODEL_RATE_FILE = config["output"]["full_model_rate_file"]
        OUTPUT_REDUCED_MODEL_RATE_FILE = config["output"]["reduced_model_rate_file"]

        ## Variables
        encounter_id = "LOG_ID"
        patient_id = "MRN"
        
        age = "AGE"
        age_original = "BIRTH_DATE"
        age_final = "age"
        age_type = "continuous"
        age_name = "Age"
        age_units = "years"
        age_lowercase = True

        height = "HEIGHT"
        height_missing = None
        height_units = "m"
        weight = "WEIGHT"
        weight_missing = None
        weight_units = "kg"
        bmi = "bmi"
        bmi_type = "continuous"
        bmi_name = "Body mass index"
        bmi_units = "kg/m^2"
        bmi_lowercase = True

        sex = "SEX"
        sex_missing = ["Unknown"]
        sex_target_value = "female"
        sex_final = "sex"
        sex_type = "categorical"
        sex_name = "Sex"
        sex_units = ""
        sex_lowercase = True
        sex_bar_colours = ["blue", "red"]
        sex_xangle = 45
        sex_class_names = {0: "Male", 1: "Female"}

        leukocytes = "Leukocytes"
        leukocytes_alt = "Leukocytes^^corrected for nucleated erythrocytes"
        leukocytes_final = "leukocytes"
        leukocytes_type = "continuous"
        leukocytes_name = "Leukocytes"
        leukocytes_units = "THOUS/MCL"
        leukocytes_lowercase = True

        ph = "pH"
        ph_final = "ph"

        hematocrit = "Hematocrit"
        hematocrit_final = "hematocrit"
        hematocrit_type = "continuous"
        hematocrit_name = "Hematocrit"
        hematocrit_units = "%"
        hematocrit_lowercase = True

        crp = "C reactive protein"
        crp_final = "crp"

        lactate = "Lactate"
        lactate_final = "lactate"

        co2 = "Carbon dioxide"
        co2_final = "co2"
        co2_type = "continuous"
        co2_name = "Carbon dioxide"
        co2_units = "mmol/L"
        co2_lowercase = True

        glucose = "Glucose"
        glucose_final = "glucose"
        glucose_type = "continuous"
        glucose_name = "Glucose"
        glucose_units = "mg/dL"
        glucose_lowercase = True

        hemoglobin = "Hemoglobin"
        hemoglobin_final = "hemoglobin"
        hemoglobin_type = "continuous"
        hemoglobin_name = "Hemoglobin"
        hemoglobin_units = "G/DL"
        hemoglobin_lowercase = True

        potassium = "Potassium"
        potassium_final = "potassium"
        potassium_type = "continuous"
        potassium_name = "Potassium"
        potassium_units = "mmol/L"
        potassium_lowercase = True

        sodium = "Sodium"
        sodium_final = "sodium"
        sodium_type = "continuous"
        sodium_name = "Sodium"
        sodium_units = "mmol/L"
        sodium_lowercase = True

        outcome = "hypoxemia"
        outcome_class_0_name = "Non-Hypoxemia"
        outcome_class_1_name = "Hypoxemia"

        info_ts = "AN_START_DATETIME"
        lab_code = "Lab Code"
        lab_name = "Lab Name"
        value = "Observation Value"
        value_missing = [9999999.0]
        unit = "Measurement Units"
        labs_ts = "Collection Datetime"
        complications = "SMRTDTA_ELEM_VALUE"

        ## Timestamp formats
        info_timestamp_format = "%m/%d/%y %H:%M"
        labs_timestamp_format = "%Y-%m-%d %H:%M:%S"

        ## Threshold for dropping columns based on missingness proportion
        drop_threshold = config["data"]["validation"]["drop_threshold"]

        ## Constants for case-control sampling
        random_state = config["data"]["case_control_split"]["random_state"]
        outcome_case_value = config["data"]["case_control_split"]["outcome_case_value"]
        neg_sample_size = config["data"]["case_control_split"]["neg_sample_size"]
        test_prop = config["data"]["case_control_split"]["test_prop"]
        value_to_replace = config["data"]["case_control_split"]["value_to_replace"]

        ## Level of significance for model results
        alpha = config["model"]["alpha"]

        ## Threshold for classification when evaluating the reduced model
        classification_threshold = config["model"]["classification_threshold"]

        # Creating output folders
        setup.create_output_folder(OUTPUT_DATA_PATH)
        setup.create_output_folder(OUTPUT_VALIDATION_PATH)
        setup.create_output_folder(OUTPUT_PLOTS_PATH)
        setup.create_output_folder(OUTPUT_MODEL_PATH)

        # Loading data files
        print("Loading raw data...")
        info_raw = setup.load_data(RAW_INFO_NAME)
        labs_raw = setup.load_data(RAW_LABS_NAME)
        postop_raw = setup.load_data(RAW_POSTOP_NAME)

        # Preprocessing
        print("––––––––PREPROCESSING DATA––––––––")

        ## Renaming a column for clarity
        info_raw.rename(columns = {age_original: age}, inplace = True)

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
            sex: sex_missing,
            height: height_missing,
            weight: weight_missing
            # dropping remaining NAs in height/weight after imputation
        }

        labs_drop_config = {value: value_missing}

        predefined_tests = [leukocytes, ph, hematocrit, crp, lactate]
        lab_search_config = {ph: {"pat": r"\bpH\b", "case": True, "regex": True},
                             lactate: {"exclude": "D-Lactate"}}

        ## Preprocessing
        print("Preprocessing information data...")
        info, info_map = preprocess.pre_process(df=info_raw,
                                                id_cols=id_cols,
                                                feature_cols=info_features,
                                                timestamp_idx=info_ts_idx,
                                                timestamp_format=info_timestamp_format)

        print("Preprocessing laboratory data...")
        labs, labs_map = preprocess.pre_process(df=labs_raw,
                                                id_cols=id_cols,
                                                feature_cols=labs_features,
                                                timestamp_idx=labs_ts_idx,
                                                timestamp_format=labs_timestamp_format)

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
        df = postop.merge(info_labs, on=id_cols, how="inner")
        df = df.drop_duplicates().reset_index(drop=True)

        print(f"Cleaning complete. Final row count: {len(df)}")

        # Feature engineering
        print("––––––––FEATURE ENGINEERING––––––––")

        ## Configuration for creating binary indicator columns
        BINARY_COL_CONFIG = [
            {
                "source_col": complications,
                "new_col": outcome,
                "target_value": outcome,
                "match_type": "in"
            },
            {
                "source_col": sex,
                "new_col": sex,
                "target_value": sex_target_value,
                "match_type": "exact"
            }
        ]
        df = feature_engineering.create_binary_cols(df, BINARY_COL_CONFIG)

        ## Keeping only rows with hypoxemia = 1 for encounters with multiple complications that have
        ## multiple rows that include both hypoxemia = 1 and hypoxemia = 0 with all other variables the same
        has_hypoxemia = df.groupby(id_cols)[outcome].transform("max")
        df = df[~((df[outcome] == 0) & (has_hypoxemia == 1))].copy()
        df = df.drop(columns=[complications])
        df = df.drop_duplicates()

        ## Configuration for calculating BMI from height and weight
        BMI_CONFIG = {
            "height_col": (height, height_units),
            "weight_col": (weight, weight_units),
            "new_col": bmi
        }
        df = feature_engineering.calculate_bmi(df, BMI_CONFIG)

        print("Feature engineering complete.")

        # Data restructuring
        print("––––––––DATA RESTRUCTURING––––––––")

        ## Defining constants
        drop_cols = [lab_code, unit, labs_ts]
        outcome_cols = [outcome]
        demo_cols = [age, sex, bmi]

        ## Renaming selected columns
        col_name_map = {
            age: age_final,
            sex: sex_final,
            bmi: bmi
        }
        drop_cols = [col_name_map.get(item, item) for item in drop_cols]
        outcome_cols = [col_name_map.get(item, item) for item in outcome_cols]
        demo_cols = [col_name_map.get(item, item) for item in demo_cols]
        df = df.rename(columns=col_name_map)
        print("Selected columns renamed successfully.")

        ## Dropping long-format columns before pivoting
        df = df.drop(columns=drop_cols, errors="ignore")

        ## Renaming lab tests
        lab_name_map = {
            crp: crp_final,
            co2: co2_final,
            glucose: glucose_final,
            hematocrit: hematocrit_final,
            hemoglobin: hemoglobin_final,
            leukocytes_alt: leukocytes_final,
            potassium: potassium_final,
            sodium: sodium_final,
            ph: ph_final,
            lactate: lactate_final
        }
        df[lab_name] = df[lab_name].map(lab_name_map)
        print("Laboratory tests renamed successfully.")

        ## Reshaping the data wider
        df_wide = data_restructuring.pivot_wider(df, id_cols, lab_name, value)

        ## Getting the names of the new wide-format lab columns
        lab_cols = df[lab_name].dropna().unique().tolist()

        ## Keeping only required columns that exist in the wide dataframe
        keep_cols = id_cols + outcome_cols + lab_cols + demo_cols
        keep_cols = [col for col in keep_cols if col in df_wide.columns]
        print("Final columns selected successfully.")

        df = df_wide[keep_cols]

        print("Data restructuring complete.")

        # Data validation
        print("––––––––DATA VALIDATION––––––––")

        ## Defining required columns
        required_cols = keep_cols

        ## Checking duplicate encounter keys
        data_validation.check_duplicate_keys(df, id_cols)

        ## Verifying required columns are present
        data_validation.check_required_columns(df, required_cols)

        ## Checking expected data types (expected column groups by type)
        expected_type_cols = {
            "string": id_cols,
            "numeric": lab_cols + [col for col in demo_cols if col != sex_final],
            "binary": outcome_cols + [sex_final]
        }
        data_validation.check_expected_column_types(df, expected_type_cols)

        ## Screening for implausible values
        data_validation.check_implausible_values(df, expected_type_cols)

        ## Checking missing values
        ## (threshold for dropping columns based on missingness proportion)
        df = data_validation.check_missingness(df, drop_threshold = drop_threshold)

        print("Data validation complete.")

        ## Saving final dataset
        print("Saving final dataset...")
        df.to_csv(OUTPUT_DATA_NAME, index=False)
        print(f"Saved final dataset to {OUTPUT_DATA_NAME}")

        # Cohort split
        print("––––––––COHORT SPLIT––––––––")

        sample = cohort_split.case_control_sample(df,
                                                  outcome,
                                                  outcome_case_value,
                                                  neg_sample_size)
        print(f"Sampled dataset shape: {sample.shape}")

        ## Performing a stratified train/test split
        train_df, test_df = train_test_split(
            sample,
            test_size = test_prop,
            stratify = sample[outcome],
            random_state = random_state
        )

        ## Resetting index
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        ## Printing outcome counts
        train_counts = train_df[outcome].value_counts()
        test_counts = test_df[outcome].value_counts()

        split_count = pd.DataFrame({
            "train": train_counts,
            "test": test_counts
        }).T.fillna(value_to_replace).astype(int)
        print("\nOutcome counts (train/test):")
        print(split_count)

        print("Cohort split complete.")

        ## Saving training and test sets
        print("Saving training set...")
        train_df.to_csv(OUTPUT_TRAIN_NAME, index=False)
        print(f"Saved training set to {OUTPUT_TRAIN_NAME}")

        print("Saving test set...")
        test_df.to_csv(OUTPUT_TEST_NAME, index=False)
        print(f"Saved test set to {OUTPUT_TEST_NAME}")

        # Creating and saving plots
        print("––––––––CREATING & SAVING PLOTS––––––––")

        ## Setting up features_dict and outcome_list for plotting
        features_dict = {
            co2_final: {
                "type": co2_type,
                "name": co2_name,
                "units": co2_units,
                "lowercase": co2_lowercase
            },
            glucose_final: {
                "type": glucose_type,
                "name": glucose_name,
                "units": glucose_units,
                "lowercase": glucose_lowercase
            },
            hematocrit_final: {
                "type": hematocrit_type,
                "name": hematocrit_name,
                "units": hematocrit_units,
                "lowercase": hematocrit_lowercase
            },
            hemoglobin_final: {
                "type": hemoglobin_type,
                "name": hemoglobin_name,
                "units": hemoglobin_units,
                "lowercase": hemoglobin_lowercase
            },
            leukocytes_final: {
                "type": leukocytes_type,
                "name": leukocytes_name,
                "units": leukocytes_units,
                "lowercase": leukocytes_lowercase
            },
            potassium_final: {
                "type": potassium_type,
                "name": potassium_name,
                "units": potassium_units,
                "lowercase": potassium_lowercase
            },
            sodium_final: {
                "type": sodium_type,
                "name": sodium_name,
                "units": sodium_units,
                "lowercase": sodium_lowercase
            },
            age_final: {
                "type": age_type,
                "name": age_name,
                "units": age_units,
                "lowercase": age_lowercase
            },
            sex_final: {
                "type": sex_type,
                "name": sex_name,
                "units": sex_units,
                "lowercase": sex_lowercase,
                "bar_colours": sex_bar_colours,
                "xangle": sex_xangle,
                "class_names": sex_class_names
            },
            bmi: {
                "type": bmi_type,
                "name": bmi_name,
                "units": bmi_units,
                "lowercase": bmi_lowercase
            }
        }

        outcome_list = [outcome, outcome_class_0_name, outcome_class_1_name]

        ## Creating and saving plots
        create_plots.create_plots(train_df,
                                  features_dict,
                                  outcome_list,
                                  OUTPUT_PLOTS_PATH)

        print("Creating and saving plots complete.")

        # Log-transforming skewed variables
        print("––––––––LOG-TRANSFORMING SKEWED VARIABLES––––––––")

        ## Setting up df_file_path_list and var_list for log-transformation
        df_file_path_list = [OUTPUT_TRAIN_NAME, OUTPUT_TEST_NAME]
        var_list = [glucose_final, leukocytes_final]

        ## Log-transforming the desired variables and
        ## saving the modified training and test sets
        df_dict = log_transform.log_transform(df_file_path_list, var_list)
        train_df = df_dict[OUTPUT_TRAIN_NAME]
        test_df = df_dict[OUTPUT_TEST_NAME]

        print("Log-transformation process complete.")

        # Analysis
        print("––––––––CONDUCTING ANALYSIS––––––––")

        ## Full model analysis
        print("––––– Full Model –––––")

        ### Fitting the full model on the training set
        predictor_var_list = [age_final,
                              sex_final,
                              bmi,
                              co2_final,
                              glucose_final,
                              hematocrit_final,
                              hemoglobin_final,
                              leukocytes_final,
                              potassium_final,
                              sodium_final]

        full_model = analysis.fit_model(train_df, outcome, predictor_var_list)

        ### Storing the significant predictors
        significant_predictors = analysis.get_significant_predictors(full_model, alpha)

        ### Printing a message to show the significant predictors if any exist
        if len(significant_predictors) > 0:
            print(f"The significant predictors from the full model "
                  f"are: {significant_predictors}")
        else:
            print("No significant predictors found.")

        ### Saving the results of the full model
        analysis.save_full_model(full_model,
                                 OUTPUT_MODEL_PATH,
                                 OUTPUT_FULL_MODEL_RESULTS_FILE)

        ### Evaluating the full model on the test set and saving the test set
        ### misclassification error rate for comparison
        analysis.evaluate_model(full_model,
                                test_df,
                                outcome,
                                OUTPUT_MODEL_PATH,
                                OUTPUT_FULL_MODEL_RATE_FILE,
                                classification_threshold)

        ## Reduced model analysis
        print("––––– Reduced Model –––––")

        ### Fitting the reduced model on the training set using only the
        ### identified significant predictors
        reduced_model = analysis.fit_model(train_df, outcome, significant_predictors)

        ### Evaluating the reduced model on the test set and saving the test set
        ### misclassification error rate
        analysis.evaluate_model(reduced_model,
                                test_df,
                                outcome,
                                OUTPUT_MODEL_PATH,
                                OUTPUT_REDUCED_MODEL_RATE_FILE,
                                classification_threshold)

        print("Analysis complete.")
        
        print("––––––––PIPELINE COMPLETE––––––––")
        
    except FileNotFoundError as e:
        print(f"{e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()