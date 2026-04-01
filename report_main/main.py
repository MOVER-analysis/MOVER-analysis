# Function
def main():
    
    # Running the pipeline if the required files and .py scripts are present
    try:
        # Importing other .py scripts
        from supporting_scripts import setup
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

        # Loading files and storing constants
        config = setup.load_yaml()

        RAW_DATA_PATH = config["data"]["raw_data_path"]
        OUTPUT_DATA_PATH = config["data"]["output_data_path"]

        # Creating an output data folder
        setup.create_output_folder(OUTPUT_DATA_PATH)
        
        # Printing a start message
        print("––––––––STARTING PIPELINE––––––––")
        
        # Cleaning
        print("––––––––CLEANING DATA––––––––")

        ## Defining file paths
        input_files = ["patient_information.csv",
                       "patient_post_op_complications.csv",
                       "patient_labs.csv"]

        raw_files = {} # store .csv files into a dictionary

        output_file = "cleaned_data.csv"

        print("--- Loading data --- ")

        ## Loading raw data files
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
            
        ## Raising an error if file not found
        except FileNotFoundError as e:
            print(f"\n[Error] A required file is missing: ")
            print(f"{e}")
            return

        ## Extracting dataframes
        postop_raw = raw_files["patient_post_op_complications"]
        info_raw = raw_files["patient_information"]
        info_raw.rename(columns = {"BIRTH_DATE": "AGE"}, inplace = True)
        labs_raw = raw_files["patient_labs"]

        print("\nAll data ready for processing")

        ## Defining column mapping configurations
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

        ## Defining missing value configurations
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

        ## Defining constants for filtering lab tests
        predefined_tests = ['Leukocytes', 'pH', 'Hematocrit',
                            'C reactive protein', 'Lactate']
    
        lab_search_config = {"pH": {"pat": r"\bpH\b", "case": True, "regex": True},
                             "Lactate": {"exclude": "D-Lactate"}
                             }

        ## Cleaning datasets
        postop = cleaning.clean_complications(postop = postop_raw,
                                              col_mapping = postop_mapping)

        info = cleaning.clean_information(info = info_raw,
                                          col_mapping = info_mapping,
                                          date_format = "%m/%d/%y %H:%M",
                                          convert_hw = True,
                                          missing_config = info_missing_config)

        labs = cleaning.clean_labs(labs = labs_raw,
                                   col_mapping = labs_mapping,
                                   date_format = "%Y-%m-%d %H:%M:%S",
                                   missing_config = labs_missing_config)

        ## Filtering test results
        labs_filtered = cleaning.filter_labs(labs = labs,
                                             col_mapping = labs_mapping,
                                             predefined_tests = predefined_tests,
                                             special_configs = lab_search_config)

        ## Merge logic optimization
        print("Pre-filtering labs to latest result before anesthesia...")

        ### Filter labs
        timing_info = info[['LOG_ID', 'MRN', 'AN_START_DATETIME']]
        labs_filtered = labs_filtered.merge(timing_info,
                                            on=['LOG_ID', 'MRN'],
                                            how='inner')

        ### Filter for labs before anesthesia
        labs_final = labs_filtered[
            labs_filtered['Collection Datetime'] <= labs_filtered['AN_START_DATETIME']
        ]

        ### Get latest indices
        latest_indices = labs_final.groupby(['LOG_ID', 'MRN', 'Lab Name'])['Collection Datetime'].idxmax()
        labs_final_subset = labs_final.loc[latest_indices].copy()

        ### Drop the AN_START_DATETIME from the subset before the final merge
        ### to avoid AN_START_DATETIME_x / _y
        labs_final_subset = labs_final_subset.drop(columns=['AN_START_DATETIME'])

        ### Final merge
        print("Performing final merge...")
        merged_patient_info = postop.merge(info, on=["LOG_ID", "MRN"], how="inner")

        ### Merging with filtered labs (this will expand the DF to one row per LOG_ID + Lab Test)
        final_df_filtered = merged_patient_info.merge(labs_final_subset, on=["LOG_ID", "MRN"], how="inner")

        print(f"Filtering complete. Final Row Count: {len(final_df_filtered)}")
        ### 695752

        ## Export to csv
        print("--- Export data ---")
        output_path = OUTPUT_DATA_PATH + output_file
        final_df_filtered.to_csv(output_path, index = False)
        print(f"File saved at {output_path}")
        
        # Feature engineering
        print("––––––––PERFORMING FEATURE ENGINEERING––––––––")

        ## Define file paths
        input_file = "cleaned_data.csv"
        output_file = "feature_engineered_data.csv"

        input_path = OUTPUT_DATA_PATH + input_file
        output_path = OUTPUT_DATA_PATH + output_file

        ## Read data
        df = setup.load_data(input_path)

        ## Feature engineering
        ### Config for creating binary indicator columns
        BINARY_COL_CONFIG = {
            "SMRTDTA_ELEM_VALUE": ("hypoxemia", "hypoxemia", "in"),
            "SEX": ("female", "sex", "exact")
        }
        df = feature_engineering.create_binary_cols(df, BINARY_COL_CONFIG)

        ### Config for calculating BMI from height and weight
        BMI_CONFIG = {
            "height_col": ("HEIGHT", "m"),
            "weight_col": ("WEIGHT", "kg"),
            "new_col": "bmi"
        }
        df = feature_engineering.calculate_bmi(df, BMI_CONFIG)

        ## Export to csv
        df.to_csv(output_path, index = False)
        print(f"Feature-engineered data saved to {output_path}.")
        
        # Data restructuring
        print("––––––––RESTRUCTURING DATA––––––––")

        ## Define file paths
        input_file = "feature_engineered_data.csv"
        output_file = "restructured_data.csv"

        input_path = OUTPUT_DATA_PATH + input_file
        output_path = OUTPUT_DATA_PATH + output_file

        ## Read data
        df = setup.load_data(input_path)

        ### Rename selected columns
        #### Columns to rename
        COL_NAME_MAP = {
            "AGE": "age",
            "HEIGHT": "height",
            "WEIGHT": "weight"
        }
        df = df.rename(columns=COL_NAME_MAP)
        print("Selected columns renamed successfully.")

        ## Data restructuring
        ### Drop long-format columns before pivoting
        #### Columns to drop before reshaping data
        DROP_COLS = [
            "Lab Code",
            "Measurement Units",
            "Collection Datetime"
        ]
        df = df.drop(columns=DROP_COLS, errors="ignore")

        ### Rename lab tests
        #### Lab name mapping
        LAB_NAME_MAP = {
            "C reactive protein": "crp",
            "Carbon dioxide": "co2",
            "Glucose": "glucose",
            "Hematocrit": "hematocrit",
            "Hemoglobin": "hemoglobin",
            "Leukocytes^^corrected for nucleated erythrocytes": "leukocytes",
            "Potassium": "potassium",
            "Sodium": "sodium",
            "pH": "pH",
            "Lactate": "lactate"
        }
        df["Lab Name"] = df["Lab Name"].map(LAB_NAME_MAP)
        print("Lab tests renamed successfully.")

        ### Reshape the data wider
        ID_COLS = ["LOG_ID", "MRN"]
        df_wide = data_restructuring.pivot_wider(df, ID_COLS, "Lab Name", "Observation Value")

        ### Get the names of the new wide-format lab columns
        wide_cols = df["Lab Name"].dropna().unique().tolist()

        ### Keep only required columns that exist in the wide dataframe
        #### Columns to keep in the output dataset
        OUTCOME_COLS = ["hypoxemia"]
        DEMO_COLS = ["age", "sex", "bmi", "height", "weight"]

        keep_cols = ID_COLS + OUTCOME_COLS + wide_cols + DEMO_COLS
        keep_cols = [col for col in keep_cols if col in df_wide.columns]
        print("Final columns selected successfully.")

        df_restructured = df_wide[keep_cols]

        ## Export to csv
        df_restructured.to_csv(output_path, index = False)
        print(f"Restructured data saved to {output_path}.")
        
        # Data validation
        print("––––––––VALIDATING DATA––––––––")

        ## Define file paths
        input_file = "restructured_data.csv"
        output_data_file = "validated_data.csv"

        input_path = OUTPUT_DATA_PATH + input_file
        output_data_path = OUTPUT_DATA_PATH + output_data_file
        output_validation_path = "validation/"

        ## Read data
        df = setup.load_data(input_path)

        ### Required columns
        ID_COLS = ["LOG_ID","MRN"]
        OUTCOME_COL = ["hypoxemia"]
        LAB_COLS = ["crp", "co2", "glucose", "hematocrit", "hemoglobin",
                    "leukocytes", "potassium", "sodium", "pH", "lactate"]
        DEMO_COLS = ["age", "sex", "bmi", "height", "weight"]
        REQUIRED_COLS = ID_COLS + OUTCOME_COL + LAB_COLS + DEMO_COLS

        ### Check duplicate encounter keys
        data_validation.check_duplicate_keys(df, ID_COLS)

        ### Verify required columns are present
        data_validation.check_required_columns(df, REQUIRED_COLS)

        ### Check expected data types
        #### Expected column groups by type
        EXPECTED_TYPE_COLS = {
            "string": ["LOG_ID", "MRN"],
            "numeric": LAB_COLS + [col for col in DEMO_COLS if col != "sex"],
            "binary": ["hypoxemia", "sex"]
        }
        data_validation.check_expected_column_types(df, EXPECTED_TYPE_COLS)

        ### Screen for implausible values
        data_validation.check_implausible_values(df, EXPECTED_TYPE_COLS)

        ### Missing value checks
        ### Threshold for dropping columns based on missingness proportion
        DROP_THRESHOLD = 0.5
        df_validated = data_validation.check_missingness(df, drop_threshold = DROP_THRESHOLD)

        ## Export to csv
        df_validated.to_csv(output_data_path, index=False)
        print(f"Validated data saved to {output_data_path}.")
        
        # Cohort construction and train-test split
        print("––––––––COHORT CONSTRUCTION AND TRAIN-TEST SPLIT––––––––")

        ## Define file paths
        input_file = "validated_data.csv"
        output_train_file = "hypoxemia_train.csv"
        output_test_file = "hypoxemia_test.csv"

        input_path = OUTPUT_DATA_PATH + input_file
        train_path = OUTPUT_DATA_PATH + output_train_file
        test_path = OUTPUT_DATA_PATH + output_test_file

        ## Read data
        df = setup.load_data(input_path)

        ## Cohort split
        ### Case-control sampling
        #### Outcome variable name, seed, sample size and test set proportion for random sampling
        OUTCOME = "hypoxemia"
        RANDOM_STATE = 42
        NEG_SAMPLE_SIZE = 200
        TEST_PROP = 0.25

        sample = cohort_split.case_control_sample(df, OUTCOME, 1, NEG_SAMPLE_SIZE)
        print(f"Sampled dataset shape: {sample.shape}")

        ### Stratified train/test split
        train_df, test_df = train_test_split(
            sample,
            test_size = TEST_PROP,
            stratify = sample[OUTCOME],
            random_state = RANDOM_STATE
        )

        #### Reset index
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        #### Print outcome counts
        train_counts = train_df[OUTCOME].value_counts()
        test_counts = test_df[OUTCOME].value_counts()

        split_count = pd.DataFrame({
            "train": train_counts,
            "test": test_counts
        }).T.fillna(0).astype(int)
        print("\nOutcome counts (train/test):")
        print(split_count)

        ## Export to csv
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"\nSaved train data to {train_path}")
        print(f"Saved test data to {test_path}")

        ## Storing training and test set paths and datasets
        training_path = config["data"]["training_set_path"]
        training_set = setup.load_data(training_path)

        test_path = config["data"]["test_set_path"]
        test_set = setup.load_data(test_path)
        
        # Creating and saving plots
        print("––––––––CREATING AND SAVING PLOTS––––––––")

        ## Storing the output folder for the plots
        output_folder = config["output"]["plots_folder"]

        ## Creating the output folder for the plots
        setup.create_output_folder(output_folder)

        ## Setting up features_dict and outcome_list for plotting
        features_dict = {
            "age": {
                "type": "continuous",
                "name": "Age",
                "units": "years",
                "lowercase": True
            },
            "bmi": {
                "type": "continuous",
                "name": "Body mass index",
                "units": "kg/m^2",
                "lowercase": True
            },
            "co2": {
                "type": "continuous",
                "name": "Carbon dioxide",
                "units": "mmol/L",
                "lowercase": True
            },
            "glucose": {
                "type": "continuous",
                "name": "Glucose",
                "units": "mg/dL",
                "lowercase": True
            },
            "hematocrit": {
                "type": "continuous",
                "name": "Hematocrit",
                "units": "%",
                "lowercase": True
            },
            "hemoglobin": {
                "type": "continuous",
                "name": "Hemoglobin",
                "units": "G/DL",
                "lowercase": True
            },
            "leukocytes": {
                "type": "continuous",
                "name": "Leukocytes",
                "units": "THOUS/MCL",
                "lowercase": True
            },
            "potassium": {
                "type": "continuous",
                "name": "Potassium",
                "units": "mmol/L",
                "lowercase": True
            },
            "sodium": {
                "type": "continuous",
                "name": "Sodium",
                "units": "mmol/L",
                "lowercase": True
            },
            "sex": {
                "type": "categorical",
                "name": "Sex",
                "units": "",
                "lowercase": True,
                "bar_colours": ["blue", "red"],
                "xangle": 45,
                "class_names": {0: "Male", 1: "Female"}
            }
        }

        outcome_list = ["hypoxemia", "Non-Hypoxemia", "Hypoxemia"]

        ## Creating and saving plots
        create_plots.create_plots(data, features_dict, outcome_list, output_folder)

        ## Printing a message
        print("Exploratory data analysis complete.")
        
        # Log-transforming variables
        print("––––––––LOG-TRANSFORMING VARIABLES––––––––")

        ## Setting up df_file_path_list and var_list for log-transformation
        df_file_path_list = [training_path, test_path]
        var_list = ["glucose"]
    
        ## Log-transforming the desired variable(s) and saving the modified datasets
        log_transform.log_transform(df_file_path_list, var_list)

        ## Printing a message
        print("Log-transformation process complete.")
        
        # Analysis
        print("––––––––CONDUCTING ANALYSIS––––––––")

        ## Storing and creating the output folder for the model-related results
        output_folder = config["output"]["model_folder"]
        setup.create_output_folder(output_folder)

        ## Printing a message about starting full model analysis
        print("––––– Full Model –––––")

        ## Fitting the full model on the training set
        outcome_var = "hypoxemia"
        predictor_var_list = ["age", "sex", "bmi", "co2", "glucose", "hematocrit",
                              "hemoglobin", "leukocytes", "potassium", "sodium"]

        full_model = analysis.fit_model(training_set, outcome_var, predictor_var_list)

        ## Storing the significant predictors at the 0.05 level of significance
        alpha = 0.05
        significant_predictors = analysis.get_significant_predictors(full_model, alpha)

        ## Printing a message to show the significant predictors if any exist
        if len(significant_predictors) > 0:
            print(
                f"The significant predictors from the full model "
                f"are: {significant_predictors}")
        else:
            print("No significant predictors found.")

        ## Saving the results of the full model
        output_file_name = "full_model_results"
    
        analysis.save_full_model(full_model, output_folder, output_file_name)

        ## Evaluating the full model on the test set and saving the test set
        ## misclassification error rate for comparison
        output_file_name = "full_model_misclassification_error_rate"

        threshold = 0.5
        analysis.evaluate_model(full_model, test_set, outcome_var,
                                output_folder, output_file_name, threshold)

        ## Printing a message about starting reduced model analysis
        print("––––– Reduced Model –––––")

        ## Fitting the reduced model on the training set using only the
        ## identified significant predictors
        reduced_model = analysis.fit_model(training_set, outcome_var, significant_predictors)

        ## Evaluating the reduced model on the test set and saving the test set
        ## misclassification error rate
        output_file_name = "reduced_model_misclassification_error_rate"

        analysis.evaluate_model(reduced_model, test_set, outcome_var,
                                output_folder, output_file_name)

        ## Printing a message
        print("Analysis complete.")
        
        # Printing a message
        print("––––––––PIPELINE COMPLETE––––––––")
    except ModuleNotFoundError as e:
        print(f"Module not found: {e}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()