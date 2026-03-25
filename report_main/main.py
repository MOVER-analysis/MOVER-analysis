# Function
def main():
    
    # Running the pipeline if the required files and .py scripts are present
    try:
        # Importing other .py scripts
        import setup
        import cleaning
        import feature_engineering
        import data_restructuring
        import data_validation
        import cohort_split
        import create_plots
        import log_transform
        import analysis

        # Loading files
        config = setup.load_yaml()
        
        training_path = config["data"]["training_set_path"]
        training_set = setup.load_data(training_path)

        test_path = config["data"]["test_set_path"]
        test_set = setup.load_data(test_path)
        
        # Printing a start message
        print("––––––––STARTING PIPELINE––––––––")
        
        # Cleaning
        print("––––––––CLEANING DATA––––––––")
        cleaning.main()
        
        # Feature engineering
        print("––––––––PERFORMING FEATURE ENGINEERING––––––––")
        feature_engineering.main()
        
        # Data restructuring
        print("––––––––RESTRUCTURING DATA––––––––")
        data_restructuring.main()
        
        # Data validation
        print("––––––––VALIDATING DATA––––––––")
        data_validation.main()
        
        # Cohort construction and train-test split
        print("––––––––COHORT CONSTRUCTION AND TRAIN-TEST SPLIT––––––––")
        cohort_split.main()
        
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

        analysis.evaluate_model(full_model, test_set, outcome_var,
                                output_folder, output_file_name)

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
        print("––––––––DATA WRANGLING COMPLETE––––––––")
    except ModuleNotFoundError as e:
        print(f"Script not found: {e.name}")
    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()