# Function
def main():
    
    # Running the pipeline if the required files and .py scripts are present
    try:
        # Importing other .py scripts
        import cleaning
        import feature_engineering
        import data_restructuring
        import data_validation
        import cohort_split
        import create_plots
        import log_transform
        import analysis
        
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
        create_plots.main()
        
        # Log-transforming variables
        print("––––––––LOG-TRANSFORMING VARIABLES––––––––")
        log_transform.main()
        
        # Analysis
        print("––––––––CONDUCTING ANALYSIS––––––––")
        analysis.main()
        
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