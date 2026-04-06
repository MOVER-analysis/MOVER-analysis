def main():
    # Feature engineering
    print("––––––––FEATURE ENGINEERING––––––––")
    
    input_path = OUTPUT_DATA_PATH + "cleaned_data.csv"
    output_path = OUTPUT_DATA_PATH + "feature_engineered_data.csv"

    # --- read data ---
    df = setup.load_data(input_path)

    ## Defining constants
    outcome = "hypoxemia"
    height_unit = "m"
    weight_unit = "kg"
    bmi = "bmi"
    
    ## config for creating binary indicator columns
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
        "target_value": "female",
        "match_type": "exact"
        }
    ]
    df = feature_engineering.create_binary_cols(df, BINARY_COL_CONFIG)
    df = df.drop(columns=[complications])
    df = df.drop_duplicates()

    ## config for calculating BMI from height and weight
    BMI_CONFIG = {
        "height_col": (height, height_unit),
        "weight_col": (weight, weight_unit),
        "new_col": bmi
    }
    df = feature_engineering.calculate_bmi(df, BMI_CONFIG)

    # --- export to csv ---
    df.to_csv(output_path, index = False)
    print(f"Feature-engineered data saved to {output_file}.")

    # Data restructuring
    print("––––––––DATA RESTRUCTURING––––––––")

    input_path = OUTPUT_DATA_PATH + "feature_engineered_data.csv"
    output_path = OUTPUT_DATA_PATH + "restructured_data.csv"

    # --- read data ---
    df = setup.load_data(input_path)

    ## Defining constants
    drop_cols = [lab_code, unit, labs_ts]
    
    outcome_cols = [outcome]
    demo_cols = [age, sex, bmi, height, weight]

    # drop long-format columns before pivoting
    ## columns to drop before reshaping data
    df = df.drop(columns=drop_cols, errors="ignore")

    # reshape the data wider
    df_wide = data_restructuring.pivot_wider(df, id_cols, lab_name, value)

    # get the names of the new wide-format lab columns
    lab_cols = df[lab_name].dropna().unique().tolist()

    # keep only required columns that exist in the wide dataframe
    keep_cols = id_cols + outcome_cols + lab_cols + demo_cols
    keep_cols = [col for col in keep_cols if col in df_wide.columns]
    print("Final columns selected successfully.")

    df_restructured = df_wide[keep_cols]

    # --- export to csv ---
    df_restructured.to_csv(output_path, index = False)
    print(f"Restructured data saved to {output_file}.")

    # Data validation
    print("––––––––DATA VALIDATION––––––––")
    
    input_path = OUTPUT_DATA_PATH + "restructured_data.csv"
    output_data_path = OUTPUT_DATA_PATH + "validated_data.csv"
    output_validation_path = "validation/"
    
    # --- read data ---
    df = setup.load_data(input_path)

    ## Defining constants
    drop_threshold = 0.5

    ## required columns
    required_cols = id_cols + outcome_cols + lab_cols + demo_cols

    ## check duplicate encounter keys
    data_validation.check_duplicate_keys(df, id_cols)

    ## verify required columns are present
    data_validation.check_required_columns(df, required_cols)

    ## check expected data types
    ### expected column groups by type
    expected_type_cols = {
        "string": id_cols, 
        "numeric": lab_cols + [col for col in demo_cols if col != sex], 
        "binary": outcome_cols + [sex]
    }
    data_validation.check_expected_column_types(df, expected_type_cols)

    ## screen for implausible values
    data_validation.check_implausible_values(df, expected_type_cols)

    ## miss value checks
    ## threshold for dropping columns based on missingness proportion
    df_validated = data_validation.check_missingness(df, drop_threshold = drop_threshold)

    # --- export to csv ---
    df_validated.to_csv(output_data_path, index=False)
    print(f"Validated data saved to {output_data_file}.")

    # Cohort split
    print("––––––––COHORT SPLIT––––––––")

    input_path = OUTPUT_DATA_PATH + "validated_data.csv"
    train_path = OUTPUT_DATA_PATH + "hypoxemia_train.csv"
    test_path = OUTPUT_DATA_PATH + "hypoxemia_test.csv"
    
    # --- read data ---
    df = setup.load_data(input_path)

    ## Defining constants for case-control sampling
    ### seed, sample size and test set proportion for random sampling
    random_state = 42
    neg_sample_size = 200
    test_prop = 0.25
    
    sample = cohort_split.case_control_sample(df, outcome, 1, neg_sample_size)
    print(f"Sampled dataset shape: {sample.shape}")

    ## stratified train/test split
    train_df, test_df = train_test_split(
        sample,
        test_size = test_prop,
        stratify = sample[outcome],
        random_state = random_state
    )

    # reset index
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # print outcome counts
    train_counts = train_df[outcome].value_counts()
    test_counts = test_df[outcome].value_counts()
    
    split_count = pd.DataFrame({
        "train": train_counts,
        "test": test_counts
    }).T.fillna(0).astype(int)
    print("\nOutcome counts (train/test):")
    print(split_count)

    # --- export to csv ---
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nSaved train data to {train_path}")
    print(f"Saved test data to {test_path}")

    

