import pandas as pd
from sklearn.model_selection import train_test_split

# === functions ===
def case_control_sample(df: pd.DataFrame, 
                        outcome: str, 
                        case_value: object, 
                        control_sample_size: int, 
                        random_state: int = 42) -> pd.DataFrame:
    """
    Keep all rows with outcome == case_value and randomly sample
    control_sample_size rows from all other outcome levels.
    """
    # keep rows with non-missing outcome only
    dat = df[df[outcome].notna()]

    # split into case and control groups
    case_df = dat[dat[outcome] == case_value]
    control_df = dat[dat[outcome] != case_value]

    # check sample size
    if len(control_df) < control_sample_size:
        raise ValueError(
            f"Requested control_sample_size ({control_sample_size}) exceeds the "
            f"available number of controls ({len(control_df)})."
        )

    # sample controls
    control_sample = control_df.sample(
        n=control_sample_size,
        random_state=random_state
    )

    # combine case group and control sample
    sample = pd.concat([case_df, control_sample], axis=0)

    return sample

# === main ===
def main():
    # --- define file paths ---
    ## path to data folder
    DATA_PATH = "data/"
    
    input_file = "validated_data.csv"
    output_train_file = "hypoxemia_train.csv"
    output_test_file = "hypoxemia_test.csv"

    input_path = DATA_PATH + input_file
    train_path = DATA_PATH + output_train_file
    test_path = DATA_PATH + output_test_file
    
    # --- read data ---
    df = load_data(input_path)

    # --- cohort split ---
    ## case-control sampling
    ### outcome variable name, seed, sample size and test set proportion for random sampling
    OUTCOME = "hypoxemia"
    RANDOM_STATE = 42
    NEG_SAMPLE_SIZE = 200
    TEST_PROP = 0.25
    
    sample = case_control_sample(df, OUTCOME, 1, NEG_SAMPLE_SIZE)
    print(f"Sampled dataset shape: {sample.shape}")

    ## stratified train/test split
    train_df, test_df = train_test_split(
        sample,
        test_size = TEST_PROP,
        stratify = sample[OUTCOME],
        random_state = RANDOM_STATE
    )

    # reset index
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # print outcome counts
    train_counts = train_df[OUTCOME].value_counts()
    test_counts = test_df[OUTCOME].value_counts()
    
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

if __name__ == "__main__":
    main()


