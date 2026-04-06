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

