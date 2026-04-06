# Libraries
import numpy as np
import pandas as pd

# Function
def log_transform(df_file_path_list: list[str],
                  var_list: list[str]):
    """
    Log-transforms skewed variable(s) and saves updated datasets in .csv format.
    df_file_path_list has format:
    ["file_path_1", "file_path_2", ...]
    var_list is a list of skewed numeric variable(s) identified visually from the plots
    produced using create_plots.py.

    The log-transformation is done by converting x to log(x + 1) to
    safely handle zero values. Here, log represents the natural logarithm.
    """

    # Creating a dictionary of datasets and their file paths
    df_dict = {}
    
    for file_path in df_file_path_list:
        try:
            with open(file_path, "r") as f:
                df = pd.read_csv(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        df_dict[file_path] = df

    # Looping through the dictionary
    for file_path, df in df_dict.items():
        # Log-transforming desired variables
        for var in var_list:
            # Checking that the variable exists in the dataset
            if var in df.columns:
                df[var] = np.log1p(df[var])
                print(f"Log-transformed {var}")
            # Printing a message if the variable does not exist in the dataset
            else:
                print(f"Variable {var} not present in {df_name}")

        # Saving the modified dataset
        try:
            df.to_csv(file_path, index = False)
            print(f"Updated dataset saved to: {file_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"File path not found: {file_path}")

    # Returning the updated datasets in a dictionary
    return df_dict