# Libraries
import numpy as np
import pandas as pd

# Functions
def load_yaml(file_path = "params.yaml"):
    """
    Loads the .yaml configuration file.
    """

    try:
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")

def load_data(file_path):
    """
    Loads a .csv dataset.
    """

    try:
        with open(file_path, "r") as f:
            data = pd.read_csv(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    return data

def log_transform(df_file_path_list, var_list):
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
        df = load_data(file_path)
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

def main():

    # Loading the configuration file and the datasets
    config = load_yaml()
    training_path = config["data"]["training_set_path"]
    test_path = config["data"]["test_set_path"]

    # Setting up df_file_path_list and var_list for log-transformation
    df_file_path_list = [training_path, test_path]
    var_list = ["glucose"]
    
    # Log-transforming the desired variable(s) and saving the modified datasets
    log_transform(df_file_path_list, var_list)

    # Printing a message
    print("Log-transformation process complete.")

if __name__ == "__main__":
    main()