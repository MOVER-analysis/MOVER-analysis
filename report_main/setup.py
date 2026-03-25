# Libraries
import yaml
import pandas as pd
import os

# Functions
def load_yaml(file_path = "params.yaml"):
    """
    Loads a .yaml configuration file.
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

def create_output_folder(output_folder):
    """
    Creates an output folder.
    """

    try:
        # Create the folder and do not give an error if the folder already exists
        os.makedirs(output_folder, exist_ok = True)
        print(f"Folder created: {output_folder}")
    except OSError as e:
        print(f"Error trying to create folder {output_folder}: {e}")