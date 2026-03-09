# functions to write:
# 1. removing invalid LOG_ID
# 2. ffill for height and weight (partitioning by MRN, sort by date)
# 3. regression imputation

# === Import libraries === 
import pandas as pd
import numpy as np

# for plotting missingness pattern
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns

# for regression imputation
import statsmodels.formula.api as smf

# === Global Constants === 
RAW_DATA_PATH = "raw/"

# === Functions for Data Cleaning ===

# === Main ===

def main():
    # --- Define file paths ---
    input_files = ["patient_information.csv", "patient_post_op_complications.csv", "patient_labs.csv"]
    raw_files = {} # store .csv files into a dictionary
    output_file = "cleaned_data.csv"

    print("--- Loading data --- ")

    # --- Load raw data files ---
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
    # Raise an error if file not found
    except FileNotFoundError as e:
        print(f"\n[Error] A required file is missing: ")
        print(f"{e}")
        return
    # Raise an error if an unexpected error occurred
    except Exception as e:
        print(f"\n[Error] An unexpected error occurred while reading files: ")
        print(f"{e}")
        return

    # --- Extract dataframes ---
    postop_raw = raw_files["patient_post_op_complications"]
    info_raw = raw_files["patient_information"]
    labs_raw = raw_files["patient_labs"]

    # --- Select relevant columns ---
    
    
    print("\n All data ready for processing")
    


if __name__ == "__main__":
    main()
            
    
    