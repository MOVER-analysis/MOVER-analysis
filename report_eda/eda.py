# Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yaml

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
    Loads the .csv training set.
    """

    try:
        with open(file_path, "r") as f:
            data = pd.read_csv(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    return data

def create_plots(df, features_dict, outcome_list, output_folder):
    """
    Creates and saves plots.
    features_dict has the following format (only one feature is shown for brevity):
    {"feature_variable": {"type": "continuous" or "categorical",
                          "name": "..." (the common name without units if applicable,
                          e.g., "Sex" or "Age"),
                          "units": "..." (the units, e.g., "years", or an empty string
                          "" if no units),
                          "lowercase": True or False (whether converting the name
                          to lowercase is appropriate),
                          "bar_colours": ["colour_1", "colour_2"] (only if categorical),
                          "class_names": {0: "...", 1: "..."} (only if categorical)}
    outcome_list has the following format:
    ["variable_name", "class_0_name", "class_1_name"]
    """

    # Storing outcome variable information
    outcome_variable_name = outcome_list[0]
    outcome_0 = outcome_list[1]
    outcome_1 = outcome_list[2]

    # Updating the names of the outcome classes for the plots

    ## Making a copy of the dataset
    df = df.copy()

    ## Changing the outcome values in the dataset to be descriptive rather than
    ## numerical
    df[outcome_variable_name] = df[outcome_variable_name].map(
        {0: outcome_0, 1: outcome_1}
    )

    # Storing the x-axis label
    xlabel = "Group"

    # Looping through the desired features
    for var, info in features_dict.items():
        # Storing the plot title depending on the appropriateness of making the
        # variable name lowercase
        title_var = info["name"]

        if info["lowercase"] == True:
            # Storing the variable name in lowercase
            title_var = info["name"].lower()

        # Storing the plot title
        title = f"Comparison of {title_var} by outcome group"

        if info["type"] == "continuous":
            # Creating a boxplot if the feature is continuous
            sns.boxplot(x = outcome_variable_name,
                        y = var,
                        data = df)
            plt.title(title)
            plt.xlabel(xlabel)

            # Setting up the y-axis label
            name = info["name"]
            units = info["units"]
            ylabel = name + "( " + units + ")"

            plt.ylabel(ylabel)
        elif info["type"] == "categorical":
            # Creating a dataframe with the proportion of each category with
            # columns for outcome_variable_name, class 0, and class 1 (wide format)
            prop_df = (df.group_by(outcome_variable_name)[var]
                      .value_counts(normalize = True)
                      .unstack())

            # Creating a bar chart if the feature is categorical
            prop_df.plot(kind = "bar",
                         stacked = False,
                         color = info["bar_colours"])
            plt.title(title)
            plt.xlabel(xlabel)

            # Setting up the y-axis label
            name = info["name"]
            ylabel = f"Proportion of {name}"

            plt.ylabel(ylabel)

            # Setting up the legend labels
            class_0_label = info[class_names[0]]
            class_1_label = info[class_names[1]]
            legend_title = info["name"]

            # Adding a legend to the plot
            plt.legend([class_0_label, class_1_label], title = legend_title)

        # Ensuring plot labels and elements do not overlap
        plt.tight_layout()

        # Saving the plot
        try:
            plt.savefig(f"{output_folder}/{var}_comparison.png",
                        bbox_inches = "tight")
            print(f"Plot saved to: {output_folder}/{var}_comparison.png")
        except FileNotFoundError:
            raise FileNotFoundError(f"Folder not found: {output_folder}")

def main():

    # Loading the configuration file and the training set
    config = load_yaml()
    data = load_data(config["data"]["training_set_path"])

    # Storing the output folder for the plots
    output_folder = config["output"]["plots_folder"]

    # Setting up features_dict and outcome_list for plotting
    features_dict = {
        "age": {
            "type": "continuous",
            "name": "Age",
            "units": "years",
            "lowercase": True
        },
        "bmi": {
            "type": "continuous"
            "name": "Body mass index",
            "units": "kg/m^2",
            "lowercase": True
        },
        "crp": {
            "type": "continuous",
            "name": "C-reactive protein",
            "units": "MG/DL",
            "lowercase": False
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
            "type": "continuous":
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
        "pH": {
            "type": "continuous",
            "name": "pH",
            "units": "",
            "lowercase": False
        },
        "lactate": {
            "type": "continuous",
            "name": "Lactate",
            "units": "mmol/L",
            "lowercase": True
        },
        "sex": {
            "type": "categorical",
            "name": "Sex",
            "units": "",
            "lowercase": True,
            "bar_colours": ["blue", "red"],
            "class_names": {0: "Male", 1: "Female"}
        }
    }

    outcome_list = ["hypoxemia", "Non-Hypoxemia", "Hypoxemia"]

    # Creating and saving plots
    create_plots(data, features_dict, outcome_list, output_folder)

    # Printing a message
    print("Exploratory data analysis complete.")

if __name__ == "__main__":
    main()