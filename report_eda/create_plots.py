# Libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Function
def create_plots(df: pd.DataFrame,
                 features_dict: dict,
                 outcome_list: list,
                 output_folder: str):
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
                          "xangle": ... (number of degrees to rotate x-axis labels by;
                          only if categorical),
                          "class_names": {0: "...", 1: "..."} (only if categorical)}
    outcome_list has the following format:
    ["variable_name", "class_0_name", "class_1_name"]
    """

    # Storing outcome variable information
    outcome_variable_name = outcome_list[0]

    ## Checking for presence of outcome variable in dataset
    if outcome_variable_name not in df.columns:
        print(f"Skipping: Outcome variable {outcome_variable_name} not found in data frame.")
        return
    
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
        # Skipping this feature if it is not in the data frame
        if var not in df.columns:
            print(f"Skipping {var}: Variable not found in data frame.")
            continue
        
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

            if info["units"] != "":
                ylabel = name + " (" + units + ")"
            elif info["units"] == "":
                ylabel = name

            plt.ylabel(ylabel)
        elif info["type"] == "categorical":
            # Creating a dataframe with the proportion of each category with
            # columns for outcome_variable_name, class 0, and class 1
            prop_df = (df.groupby(outcome_variable_name)[var]
                      .value_counts(normalize = True)
                      .unstack())

            # Creating a bar chart if the feature is categorical
            prop_df.plot(kind = "bar",
                         stacked = False,
                         color = info["bar_colours"])
            plt.title(title)
            plt.xlabel(xlabel)

            # Setting up the y-axis label depending on lowercase appropriateness
            ylabel = f"Proportion of {title_var}"
            plt.ylabel(ylabel)

            # Setting up the legend labels
            class_0_label = info["class_names"][0]
            class_1_label = info["class_names"][1]
            legend_title = info["name"]

            # Adding a legend to the plot
            plt.legend([class_0_label, class_1_label], title = legend_title)

            # Rotating labels on the x-axis
            plt.xticks(rotation = info["xangle"])

        # Ensuring plot labels and elements do not overlap
        plt.tight_layout()

        # Saving the plot
        try:
            plt.savefig(f"{output_folder}/{var}_comparison.png",
                        bbox_inches = "tight")
            print(f"Plot saved to: {output_folder}/{var}_comparison.png")
        except FileNotFoundError:
            raise FileNotFoundError(f"Folder not found: {output_folder}")

        plt.clf()