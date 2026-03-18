# Libraries
import pandas as pd
import statsmodels.formula.api as smf
import yaml
import os

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

def load_data(training_set_file_path, test_set_file_path):
    """
    Loads the training and test sets.
    """

    # Loading the training set
    try:
        with open(training_set_file_path, "r") as f:
            training_set = pd.read_csv(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Training set not found: {training_set_file_path}")

    # Loading the test set
    try:
        with open(test_set_file_path, "r") as f:
            test_set = pd.read_csv(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Test set not found: {test_set_file_path}")

    return training_set, test_set

def fit_model(training_set, outcome_var, predictor_var_list):
    """
    Fits a logistic regression model on a training set.
    """

    # Catching the case if there are no predictors in the list
    if len(predictor_var_list) == 0:
        print("Cannot fit model due to a lack of predictors.")
        return None

    # Concatenating main effects using "+"
    formula_parts = " + ".join(predictor_var_list)
    
    # Storing the model formula
    model_formula = f"{outcome_var} ~ {formula_parts}"

    # Printing the model formula
    print(f"Fitting model: {model_formula}")

    # Storing the fitted model object while hiding output
    try:
        model = smf.logit(formula = model_formula, data = training_set).fit(disp = 0)
        return model
    except Exception as e:
        print(f"Error fitting model: {e}")
        return None

def create_output_folder(output_folder):
    """
    Creates output folder for model results.
    """

    try:
        # Create the folder and do not give an error if the folder already exists
        os.makedirs(output_folder, exist_ok = True)
        print(f"Folder created: {output_folder}")
    except OSError as e:
        print(f"Error trying to create folder {output_folder}: {e}")

def save_full_model(model, output_folder, output_file_name):
    """
    Saves the estimated coefficients and Wald test p-values from the full model
    in .csv format.
    output_file_name should not contain the .csv extension (e.g., "model_results").
    """

    # Preparing the dataframe
    df = pd.DataFrame({
        "Variable": model.params.index,
        "Coefficient Estimate": model.params.values,
        "p-value": model.pvalues.values
    })

    # Saving the model results
    try:
        df.to_csv(f"{output_folder}/{output_file_name}.csv", index = False)
        print(f"Full model results saved to: {output_folder}/{output_file_name}.csv")
    except FileNotFoundError:
        raise FileNotFoundError(f"Folder not found: {output_folder}")

def get_significant_predictors(model, alpha = 0.05):
    """
    Returns the predictors that have a p-value less than alpha for the first
    research question.
    """

    # Storing the p-values from the model
    p_values = model.pvalues

    # Remove the intercept p-value
    p_values = p_values.drop("Intercept", errors = "ignore")

    # Storing the significant predictors
    significant_predictors = p_values[p_values < alpha].index.tolist()

    return significant_predictors

def evaluate_model(model, df, outcome_var, output_folder, output_file_name):
    """
    Calculates the test set misclassification error rate for a model
    and saves it in .txt format.
    output_file_name should not contain the .txt extension
    (e.g., "misclassification_error_rate").
    """

    # Catching the case if the model could not be fitted previously due to
    # a lack of predictors
    if model == None:
        print("Cannot evaluate model on the test set due to a lack of predictors.")
        return None

    # Obtaining the predicted outcome values from the model using a
    # 0.5 threshold
    predicted_values = (model.predict(df) > 0.5).astype(int)

    # Obtaining the actual values from the test set
    actual_values = df[outcome_var]

    # Calculating the test set misclassification error rate
    rate = (predicted_values != actual_values).mean()

    # Saving the test set misclassification error rate
    try:
        with open(f"{output_folder}/{output_file_name}.txt", "w") as f:
            f.write(
                f"The test set misclassification error rate is: {rate}"
            )
            print(
                f"Test set misclassification error rate "
                f"saved to: {output_folder}/{output_file_name}.txt"
            )
    except FileNotFoundError:
        raise FileNotFoundError(f"Folder not found: {output_folder}")

def main():

    # Loading the configuration file, training set, and test set
    config = load_yaml()
    training_set_path = config["data"]["training_set_path"]
    test_set_path = config["data"]["test_set_path"]
    training_set, test_set = load_data(training_set_path, test_set_path)

    # Storing and creating the output folder for the model-related results
    output_folder = config["output"]["model_folder"]
    create_output_folder(output_folder)

    # Printing a message about starting full model analysis
    print("––––– Full Model –––––")

    # Fitting the full model on the training set
    outcome_var = "hypoxemia"
    predictor_var_list = ["age", "sex", "bmi", "co2", "glucose", "hematocrit",
                          "hemoglobin", "leukocytes", "potassium", "sodium"]

    full_model = fit_model(training_set, outcome_var, predictor_var_list)

    # Storing the significant predictors at the 0.05 level of significance
    alpha = 0.05
    significant_predictors = get_significant_predictors(full_model, alpha)

    # Printing a message to show the significant predictors if any exist
    if len(significant_predictors) > 0:
        print(
            f"The significant predictors from the full model "
            f"are: {significant_predictors}")
    else:
        print("No significant predictors found.")

    # Saving the results of the full model
    output_file_name = "full_model_results"
    
    save_full_model(full_model, output_folder, output_file_name)

    # Evaluating the full model on the test set and saving the test set
    # misclassification error rate for comparison
    output_file_name = "full_model_misclassification_error_rate"

    evaluate_model(full_model, test_set, outcome_var,
                   output_folder, output_file_name)

    # Printing a message about starting reduced model analysis
    print("––––– Reduced Model –––––")

    # Fitting the reduced model on the training set using only the
    # identified significant predictors
    reduced_model = fit_model(training_set, outcome_var, significant_predictors)

    # Evaluating the reduced model on the test set and saving the test set
    # misclassification error rate
    output_file_name = "reduced_model_misclassification_error_rate"

    evaluate_model(reduced_model, test_set, outcome_var,
                   output_folder, output_file_name)

    # Printing a message
    print("Analysis complete.")

if __name__ == "__main__":
    main()