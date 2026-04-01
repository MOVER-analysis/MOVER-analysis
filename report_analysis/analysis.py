# Libraries
import pandas as pd
import statsmodels.formula.api as smf

# Functions
def fit_model(training_set: pd.DataFrame,
              outcome_var: str,
              predictor_var_list: list[str]):
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

def save_full_model(model: statsmodels.discrete.discrete_model.BinaryResults,
                    output_folder: str,
                    output_file_name: str):
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

def get_significant_predictors(model: statsmodels.discrete.discrete_model.BinaryResults,
                               alpha: float = 0.05) -> list[str]:
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

def evaluate_model(model: statsmodels.discrete.discrete_model.BinaryResults,
                   df: pd.DataFrame,
                   outcome_var: str,
                   output_folder: str,
                   output_file_name: str,
                   threshold: float = 0.5):
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

    # Obtaining the predicted outcome values from the model using
    # the threshold (0.5 by default)
    predicted_values = (model.predict(df) > threshold).astype(int)

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