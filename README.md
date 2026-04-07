# Investigating Postoperative Hypoxemia Using the Medical Informatics Operating Room Vitals and Events Repository (MOVER)

This repository contains the files needed to execute the pipeline for this analysis as well as the project proposal, project management plan, and poster.

## Steps to Run the Pipeline

To run the pipeline, follow these steps:

1. Install and initialize Miniconda3 if not already done.
2. Install Git if not already done.
3. Create and activate a virtual environment if not already done. On macOS, this can be done in Terminal by navigating to the `miniconda3` directory and executing these two lines:

- `conda create -y -n MOVER python=3.10`


- `conda activate MOVER`

4. Install the packages in `requirements.txt`. On macOS, this can be done in Terminal immediately after executing the lines in step 3 by executing this line:

- `pip install -r requirements.txt`

5. Clone this repository. On macOS, this can be done in Terminal by navigating to a particular folder of your choosing and executing these two lines:

- `git clone https://github.com/MOVER-analysis/MOVER-analysis.git`


- `cd MOVER-analysis`

6. Create a folder in the same directory as `main.py` named `raw`.
7. Inside the `raw` folder, ensure the following three MOVER EPIC data tables from 2017-2022 are present:

- `patient_information.csv`


- `patient_labs.csv`


- `patient_post_op_complications.csv`

8. Execute `main.py`. On macOS, this can be done in Terminal immediately after executing the lines in step 5 by executing this line:

- `python main.py`

After `main.py` finishes running and `"––––––––PIPELINE COMPLETE––––––––"` is displayed, the following folders should be visible in the same directory as `main.py`:

1. `data`, containing the final dataset (`final_data.csv`), the training set (`hypoxemia_train.csv`), and the test set (`hypoxemia_test.csv`)
2. `model`, containing the results from the full model (`full_model_results.csv`), the misclassification error rate on the test set for the full model (`full_model_misclassification_error_rate.txt`), and the misclassification error rate on the test set for the reduced model (`reduced_model_misclassification_error_rate.txt`)
3. `plots`, containing `.png` files comparing the distributions of variables by outcome group
4. `validation`, containing data validation-related images and tables
