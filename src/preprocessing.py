# src/preprocessing.py

import pandas as pd
import numpy as np

def preprocess_data(file_path):
    """
    This function loads the raw diabetic data, performs the necessary
    preprocessing steps identified in our EDA, and returns a clean
    dataframe ready for modeling.

    Args:
        file_path (str): The path to the raw CSV file.

    Returns:
        pandas.DataFrame: The cleaned and preprocessed dataframe.
    """
    print("Starting preprocessing...")

    # Load the dataset
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows from {file_path}")

    # --- Initial Cleaning ---
    # Replace '?' with NaN for consistent missing value handling
    df.replace('?', np.nan, inplace=True)

    # --- Handling Specific Columns based on the research paper and EDA ---

    # The paper excluded encounters that ended in death or hospice.
    # We will do the same as these patients cannot be readmitted.
    # 'discharge_disposition_id' codes 11, 13, 14, 19, 20, 21 are related to death or hospice.
    df = df[~df['discharge_disposition_id'].isin([11, 13, 14, 19, 20, 21])]
    print(f"Removed encounters ending in death/hospice. {len(df)} rows remaining.")

    # The paper also states they used only the first encounter for each patient.
    # We will keep the first encounter to ensure independence of observations.
    df.drop_duplicates(subset=['patient_nbr'], keep='first', inplace=True)
    print(f"Removed duplicate patient encounters. {len(df)} rows remaining.")

    # Drop columns with excessive missing values or that are not useful for prediction,
    # as identified in the paper and our EDA.
    # 'weight' and 'payer_code' are too sparse. 'medical_specialty' is also sparse but we might handle it differently.
    # For now, let's drop them as a baseline. 'encounter_id' and 'patient_nbr' are just identifiers.
    cols_to_drop = ['weight', 'payer_code', 'encounter_id', 'patient_nbr']
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"Dropped columns: {cols_to_drop}")

    # --- Feature Engineering: Creating the Target Variable ---
    # Create our binary target: 1 if readmitted in <30 days, 0 otherwise.
    df['target'] = np.where(df['readmitted'] == '<30', 1, 0)
    # Now we can drop the original 'readmitted' column
    df.drop(columns=['readmitted'], inplace=True)
    print("Created binary target variable 'target'.")

    # For the remaining columns with missing values, we'll implement a simple imputation strategy later.
    # For now, our baseline script is complete.

    print("Preprocessing complete.")
    return df



# This block allows us to run the script directly from the command line
if __name__ == "__main__":
    # Define the input and output paths
    raw_data_path = '../data/diabetic_data.csv'
    processed_data_path = '../data/processed_diabetic_data.csv'

    # Run the preprocessing function
    processed_df = preprocess_data(raw_data_path)

    # Save the processed data to a new CSV file
    processed_df.to_csv(processed_data_path, index=False)
    print(f"Processed data saved to {processed_data_path}")