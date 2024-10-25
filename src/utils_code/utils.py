import sys
import os
import pandas as pd
import yaml
import logging


from logger_function.logger import setup_logging
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Set up logging
setup_logging()

# Loading the parameters from param.yaml file
try:
    with open("params.yaml") as file:
        params = yaml.safe_load(file)['preprocess']
except Exception as e:
    logging.error(f"Error loading parameters from params.yaml: {e}")
    sys.exit(1)

def preprocess(input_path, output_path):
    try:
        # Step 1: Load the data
        logging.info(f"Loading data from {input_path}")
        data = pd.read_csv(input_path)

        # Step 2: Remove duplicate rows
        logging.info("Removing duplicate rows")
        data = data.drop_duplicates()

        # Step 3: Handle missing values
        logging.info("Handling missing values")
        data.fillna(method='ffill', inplace=True)

        # Step 4: Encoding categorical variables
        logging.info("Encoding categorical variables")
        for col in data.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])

        # Step 5: Feature scaling for numerical values
        logging.info("Scaling numerical features")
        scaler = StandardScaler()
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

        # Step 6: Save the preprocessed data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data.to_csv(output_path, header=None, index=False)
        logging.info(f"Preprocessed data saved to {output_path}")

    except Exception as e:
        logging.error(f"An error occurred during preprocessing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    preprocess(params["input"], params["output"])
