import sys
import os
import pandas as pd
import yaml
import logging
import sys
import os


from src.logger_function.logger import logger
from sklearn.preprocessing import StandardScaler, LabelEncoder



# Loading the parameters from param.yaml file
try:
    with open("params.yaml") as file:
        params = yaml.safe_load(file)['preprocess']
except Exception as e:
    logger.info(f"Error loading parameters from params.yaml: {e}")
    sys.exit(1)

def preprocess(input_path, output_path):
    try:
        # Step 1: Load the data
        logger.info(f"Loading data from {input_path}")
        data = pd.read_csv(input_path)

        # Step 2: Remove duplicate rows
        logger.info("Removing duplicate rows")
        data = data.drop_duplicates()

        # Step 3: Handle missing values
        logger.info("Handling missing values")
        data.fillna(method='ffill', inplace=True)

        # Step 4: Encoding categorical variables
        logger.info("Encoding categorical variables")
        for col in data.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])

        # Step 5: Feature scaling for numerical values
        logger.info("Scaling numerical features")
        scaler = StandardScaler()

        # Select numeric columns excluding 'Outcome'
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        logging.info("Excluding the outcome(Target) feature from scaling")
        numeric_cols.remove('Outcome')  # Remove the 'Outcome' column from the list
        
        # Scale the remaining numeric columns
        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

        # Step 6: Save the preprocessed data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        data.to_csv(output_path, index=False)
        logger.info(f"Preprocessed data saved to {output_path}")

    except Exception as e:
        logger.error(f"An error occurred during preprocessing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    preprocess(params["input"], params["output"])
