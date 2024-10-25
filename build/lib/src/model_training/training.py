import pandas as pd
import numpy as np
from src.logger_function.logger import logger
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
import os
import mlflow
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlflow.models import infer_signature
from urllib.parse import urlparse

# Set up MLflow tracking URI and credentials
os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/vijay8672/MachineLearningPipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = 'vijay8672'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'b5e1ccfce9a243cd33750ce8339c7432d051c447'

def hyperparameter_tuning(X_train, y_train, param_grid):
    """Perform hyperparameter tuning using GridSearchCV."""
    classifier = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search

logger.info("Loading parameters from params.yaml")

# Safely open the parameters YAML file
with open("params.yaml") as file:
    params = yaml.safe_load(file)['train']

def train(data_path, model_path):
    """Train a Random Forest model with hyperparameter tuning and logging."""
    # Load data
    data = pd.read_csv(data_path)
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']

    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    logger.info("Starting MLflow run")

    with mlflow.start_run():
        logger.info("Splitting the dataset into training and testing data")
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        # Infer model signature
        signature = infer_signature(X_train, y_train)

        logger.info("Defining the hyperparameter grid")

        # Define hyperparameter grid for tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

        logger.info("Performing hyperparameter tuning")
        grid_search = hyperparameter_tuning(X_train, y_train, param_grid)

        logger.info("Obtaining the best model")
        best_model = grid_search.best_estimator_

        logger.info("Predicting and evaluating the model")
        y_pred = best_model.predict(X_test)

        # Calculate accuracy score
        acc_score = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc_score:.4f}")

        logger.info("Logging metrics")
        mlflow.log_metric("accuracy", acc_score)

        # Log best parameters dynamically
        for param, value in grid_search.best_params_.items():
            mlflow.log_param(param, value)

        logger.info("Logging confusion matrix and classification report")
        cm = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        # Log confusion matrix as text
        cm_str = str(cm)
        mlflow.log_text(cm_str, "confusion_matrix.txt")

        # Log classification report as text
        mlflow.log_text(class_report, "classification_report.txt")

        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Log model with MLflow
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(best_model, "model", registered_model_name="Best_model")
        else:
            mlflow.sklearn.log_model(best_model, "model", signature=signature)

        logger.info("Creating the directory to save the model.pkl file")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Save the best model locally
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)

        print(f"Model saved to {model_path}")

if __name__ == '__main__':
    train(params['data'], params['model'])
