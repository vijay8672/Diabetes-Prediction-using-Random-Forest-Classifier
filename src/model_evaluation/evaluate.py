import pandas as pd
import pickle

from src.logger_function.logger import logger
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow

from urllib.parse import urlparse

# Set up MLflow tracking URI and credentials
os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/vijay8672/MachineLearningPipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = 'vijay8672'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'b5e1ccfce9a243cd33750ce8339c7432d051c447'

params=yaml.safe_load(open("params.yaml"))['train']

def evaluate(data_path, model_path):
  data=pd.read_csv(data_path)
  X=data.drop(columns=["Outcome"])
  y=data["Outcome"]


  mlflow.set_tracking_uri('https://dagshub.com/vijay8672/MachineLearningPipeline.mlflow')

  logger.info("loading the model pickle file from the disk")

  model=pickle.load(open(model_path, 'rb'))


  prediction=model.predict(X)
  acc_score=accuracy_score(y,prediction)

  logger.info("Logging the metrics to MLflow")

  mlflow.log_metric("accuracy", acc_score)
  logger.info("Model Accuracy:", {acc_score})

if __name__=="__main__":
  evaluate(params['data'], params['model'])