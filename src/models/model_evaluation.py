import json
import os
from dotenv import load_dotenv

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from src.logger import get_logger
load_dotenv()

TESTING_DATA_PATH="./data/processed/test.csv"
MODEL_PATH='./models/random_forest_model.pkl'
METRICS_PATH='./reports/metrics.json'

# Set up DagsHub credentials for MLflow tracking only
dagshub_token = os.getenv("DAGSHUB_ACCESS_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = os.getenv('DAGSHUB_REPO_OWNER')
repo_name = os.getenv('DAGSHUB_REPO_NAME')

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')



logger = get_logger("model_evaluation")

def load_data(file_path:str)-> pd.DataFrame:
    """load data from a csv file"""
    try:
        df = pd.read_csv(file_path)
        logger.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def load_model(file_path: str):
    """Load the trained model from a file using joblib."""
    try:
        model = joblib.load(file_path)
        logger.info('Model loaded successfully from %s', file_path)
        return model
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model: %s', e)
        raise

def evaluate_model(rf_model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = rf_model.predict(X_test)
        y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logger.info('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.info('Metrics saved to %s', file_path)
        logger.info(f"Model Evaluation Results:\n"
                    f"Accuracy: {metrics['accuracy']:.4f}, "
                    f"Precision: {metrics['precision']:.4f}, "
                    f"Recall: {metrics['recall']:.4f}, "
                    f"AUC: {metrics['auc']:.4f}")

    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise

def main() -> None:
    """Main function for model evaluation pipeline."""
    try:
        logger.info("\n --------------------------------------------* starting model evaluation process. *-----------------------------------------------------\n")

        # load data
        testing_data = load_data(TESTING_DATA_PATH)
        X_test = testing_data.iloc[:, :-1].values
        y_test = testing_data.iloc[:, -1].values

        # load model
        rf_model = load_model(MODEL_PATH)

        # evaluate metrics and saved
        metrics = evaluate_model(rf_model, X_test, y_test)
        save_metrics(metrics, METRICS_PATH)


        mlflow.set_experiment("my-dvc-pipeline")
        with mlflow.start_run() as run:

            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log model parameters to MLflow
            if hasattr(rf_model, 'get_params'):
                params = rf_model.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)

            # Log model to MLflow
            mlflow.sklearn.log_model(rf_model, "model")

            # Save model info
            save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')

            # Log the metrics file to MLflow
            mlflow.log_artifact('reports/metrics.json')

        logger.info("\n --------------------------------------------* completed model evaluation process. *-----------------------------------------------------\n")

    except Exception as e:
        logger.error(f"Failed to complete the model building process: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()