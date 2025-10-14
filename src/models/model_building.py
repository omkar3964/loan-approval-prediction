import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
import joblib
import os


from src.logger import get_logger

logger = get_logger('model_building')

DATA_PATH='./data/processed/train.csv'
MODEL_PATH = './models/random_forest_model.pkl'
PARAMS_PATH='params.yaml'



def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.info('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
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

def train_model(X_train: np.ndarray, y_train: np.ndarray, model_params) -> RandomForestClassifier:
    """Train the RandomForest model."""
    try:
        logger.info('Random Forest model training started....')
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)
        logger.info('Random Forest model training completed')
        return model
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise

def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(model, file_path)
        logger.info(f"Trained Model saved successfully at {file_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving the model: {e}")
        raise


def main() -> None:
    """Main function for model building pipeline."""
    try:
        logger.info("\n --------------------------------------------* starting model building process. *-----------------------------------------------------\n")
        # load data
        train_data = load_data(DATA_PATH)
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        # load trainning parameter
        params = load_params(PARAMS_PATH)
        model_params = params['model_building']

        # training model
        random_forest_model = train_model(X_train, y_train, model_params)

        # saving model
        save_model(random_forest_model, MODEL_PATH)

    except Exception as e:
        logger.error(f"Failed to complete the model building process: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()