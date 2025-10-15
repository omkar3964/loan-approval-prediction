import json
import os
import mlflow

from src.logger import get_logger
from dotenv import load_dotenv
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")


logger = get_logger('register_model')
load_dotenv()

MODEL_INFO_PATH = './reports/experiment_info.json'
MODEL_NAME = "my_model"

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


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise


def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)

        # Transition the model to "Staging" stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        logger.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise



def main() -> None:
    """Main function for register model  pipeline."""
    try:
        logger.info("\n --------------------------------------------* starting register model process. *-----------------------------------------------------\n")
        # LOAD MODEL INFO
        model_info = load_model_info(MODEL_INFO_PATH)

        # REGISTER MODEL
        register_model(MODEL_NAME, model_info)

        logger.info("\n --------------------------------------------* completed register model process. *-----------------------------------------------------\n")

    except Exception as e:
        logger.error(f"Failed to complete the register model process: {e}")
        print(f"Error: {e}")


if __name__ == "__main__":
    main()