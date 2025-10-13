import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from src.logger import get_logger
from src.connections.s3_connection import S3Operations

# Initialize logger
logger = get_logger('data_ingestion.py')

# Load environment variables
load_dotenv()
BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'default-bucket')
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
REGION_NAME = os.getenv('region', 'us-east-1')
FILE_NAME = os.getenv('file_name', 'loan_approval_dataset.csv')
RAW_DATA_DIR = Path(os.getenv('RAW_DATA_DIR', './data/raw'))


def save_data(raw_data: pd.DataFrame, data_path: Path) -> None:
    """
    Save raw data to local CSV.

    Args:
        raw_data (pd.DataFrame): DataFrame to save.
        data_path (Path): Directory path to save the CSV file.
    """
    if raw_data is None or raw_data.empty:
        logger.warning("No data to save.")
        return

    try:
        data_path.mkdir(parents=True, exist_ok=True)
        file_path = data_path / 'raw_data.csv'
        raw_data.to_csv(file_path, index=False)
        logger.info(f"Data saved successfully at {file_path}")
    except Exception as e:
        logger.error(f"Unexpected error occurred while saving data: {e}")
        raise


def main() -> None:
    """Main function for data ingestion pipeline."""
    try:

        logger.info("\n --------------------------------------------* Starting data ingestion process... *-----------------------------------------------------\n")

        # Initialize S3 client
        s3_client = S3Operations(BUCKET_NAME, AWS_ACCESS_KEY, AWS_SECRET_KEY, region_name=REGION_NAME)
        df = s3_client.fetch_csv(FILE_NAME)

        # Save raw data locally
        save_data(df, RAW_DATA_DIR)

        logger.info("\n --------------------------------------------* completed data ingestion process... *-----------------------------------------------------\n")

    except Exception as e:
        logger.error(f"Failed to complete the data ingestion process: {e}")
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
