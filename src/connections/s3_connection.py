import boto3
import pandas as pd
from io import StringIO
from src.logger import get_logger  # ✅ Import your custom logger

# Initialize a module-specific logger
logger = get_logger(__name__)

class S3Operations:
    def __init__(self, bucket_name, aws_access_key, aws_secret_key, region_name="us-east-1"):
        """
        Initialize S3Operations with AWS credentials and bucket details.
        """
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region_name
        )
        logger.info(f"Initialized S3 client for bucket: {bucket_name}")

    def fetch_csv(self, file_key):
        """
        Fetch a CSV file from S3 and return as a pandas DataFrame.
        """
        try:
            logger.info(f"Fetching '{file_key}' from S3 bucket '{self.bucket_name}'...")
            obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
            df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
            logger.info(f"Successfully fetched '{file_key}' with {len(df)} records.")
            return df
        except Exception as e:
            logger.exception(f"Failed to fetch '{file_key}' from S3: {e}")
            return None
