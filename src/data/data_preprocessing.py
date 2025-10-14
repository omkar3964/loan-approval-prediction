import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

from src.logger import get_logger



logger = get_logger('data_preprocessing')
RAW_DATA_PATH='./data/raw/raw_data.csv'
PROCESSED_DATA_DIR='./data/processed'


def cap_outliers(df, columns):
    """Cap outliers using IQR method."""
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower, lower, np.where(df[col] > upper, upper, df[col]))
    return df


def raw_preprocessing(data:pd.DataFrame)-> pd.DataFrame:

    data.fillna(data.median(numeric_only=True), inplace=True)
    data.fillna("Unknown", inplace=True)
    logger.info("Missing values handled")

    # Fix negative values in specific columns
    num_cols_to_clip = ['residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'    ]
    for col in num_cols_to_clip:
        if col in data.columns:
            data[col] = data[col].clip(lower=0)
    logger.info("Negative values fixed in asset columns")

    # Cap outliers using IQR method
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns
    data = cap_outliers(data, num_cols)
    logger.info("Outliers capped using IQR")

    # Label encode categorical columns
    cat_cols = ['education', 'self_employed', 'loan_status']
    le = LabelEncoder()
    for col in cat_cols:
        if col in data.columns:
            data[col] = le.fit_transform(data[col])
    logger.info(" Categorical features encoded")

    return data



def main() -> None:
    """Main function for data Preprocessing pipeline."""
    try:
        logger.info("\n --------------------------------------------* Starting data processing process... *-----------------------------------------------------\n")
        # Fetch the data from data/raw

        logger.info('loading data from data/raw/raw_data.csv for processing....')
        raw_data = pd.read_csv(RAW_DATA_PATH)

        # Strip whitespace from column names
        raw_data.columns = raw_data.columns.str.strip()
        logger.info("Column names cleaned")

        process_data = raw_preprocessing(raw_data)

        # saving data
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

        processed_data_path = os.path.join(PROCESSED_DATA_DIR, 'processed_data.csv')
        process_data.to_csv(processed_data_path, index=False)
        logger.info('saved processed  data into data/processed/processed_data.csv.')



        logger.info("\n --------------------------------------------* completed data processing process. *-----------------------------------------------------\n")
    except Exception as e:
        logger.error(f"Failed to complete the data preprocessing process: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
