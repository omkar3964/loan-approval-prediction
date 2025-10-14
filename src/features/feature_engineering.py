import os

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from src.logger import get_logger

logger = get_logger("feature_engineering.py")
PROCESSED_DATA_PATH='./data/processed/processed_data.csv'
FEATURED_DATA_PATH='./data/processed'


def check_balance_dataset(X, y):
    """
    Check if the dataset is balanced; if not, apply SMOTE to balance it.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : pd.Series or np.ndarray
        Target variable.

    Returns
    -------
    X_balanced, y_balanced : np.ndarray
        Balanced dataset after applying SMOTE (if needed).
    """
    try:
        logger.info("Checking class distribution before balancing...")
        class_counts = pd.Series(y).value_counts()
        logger.info(f"Class distribution before balancing:\n{class_counts}")

        # Check imbalance threshold (e.g., difference > 20%)
        imbalance_ratio = class_counts.max() / class_counts.min()
        if imbalance_ratio <= 1.2:
            logger.info("Dataset is already balanced. No SMOTE applied.")
            return X, y

        logger.info("Dataset is imbalanced. Applying SMOTE...")

        # Scale features first
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

        logger.info(f"Applied SMOTE successfully.")
        logger.info(f"New class distribution:\n{pd.Series(y_balanced).value_counts()}")

        return X_balanced, y_balanced

    except Exception as e:
        logger.error(f"Error in balancing dataset: {e}")
        raise e

def feature_selection(X, y, importance_threshold=0.01):
    """
    Train a Random Forest to compute feature importance and drop low-importance features.
    """
    logger.info("Training Random Forest to compute feature importance...")
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)

    importances = rf.feature_importances_
    feature_names = X.columns
    feat_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    logger.info("Feature importance:\n%s", feat_importance.to_string())

    # Drop features below threshold
    low_importance_features = feat_importance[feat_importance < importance_threshold].index.tolist()
    if low_importance_features:
        logger.info("Dropping low-importance features: %s", low_importance_features)
        X_reduced = X.drop(columns=low_importance_features)
    else:
        logger.info("No features dropped. All above importance threshold.")
        X_reduced = X

    return X_reduced

def main() -> None:
    """Main function for data Preprocessing pipeline."""
    try:
        logger.info("\n --------------------------------------------* starting feature engineering process. *-----------------------------------------------------\n")
        data = pd.read_csv(PROCESSED_DATA_PATH)
        logger.info("Loaded processed data from %s", PROCESSED_DATA_PATH)

        # Remove the 'loan_id' column if it exists
        if 'loan_id' in data.columns:
            data.drop(columns=['loan_id'], inplace=True)
            logger.info("Removed 'loan_id' column from dataset.")

        # Split features and target
        X = data.drop(columns=['loan_status'])
        y = data['loan_status']

        # Check and balance dataset
        X_balanced, y_balanced = check_balance_dataset(X, y)

        # Feature selection using Random Forest importance
        X_final = feature_selection(pd.DataFrame(X_balanced, columns=X.columns), y_balanced)

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
        )
        logger.info("Split dataset into training and testing sets.")

        # Save train and test sets
        os.makedirs(FEATURED_DATA_PATH, exist_ok=True)
        train_path = os.path.join(FEATURED_DATA_PATH, "train.csv")
        test_path = os.path.join(FEATURED_DATA_PATH, "test.csv")


        train_df = X_train.copy()
        train_df['loan_status'] = y_train
        test_df = X_test.copy()
        test_df['loan_status'] = y_test

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info("Saved featured training data to %s", train_path)
        logger.info("Saved  featured testing data to %s", test_path)

        logger.info("\n --------------------------------------------* completed feature engineering process. *-----------------------------------------------------\n")
    except Exception as e:
        logger.error(f"Failed to complete the feature engineering process: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
