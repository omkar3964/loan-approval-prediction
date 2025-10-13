import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
import sys

# Constants
LOG_DIR = 'logs'
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3  # Number of backup log files

# Create logs directory relative to project root
root_dir = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
log_dir_path = os.path.join(root_dir, LOG_DIR)
os.makedirs(log_dir_path, exist_ok=True)

# Create a log file with timestamp
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
log_file_path = os.path.join(log_dir_path, LOG_FILE)

def get_logger(name: str) -> logging.Logger:
    """
    Returns a module-specific logger with rotating file handler and console output.
    
    Args:
        name (str): Name of the logger, typically __name__ in the calling module.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Avoid adding handlers multiple times
    if not logger.handlers:
        # Formatter
        formatter = logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s")

        # File handler
        file_handler = RotatingFileHandler(
            log_file_path, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT, encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

# Example usage in a module
if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.info("Module-specific logger initialized successfully.")
