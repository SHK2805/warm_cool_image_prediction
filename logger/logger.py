import logging
import os
from datetime import datetime


# Set up logger
logger = logging.getLogger("CVLogger")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Define log message format
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] - %(message)s')
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)

# Example usage
if __name__ == "__main__":
    logger.info("Starting the machine learning project...")
    try:
        # Your machine learning code here
        logger.debug("Debugging information")
        logger.info("Training model...")
        # Simulate a warning
        logger.warning("This is a warning message")
        # Simulate an error
        # raise ValueError("This is an error message")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    logger.info("Finished the machine learning project.")
