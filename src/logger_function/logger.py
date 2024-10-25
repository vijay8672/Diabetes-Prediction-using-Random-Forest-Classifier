import logging
import os
from datetime import datetime

def setup_logging():
    try:
        # Specify the logs directory
        logs_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(logs_dir, exist_ok=True)

        # Create a unique log file name
        LOG_FILE = f"{datetime.now().strftime('%H_%M_%S_%m_%d_%Y')}.log"
        LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

        # Clear existing handlers to avoid duplicate log messages
        logging.getLogger().handlers.clear()

        # Set up the logging configuration
        logging.basicConfig(
            filename=LOG_FILE_PATH,
            format="[%(asctime)s] %(levelname)s %(lineno)d - %(message)s",
            level=logging.INFO
        )

        # Add console handler to display logs in the console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(lineno)d - %(message)s"))
        logging.getLogger().addHandler(console_handler)

        logger = logging.getLogger(__name__)
        logger.info("Logging has started")

        return logger

    except Exception as e:
        print(f"An error occurred: {e}")

# Instantiate the logger object
logger = setup_logging()
