from src.logger_function.logger import logger  # Import the logger object
import sys

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        _, _, exc_tb = error_detail.exc_info()
        self.line_no = exc_tb.tb_lineno if exc_tb else None
        self.file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else None
        self.error_message = error_message
        logger.info("Created the custom exception class")

    def __str__(self):
        return "Error occurred in python script name [{0}] line number [{1}] error message detail [{2}]".format(
            self.file_name, self.line_no, str(self.error_message)
        )

if __name__ == '__main__':
    try:
        logger.info("Enter the try block")
        a = 1 / 0  # This will raise a ZeroDivisionError
        print("Zero Division Error", a)

    except Exception as e:
        custom_exception = CustomException(e, sys)  # Instantiate before logging
        logger.error(f"An exception occurred: {custom_exception}")  # Log the custom exception details
        raise custom_exception  # Raise the custom exception
