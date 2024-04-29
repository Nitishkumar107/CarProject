from src.ml_project.logger import logging
from src.ml_project.exception import CustomException
import sys



# programe entry point
if __name__ == "__main__":
    logging.info (" the execution of programe started ")

    try :
        a = 1/0
    except Exception as e :
        logging.info ("custome exception")
        raise CustomException(e,sys)

