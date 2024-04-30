from src.ml_project.logger import logging
from src.ml_project.exception import CustomException
import sys
from src.ml_project.components.data_ingestion import DataIngestion
from src.ml_project.components.data_ingestion import dataIngestionConfig



# programe entry point
if __name__ == "__main__":
    logging.info (" the execution of programe started ")

    try :
        #data_ingestion_config = dataIngestionConfig()
        data_ingestion = DataIngestion()
        #DataIngestion.initiate_data_ingestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()
    except Exception as e :
        logging.info ("custome exception")
        raise CustomException(e,sys)

