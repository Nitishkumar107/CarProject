from src.ml_project.logger import logging
from src.ml_project.exception import CustomException
import sys
from src.ml_project.components.data_ingestion import DataIngestion
from src.ml_project.components.data_ingestion import dataIngestionConfig
from src.ml_project.components.data_transformation import dataTransformationConfig,DataTransformation

from src.ml_project.components.model_trainer import ModelTrainer,ModelTrainerConfig


# programe entry point
if __name__ == "__main__":
    logging.info (" the execution of programe started ")

    try :
        #data_ingestion_config = dataIngestionConfig()
        data_ingestion = DataIngestion()
        #DataIngestion.initiate_data_ingestion(data_ingestion)
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()
        #                       || data_ingestion Done and saved in path train_data_path,test_data_path
        #                       

        #data_transformation_config = dataTransformationConfig()
        data_transformation = DataTransformation()
        train_arr,test_arr,_= data_transformation.initiate_data_transformation(train_data_path,test_data_path)
        #logging.info (" data   transformation completed successfully")
        #                       || data_transformation done with ingested data and saved in train_arr,test_arr,_ 

        # model training 
        ModelTraining = ModelTrainer()
        print(ModelTraining.initiatde_model_trainer(train_arr, test_arr))
        #                       || data modelling is done with the transformed data (train_arr,test_arr,_) and then printed


    except Exception as e :
        logging.info ("custome exception")
        raise CustomException(e,sys)

