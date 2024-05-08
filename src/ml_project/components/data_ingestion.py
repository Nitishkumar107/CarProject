# mysql ----> Train, test, split--->dataset
import os
import sys
from src.ml_project.exception import CustomException
from src.ml_project.logger import logging
import pandas as pd
from src.ml_project.utils import read_sql_data
from sklearn.model_selection import train_test_split


# using dataclasses
class dataIngestionConfig:
    train_data_path = os.path.join("artifacts","train.csv")
    test_data_path = os.path.join("artifacts","test.csv")
    raw_data_path = os.path.join("artifacts","raw.csv")
class DataIngestion:
    def __init__(self):
        self.ingestion_config = dataIngestionConfig()


    def initiate_data_ingestion(self):
        try:
            # Reading the data from the dataBase directory
            #df = read_sql_data() # here we get raw data from the utils
            df = pd.read_csv(os.path.join('notebook/data','raw.csv'))
            logging.info('Reading completed from mysql database...')


            # creating directory to save the raw data which get from the database
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) 

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header =True) # saving utils data to raw data file in artifact folder
            
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header =True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header =True)

            logging.info('Data ingestion completed...')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
                #self.ingestion_config.raw_data_path
            )
            

        except Exception as e:
            raise CustomException(e,sys)




