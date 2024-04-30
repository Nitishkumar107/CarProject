from src.ml_project.logger import logging
from src.ml_project.exception import CustomException
import sys
import os
import pandas as pd
from dotenv import load_dotenv
import pymysql


load_dotenv()

host = os.getenv('host')
user = os.getenv('user')
password = os.getenv('password')
db = os.getenv('db')    

def read_sql_data():
    logging.info('reading SQL databse started')
    try:
        conn = pymysql.connect(host=host, user=user, password=password, db=db)
        logging.info('connection established',conn)
        df = pd.read_sql_query('select * from student', conn)
        print(df.head())

        return df

    except Exception as e:
        raise CustomException(e)















