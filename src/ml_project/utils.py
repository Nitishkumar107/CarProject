from src.ml_project.logger import logging
from src.ml_project.exception import CustomException
import sys
import os
import pandas as pd
from dotenv import load_dotenv
import pymysql
from sklearn.metrics import r2_score
import pickle
import numpy as np

from src.ml_project.components.model_trainer import ModelTrainer,ModelTrainerConfig


from sklearn.model_selection import GridSearchCV


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

# to save the data in pickle file at the given location , wb is write mode
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok = True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e)


def evaluate_models(X_train, Y_train,X_test,Y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model =list(models.values())[i]
            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv =3) # we are giving every model to GridSearchCV to evaluate
            gs.fit(X_train,Y_train)


            model.set_params(**gs.best_params_)
            model.fit(X_train,Y_train)

            model.fit(X_train,Y_train) # train model

            Y_train_pred = model.predict(X_train)
            Y_test_pred = model.predict(X_test)

            train_model_score = r2_score(Y_train, Y_train_pred)
            test_model_score = r2_score(Y_test, Y_test_pred)

            report = [list(models.keys())[i]]=test_model_score

        return report

    except Exception as e:
        raise CustomException(e,sys)











