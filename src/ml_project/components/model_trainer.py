import os
import sys
import numpy as np
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error

#from src.ml_project.components.data_ingestion import DataIngestion

from src.ml_project.components.data_transformation import dataTransformationConfig,DataTransformation
from src.ml_project.logger import logging
from src.ml_project.exception import CustomException
from src.ml_project.utils import evaluate_models,save_object



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl") #final model will save here

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig() #class path object initiated

    def eval_metrics(self,actual,pred):
        rmse = np.sqrt(mean_squared_error(actual,pred))
        mae = mean_absolute_error(actual,pred)
        r2 = r2_score(actual,pred)
        return r2, mae, rmse


    def initiatde_model_trainer(self, train_array, test_array):
        try:
            logging.info('split training and test input data')
            X_train, Y_train,X_test,Y_test=(
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {

                "LinearRegression": LinearRegression(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=False),  
                                                }
            # parameters tuning for each model
            params = {
                "Decision Tree" :{
                    'criterion':['squared_error', 'squared_error','friedman_mse','absolute_error','poisson'],
                    'splitter ': ['best','random'] ,
                    'max_features':['sqrt','log2']
                },
                "Random Forest" :{
                    'criterion':['squared_error', 'squared_error','friedman_mse','absolute_error','poisson_error'],
                    'max_features':['sqrt','log2',None],
                    'n_estimators':[8,16,32,64,128,256]
                },
                'Gradient Boosting' :{
                        'loss':['squared_error', 'huber','absolute_error','quantile'],
                        'learning_rate':[.1,.01,.05,.001],
                        'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                        'criterion':['squared_error', 'friedman_mse'],
                        'max_features':['sqrt','log2','auto'],
                        'n_estimators':[8,16,32,64,128,256]
                },
                'XGBoost' :{
                        'loss':['squared_error', 'huber','absolute_error','quantile'],
                        'learning_rate':[.1,.01,.05,.001],
                        'n_estimators':[8,16,32,64,128,256],
                        'criterion':['squared_error', 'friedman_mse']

                },
                ' LinearRegression ': {},
                'KNeighborsRegressor':{
                        'n_neighbors':[5,10,15,20,25,30,35,40,45,50],
                        'weights':['uniform','distance'],
                        'algorithm':['auto','ball_tree','kd_tree','brute'],
                        'leaf_size':[10,20,30,40,50,60,70,8]
                },

                'AdaBoostRegressor':{
                    'n_estimators':[8,16,32,64,128,256],
                    'learning_rate':[.1,.01,.05,.001],
                    'loss':['linear','square','exponential']
                },  
                'CatBoostRegressor':{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators':[8,16,32,64,128,256],
                    'loss_function':['RMSE','MAE','MAPE','Poisson'],
                    'iterations':[30,50,100]
                }

            }
            
            model_report: dict = evaluate_models(X_train,Y_train,X_test,Y_test,models,params)

            # to get best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            # to get best model name from dictionary
            best_model_name = list(model_report.keys())[
                    list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]


            if best_model_score < 0.6:
                raise CustomException('No best model found')
            logging.info(f'Best found model on both training and testing dataset')

            logging.info(f'best model is {best_model_name}')

            # for model saving in directory if >60%
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,obj=best_model
            )
            # best model pridiction 
            predicted = best_model.predict(X_test)

            r2_square = r2_score(Y_test, predicted)
            return r2_square



        except Exception as e:
            raise CustomException(e,sys)



