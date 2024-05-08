import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.ml_project.logger import logging
from src.ml_project.utils import CustomException

import os   

from src.ml_project.utils import save_object

@dataclass
class dataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts","preprocessor.pkl") # after transformation the data will saved at this pickle file

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = dataTransformationConfig() # initialize the data transformation config object

    def get_data_transformation_object(self):  # this function will be responsible for feature transformation
        " This function will be responsible for data transformation"
        
        try:
            '''# Create Column Transformer with 3 types of transformers
            num_features = X.select_dtypes(exclude='object').columns
            cat_features = X.select_dtypes(include='object').columns
            '''
            # OR You can write like this also
            numerical_columns = ['Reading_Score','Writing_Score']
            categorical_columns = ['Gender',	'Race_Ethnicity',	'Parental_Level_of_Education',	'Lunch',	'Test_Preparation'		]

            # to handle missing values with  median values
            num_pipeline = Pipeline(steps=[
                                ('imputer',SimpleImputer(strategy= 'median') ),
                                ('scaler',StandardScaler())
                            ])
            
            # to hadle missing values with categorical values with most frequent values or mode 
            cat_pipeline = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy = 'most_fre')),
                    ('one_hot_encoder',OneHotEncoder()), # OneHotEncoder is a commonly used technique for converting categorical variables into a numerical representation 
                    ('scaler',StandardScaler(with_mean = False)) # standardize features by removing the mean and scaling them to unit variance 

                ])

            logging.info(f'Categorical Colunns :{categorical_columns}')
            logging.info(f'Numerical Columns: {numerical_columns}')

            # num_pipeline and cat_pipeline are two different pipelines but we need to combine them together
            # to do that we need to use ColumnTransformer
            preprocessor = ColumnTransformer(transformers=[
                # pipeline name, pipeline object, columns
                ('num_pipeline',num_pipeline,numerical_columns),
                ('cat_pipeline',cat_pipeline,categorical_columns)
            ])

            return preprocessor

        except Exception as e :
            raise CustomException(e,sys)




    def initiate_data_transformation(self,train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading data from tain and test files...")

            preproccessing_obj = self.get_data_transformation_object()

            target_columns_name = 'Maths_Score'
            numerical_columns = ['Writing_Score', 'Reading_Score']

            # divide the train dataset to independent and dependent features
            input_feature_train_df = train_df.drop(columns=[target_columns_name],axis=1)
            target_feature_train_df = train_df[target_columns_name]


            # divide the test dataset to independent and dependent features
            input_feature_test_df = test_df.drop(columns=[target_columns_name],axis=1)
            target_feature_test_df = test_df[target_columns_name]


            logging.info('applying processing on training and test datasets')

            input_feature_train_arr=preproccessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preproccessing_obj.transform(input_feature_test_df)

            train_arr =np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr =np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            logging.info(f"saved preprocessing object")

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,obj=preproccessing_obj)
            return (
                train_arr,test_arr, self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
                    









