import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self,input_features):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            # Create Column Transformer with 3 types of transformers
            num_features = input_features.select_dtypes(exclude="object").columns
            cat_features = input_features.select_dtypes(include="object").columns

            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()

            logging.info(f"Categorical columns: {cat_features}")
            logging.info(f"Numerical columns: {num_features}")

            preprocessor = ColumnTransformer([
                ("OneHotEncoder", oh_transformer, cat_features),
                ("StandardScaler", numeric_transformer, num_features),        
                ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
            
            input_feature_train_df = train_df.drop(columns=['Default','Status'],axis=1)
            target_feature_train_df = train_df['Default']

            input_feature_test_df=test_df.drop(columns=['Default','Status'],axis=1)
            target_feature_test_df=test_df['Default']
            
            preprocessing_obj=self.get_data_transformer_object(input_feature_train_df)

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            target_feature_train_df = target_feature_train_df.values
            target_feature_train_df_v2 = target_feature_train_df.reshape(-1, 1) 
            
            target_feature_test_df = target_feature_test_df.values
            target_feature_test_df_v2 = target_feature_test_df.reshape(-1, 1)

            logging.info(target_feature_train_df_v2.shape)
            logging.info(input_feature_train_arr.shape)
            
            logging.info(type(input_feature_train_arr))
            
            logging.info(target_feature_train_df_v2[:5])

            train_arr = np.hstack((input_feature_train_arr,target_feature_train_df_v2))
            test_arr = np.hstack((input_feature_test_arr,target_feature_test_df_v2))
            
            logging.info(train_arr.shape)
            logging.info(test_arr.shape)

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)
