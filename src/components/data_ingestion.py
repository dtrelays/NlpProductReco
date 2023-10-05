import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np

from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    raw_data_path: str=os.path.join('artifacts',"data.csv")
    clean_data_path: str=os.path.join('artifacts',"clean_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook/data/Bigbasket_Data.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.clean_data_path),exist_ok=True)
         
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
                
            df['combined_text'] = df['product'] + ' ' + df['category'] + ' ' + df['sub_category'] + ' ' + df['brand']+ ' ' + df['type']

            df = df.dropna()
            
            clean_data = df
           
            clean_data.to_csv(self.ingestion_config.clean_data_path,index=False,header=True)

            logging.info("Inmgestion of the data is completed")

            return(
                self.ingestion_config.clean_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    
    obj=DataIngestion()
    clean_data=obj.initiate_data_ingestion()
    
    data_transformation=DataTransformation()
    sentences,df_clean=data_transformation.initiate_data_transformation(clean_data)

    modeltrainer=ModelTrainer()
    product_vec_wvc,product_vec_fastext = modeltrainer.initiate_model_trainer(sentences,df_clean)
    
    print("Product vector length using word2vec is ",product_vec_wvc)
    print("Product vector length using fastext is ",product_vec_fastext)
    # print("Product vector length using bert is ",produc_vec_bert)
    
    