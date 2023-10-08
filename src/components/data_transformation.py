import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object,remove_special_characters,remove_stop_words_and_lemmatize

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def initiate_data_transformation(self,clean_data_path):

        try:
            clean_data=pd.read_csv(clean_data_path)

            logging.info("Read clean data completed")
            
            # List of column names to clean
            columns_to_clean = ['combined_text','product', 'category', 'sub_category','brand','type']

            # Apply the functions to each column in the DataFrame
            for column in columns_to_clean:
            # Apply the special character removal function
                clean_data[column + '_v1'] = clean_data[column].apply(remove_special_characters)
                
                # Apply the stop word removal function
                clean_data[column + '_v1'] = clean_data[column + '_v1'].apply(remove_stop_words_and_lemmatize)
            
            clean_data['length'] = clean_data['combined_text_v1'].str.len()
            
            clean_data['word_count'] = clean_data['combined_text_v1'].apply(lambda x: len(x.split()))
            
            # Initialize an empty list for sentences
            sentences = []

            # List of columns you want to include
            columns_to_train = ['combined_text_v1','product_v1', 'category_v1', 'sub_category_v1','brand_v1','type_v1']

            # Iterate through the columns and extend the sentences list
            for column in columns_to_train:
                sentences.extend([text.split() for text in clean_data[column]])
                            
            logging.info(
                f"Created sentences from the combined text column"
            )

            return sentences,clean_data
        
        except Exception as e:
            raise CustomException(e,sys)
