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
            
            clean_data['combined_text'] = clean_data['combined_text'].apply(remove_special_characters)
            clean_data['combined_text'] = clean_data['combined_text'].apply(remove_stop_words_and_lemmatize)
            
            clean_data['length'] = clean_data['combined_text'].str.len()
            
            clean_data['word_count'] = clean_data['combined_text'].apply(lambda x: len(x.split()))
            
            sentences = [text.split() for text in clean_data['combined_text']]
            
            logging.info(
                f"Created sentences from the combined text column"
            )


            return sentences,clean_data
        
        except Exception as e:
            raise CustomException(e,sys)
