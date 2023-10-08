import numpy as np
import os
import sys
from dataclasses import dataclass

from gensim.models import FastText
from gensim.models import Word2Vec
# from sentence_transformers import SentenceTransformer
import pickle

from src.exception import CustomException
from src.logger import logging
from src.utils import calculate_weighted_average_vector,calculate_weighted_average_vector_fasttext


@dataclass
class ModelTrainerConfig:
    trained_model_word2vec=os.path.join("artifacts","model_word2vec.model")
    trained_model_fastext=os.path.join("artifacts","model_fastext.model")
    trained_vector_path_fastext=os.path.join("artifacts","product_vectors_fastext.npy")
    trained_vector_path_wordtovec=os.path.join("artifacts","product_vectors_wordtovec.npy")
    # trained_vector_path_bert=os.path.join("artifacts","product_embeddings_bert.npy")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,sentence_list,df_clean_data):
        try:
            logging.info("Training vocab using word2vec and fastext")
            
            # Model using Word2Vec
            model_wordtovec = Word2Vec(sentence_list, vector_size=200, window=20, min_count=1, sg=0,alpha=0.05, epochs=100, negative=5) 
          
            # Model using FastText
            model_fastext = FastText(sentence_list, vector_size=200, window=20, min_count=1, sg=0,epochs=100)

            logging.info("Saving the trained model fastext and word2vec")

            model_fastext.save(self.model_trainer_config.trained_model_fastext)
            model_wordtovec.save(self.model_trainer_config.trained_model_word2vec)

            product_vectors_fastext = []

            # for product_name_description_sentence in df_clean_data['combined_text']:
            #     vector_fastext = model_fastext.wv[product_name_description_sentence]
            #     product_vectors_fastext.append(vector_fastext)
            
            # Create a product vector for each row in the DataFrame for fasttext
            product_vectors_fastext = df_clean_data[['product_v1', 'category_v1', 'sub_category_v1', 'brand_v1', 'type_v1']].apply(lambda row: calculate_weighted_average_vector_fasttext(row.tolist(), [0.45, 0.15, 0.1, 0.25, 0.05],model_fastext), axis=1)
            
            logging.info("Embedding of the product vectors trained using fastext")
            
            # Save the sentence vectors to a file using a format like Pickle or NumPy
            np.save(self.model_trainer_config.trained_vector_path_fastext, np.array(product_vectors_fastext))

            sentence_vectors_wordtovec = []

            # for product_name_description_sentence in df_clean_data['combined_text']:
            #     words = product_name_description_sentence.split()
            #     vector = model_wordtovec.wv[words].mean(axis=0)
            #     sentence_vectors_wordtovec.append(vector)
                
            # Create a product vector for each row in the DataFrame for Word2Vec
            sentence_vectors_wordtovec = df_clean_data[['product_v1', 'category_v1', 'sub_category_v1', 'brand_v1', 'type_v1']].apply(lambda row: calculate_weighted_average_vector(row.tolist(), [0.45, 0.15, 0.1, 0.25, 0.05],model_wordtovec), axis=1)

            logging.info("Saving Embedding of the product vectors trained using word2vec")

            # Save the sentence vectors to a file using a format like Pickle or NumPy
            np.save(self.model_trainer_config.trained_vector_path_wordtovec, np.array(sentence_vectors_wordtovec))
            
            trained_wordtovec_len = len(sentence_vectors_wordtovec)
            trained_fastext_len = len(product_vectors_fastext)
              
            return trained_wordtovec_len,trained_fastext_len
     
        except Exception as e:
            raise CustomException(e,sys)
        
        
    
