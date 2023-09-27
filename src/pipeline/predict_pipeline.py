import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import remove_special_characters,remove_stop_words
from src.components.model_trainer import ModelTrainerConfig
from src.components.data_transformation import DataTransformation
from src.components.data_ingestion import DataIngestionConfig

import os

data_transformation=DataTransformation()

clean_data_path = DataIngestionConfig().clean_data_path


class PredictPipeline:
    def __init__(self):
        pass
    

    def predict(self,query,selected_model,model_final,product_vector_final):
        
        try:    
            sentences,df_clean=data_transformation.initiate_data_transformation(clean_data_path)
                        
            if(selected_model=="word2vec"):
                model = model_final
                similarity_score = get_similarity_word2vec(model, query,product_vector_final)
                df_clean['similarity_score_word2vec'] = similarity_score
                df_clean = df_clean.sort_values(by=['similarity_score_word2vec'], ascending=False)
                
            elif(selected_model=="fastext"):
                model = model_final
                similarity_score = get_similarity_fastext(model, query,product_vector_final)
                df_clean['similarity_score_fastext'] = similarity_score     
                df_clean = df_clean.sort_values(by=['similarity_score_fastext'], ascending=False)
            
            elif(selected_model=="bert"):
                model = model_final
                similarity_score = get_similarity_bert(model, query,product_vector_final)
                df_clean['similarity_score_bert'] = similarity_score
                df_clean = df_clean.sort_values(by=['similarity_score_bert'], ascending=False)
                
            df_clean_final = df_clean[['product','category','brand','market_price']].head(10)
            
            df_clean_final['market_price'] = df_clean_final['market_price'].astype(int)
            df_clean_final.rename(columns = {'product':'Product'}, inplace = True)
            df_clean_final.rename(columns = {'category':'Category'}, inplace = True)
            df_clean_final.rename(columns = {'brand':'Brand'}, inplace = True)
            df_clean_final.rename(columns = {'market_price':'MRP'}, inplace = True)
            
            #hide index before displaying
            df_clean_final.index = np.arange(1, len(df_clean_final) + 1)
            df_clean_final.index.name = 'Rank'
            
            return df_clean_final
        
        except Exception as e:
            raise CustomException(e,sys)


def get_similarity_fastext(model,user_query,product_vector_final):
    
    user_query = remove_special_characters(user_query)
    user_query = remove_stop_words(user_query)

    # Load the saved sentence vectors
    sentence_vectors = product_vector_final

    # Convert the query to a FastText vector
    query_vector = model.wv[user_query]

    # Calculate the cosine similarity between the query and the pre-computed sentence vectors
    cosine_similarity_scores = []

    for product_vector in sentence_vectors:
        cosine_similarity_score = np.dot(query_vector, product_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(product_vector))
        cosine_similarity_scores.append(cosine_similarity_score)

    return cosine_similarity_scores


def get_similarity_word2vec(model, user_query,product_vector_final):
    
    try:
        user_query = remove_special_characters(user_query)
        user_query = remove_stop_words(user_query)

        # Load the saved sentence vectors
        sentence_vectors = product_vector_final
        
        # Convert the query to a list of words
        query_words = user_query.split()

        # Calculate the query vector
        query_vector = model.wv[query_words].mean(axis=0)

        # Calculate the cosine similarity between the query and the pre-computed sentence vectors
        cosine_similarity_scores = []

        for product_vector in sentence_vectors:
            cosine_similarity_score = np.dot(query_vector, product_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(product_vector))
            cosine_similarity_scores.append(cosine_similarity_score)

        return cosine_similarity_scores

    except Exception as e:
            raise CustomException(e,sys)


def get_similarity_bert(model,user_query,product_vector_final):
    
    product_embeddings = product_vector_final
    query_bert = remove_special_characters(user_query)
    query_bert = remove_stop_words(query_bert)

    query_embedding = model.encode(query_bert)

    cosine_similarity_scores_bert = []

    for product_embedding in product_embeddings:
        cosine_similarity_score = np.dot(query_embedding, product_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(product_embedding))
        cosine_similarity_scores_bert.append(cosine_similarity_score)

    return cosine_similarity_scores_bert