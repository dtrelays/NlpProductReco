import os
import sys
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np 
import pandas as pd
import dill
import pickle  

from src.exception import CustomException

nltk.data.path.append("./nltk_data")

nltk.download('stopwords')
nltk.download('wordnet')


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def remove_special_characters(text):
    # Use regex to remove all non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()

    return text

def remove_stop_words_and_lemmatize(text):
    words = text.split()
    words = [lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stop_words]
    return ' '.join(words)


def calculate_weighted_average_vector(list_of_columns,weights,model):

        vectors = []
        for i, column in enumerate(list_of_columns):
            # If the word is a phrase, split it into individual words.
            if isinstance(column, str) and len(column.split()) > 1:
                words = column.split()
                for word in words:
                    if word in model.wv:
                        vectors.append(model.wv[word] * weights[i])
        if vectors:
            return np.sum(vectors, axis=0)/np.sum(weights) 
        else:
            return np.zeros(model.vector_size)

def calculate_weighted_average_vector_fasttext(list_of_columns, weights,model):
 
  vectors = []
  for i, column in enumerate(list_of_columns):
    vectors.append(model.wv[column] * weights[i])
  return  np.sum(vectors, axis=0)/np.sum(weights) 
