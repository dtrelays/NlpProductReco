import os
import sys
import re
import nltk
from nltk.corpus import stopwords
import numpy as np 
import pandas as pd
import dill
import pickle  

from src.exception import CustomException

nltk.data.path.append("./nltk_data")

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

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

def remove_stop_words(text):
    words = text.split()
    words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(words)