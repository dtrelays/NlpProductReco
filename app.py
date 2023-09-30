import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

from src.components.model_trainer import ModelTrainerConfig
from src.exception import CustomException

from gensim.models import FastText
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer


from src.pipeline.predict_pipeline import PredictPipeline 

config = ModelTrainerConfig()

# Add the path to the src package to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

word2vec_model_path = config.trained_model_word2vec
fastext_model_path = config.trained_model_fastext
fastext_product_vector_path =config.trained_vector_path_fastext
word2vec_product_vector_path = config.trained_vector_path_wordtovec
bert_product_vector_path = config.trained_vector_path_bert

@st.cache_resource(ttl=60 * 60 * 24) 
def load_model():

    model_fastext = FastText.load(fastext_model_path)
    model_word2vec = Word2Vec.load(word2vec_model_path)
    model_bert=SentenceTransformer('bert-base-nli-mean-tokens')

    product_vector_fastext = np.load(fastext_product_vector_path,allow_pickle=True)
    product_vector_word2vec = np.load(word2vec_product_vector_path,allow_pickle=True)
    product_vector_bert = np.load(bert_product_vector_path,allow_pickle=True)

    return model_fastext,model_word2vec,model_bert,product_vector_fastext,product_vector_word2vec,product_vector_bert


model_ftx,model_wvc,model_brt,pv_ftx,pv_wvc,pv_brt=load_model()


def main():
    
    # Set the background color for the app

    html_temp = """
    <div style="background-color: #FF725C; padding: 10px; text-align: center;">
        <h1 style="color: #FFFFFF; font-size: 26px; font-weight: bold; text-transform: uppercase;">Grocery Mart Product Recommendation App</h1>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)

    # Custom styling for the input fields
    input_style = """
    <style>
    .stTextInput, .stNumberInput, .stSelectbox {
        background-color: #FFFFFF;
        color: #333333;
        font-size: 16px;
        font-weight: bold;
        padding: 8px;
        border-radius: 8px;
        box-shadow: none;
        border: 1px solid #111111;
        margin-bottom: 16px;
    }
    .stTextInput:focus, .stNumberInput:focus, .stSelectbox:focus {
        border: 2px solid #05386B;
        box-shadow: none;
    }
    </style>
    """
    st.markdown(input_style, unsafe_allow_html=True)
    
    # Create a search box
    search_query = st.text_input("Enter a product name:")
    
    # Create a dropdown to select the model
    model_options = [ "fastext","word2vec","bert"]
    selected_model = st.selectbox("Select a model:", model_options)


    # Create a button
    if st.button("Search"):
        
        obj = PredictPipeline()
        
        try:
            if selected_model=="fastext":
                model_final = model_ftx
                product_vector_final = pv_ftx
                
            elif selected_model=="word2vec":
                model_final = model_wvc
                product_vector_final = pv_wvc
                
            elif selected_model=="bert":
                model_final = model_brt
                product_vector_final = pv_brt
                    
            df_clean_final = obj.predict(search_query, selected_model, model_final, product_vector_final)    
            
            # Display the product information in a table
            if not df_clean_final.empty:
                 # Add a title to the table
                st.markdown("<h2 style='text-align: left; color: #006699;font-size: 24px;'>Top Recommended Products:</h2>", unsafe_allow_html=True)
                
                st.table(df_clean_final.style.set_properties(**{'text-align': 'center'}))
            else:
                st.write("No products found.")
            
       
        except CustomException as e:

            if "Key" in str(e):
                st.error("Please choose a different model, the query is out of scope for word2vec model.")
            else:
                st.write("An error occurred:", e)
            
        
        # Reset button
        if st.button("Reset"):
            # Clear all inputs and selections
            st.experimental_rerun()


if __name__=="__main__":
    main()
