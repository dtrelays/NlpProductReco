![Product Search Recommendation](artifacts/image.png)

# NLP Product Search Recommendation

This project is an implementation of NLP-based product search and recommendation using FastText and Word2Vec. It aims to provide a solution for searching and recommending products based on user queries. The project includes the training of word embedding models using FastText and Word2Vec and a Streamlit app for interactive product search and recommendation.

## 1. Project Overview

The goal of this project is to build a recommendation system that can help users find products based on their search queries. This system utilizes Natural Language Processing (NLP) techniques and word embedding models, such as FastText and Word2Vec, to understand the semantics of product descriptions, product names, and other textual data. The models learn to represent words and phrases in a continuous vector space, allowing for efficient similarity calculations and recommendations.


## 2. Data Collection

The project utilizes a dataset from Kaggle that contains product information, including product names, categories, brands, and prices. This dataset serves as the foundation for training the NLP models.

You can find the dataset -> https://www.kaggle.com/datasets/surajjha101/bigbasket-entire-product-list-28k-datapoints

## 3. Data Preprocessing and EDA

Before training the NLP models and building the recommendation system, the project involved several key data preprocessing steps and exploratory data analysis to clean and understand the dataset.

- Removing stop words
- Lemmatization
- Punctuation Removal

EDA was conducted to gain insights into the dataset and make informed decisions about model training parameters. Here are some aspects explored during EDA:

Sentence Length Analysis: The distribution of sentence lengths (e.g., product names or descriptions) was examined to understand the range of text lengths in the dataset. This information is valuable for setting model input parameters, such as maximum sequence length.

## 4. Model Selection and Training

The project includes two key word embedding models:

FastText: FastText is used to learn word representations with subword information, which can capture the semantics of both words and subword units.

Word2Vec: Word2Vec is used to create word vectors in a continuous vector space, allowing for semantic similarity calculations.

These models are trained on the dataset to capture the relationships between words and phrases.

## 5.Model Inference

User queries are converted into vectors using the chosen NLP model (FastText or Word2Vec). Cosine similarity is then used to rank and fetch the most relevant products for the query. This approach ensures that users receive accurate and meaningful product recommendations based on semantic similarity, rather than simple keyword matching.

This enhanced search and recommendation system leverages vectorization and cosine similarity to improve the user experience and provide highly relevant product suggestions.

## 6. Git details

Alternatively you can clone my repository and try on your own in local

```bash
## Clone the repository
git clone https://github.com/dtrelays/NlpProductReco.git

## Install dependencies 
pip install -r requirements.txt

## Run the app
streamlit run app.py