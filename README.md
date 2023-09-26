# Credit Risk Prediction Classifier

![Credit Risk Prediction](artifacts/creditrisk.png)


## 1. Problem Statement

The goal of this project is to develop a machine learning classifier for identifying loan customers with a high propensity to default. The objective is to predict and flag such customers during the loan approval process.

## 2. Data Collection

- **Dataset Source**: [Kaggle - Credit Risk Analysis](https://www.kaggle.com/datasets/nanditapore/credit-risk-analysis)
- **Dataset Overview**: The dataset consists of 12 columns and approximately 32,000 rows.

### Data Columns

- **Demographics**: Age
- **Employment**: Employment Length (providing insights into the duration of employment)
- **Financial Information**: Income, Home Details (e.g., Own/Rent), and Credit History Length (offering insights into financial stability and credit behavior)
- **Loan Details**: Loan Amount, Loan Intent, Interest Rate, etc.
- **Default**: Indicates whether the customer defaulted on the loan (target variable)

## 3. Data Checks and EDA

Before proceeding with modeling, several data checks and EDA were performed:

- Checking for missing values
- Identifying and handling duplicates
- Verifying data types of columns
- Investigating various categories present in different categorical columns
- Did EDA using univariate, bivariate and multivariate analysis

## 4. Model Selection and Performance

Various machine learning models were experimented with, including:

- Logistic Regression
- Decision Tree
- XGBoost
- LightGBM
- Random Forest
- Neural Network

The primary aim was to maximize recall while maintaining a balance between accuracy and precision. The project achieved a __recall of 0.9__ using **LightGBM** and an __ROC AUC score of 0.85__.

## 5. Deployment

The project's application is deployed and accessible at the following URL: [Credit Risk Prediction App](https://creditriskprediction-ur6bezvcodjpx3rqfcje7s.streamlit.app/).


Alternatively you can clone my repository and try on your own in local

```bash
## Clone the repository
git clone https://github.com/dtrelays/CreditRiskPrediction.git

## Install dependencies 
pip install -r requirements.txt

## Run the app
streamlit run app.py
