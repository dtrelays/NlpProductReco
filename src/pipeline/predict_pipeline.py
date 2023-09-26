import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features,model,preprocessor):
        try:
            data_scaled=preprocessor.transform(features)
            preds = model.predict(data_scaled)
            
            print(preds[[0]])
            
            if preds[0]==0:
                default="Low"
            else:
                default="High"
            
            return default
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        age: int,
        income: int,
        credit_history_length: int,
        loan_amount: int,
        interest_rate:float,
        employment_length:int,
        home_type:str,
        intent_type:str):
 

        self.age = age
        self.income = income
        self.credit_history_length = credit_history_length
        self.loan_amount = loan_amount
        self.interest_rate = interest_rate
        self.employment_length = employment_length
        self.home_type = home_type
        self.intent_type = intent_type
        


    def get_data_as_data_frame(self):
        try:
        
            custom_data_input_dict = {
                "Age": [self.age],
                "Income": [self.income],
                "Cred_length": [self.credit_history_length],
                "Amount": [self.loan_amount],
                "Rate": [self.interest_rate],
                "Emp_length": [self.employment_length],
                "Home": [self.home_type],
                "Intent": [self.intent_type],
                "Percent_income": [self.loan_amount/self.income]
            }

            df = pd.DataFrame(custom_data_input_dict)
            df['Percent_income'] = round(df['Percent_income'],2)
            
            return df

        except Exception as e:
            raise CustomException(e, sys)

