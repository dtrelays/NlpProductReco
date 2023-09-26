import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle  
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from sklearn.metrics import make_scorer, recall_score,precision_recall_curve, auc
from imblearn.under_sampling import RandomUnderSampler


from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def get_model_accuracy(true, predicted):
    # Calculate accuracy
    accuracy = accuracy_score(true, predicted)
    precision = precision_score(true, predicted)
    recall = recall_score(true, predicted)
    f1 = f1_score(true, predicted)
    # Calculate ROC AUC
    roc_auc = roc_auc_score(true, predicted)
    
    return accuracy, precision, recall, f1,roc_auc
    
def evaluate_models(X_train, y_train,X_val,y_val,models,params):
    try:   
        
        #Defining data_list to get the performance benchmark for all models

        data_list = []
        
        # Initialize the random undersampler
        undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)

        # Apply the undersampler to the training data
        X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)



        # Define a custom scorer for recall
        scorer = make_scorer(recall_score, average='binary') 

        #Hyper parameter tuning for each model in the original list across different params
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=params[list(models.keys())[i]]
            
            gs = RandomizedSearchCV(model, param_distributions=para,n_iter=10, scoring='accuracy', n_jobs=-1, cv=5, verbose=0, random_state=42)
        
            gs.fit(X_train_resampled,y_train_resampled)
            
            best_params = gs.best_params_
            best_estimator = gs.best_estimator_

            best_model = model.set_params(**best_params)
            
            best_model.fit(X_train_resampled,y_train_resampled)

            # Make predictions
            y_train_pred = best_model.predict(X_train_resampled)
            y_val_pred = best_model.predict(X_val)
            
            # Evaluate Train and Test dataset
            model_train_accuracy,model_train_precision,model_train_recall,model_train_f1,train_roc_auc = get_model_accuracy(y_train_resampled, y_train_pred)

            model_test_accuracy, model_test_precision, model_test_recall,model_test_f1,test_roc_auc = get_model_accuracy(y_val, y_val_pred)
                
            print(list(models.keys())[i])
            
            # Print the best parameters and best score
            print("Best Parameters:", best_params)
            
            print('Model performance for Training set')
            print("- Accuracy: {:.4f}".format(model_train_accuracy))
            print("- Precision: {:.4f}".format(model_train_precision))
            print("- Recall: {:.4f}".format(model_train_recall))
            print("- F1 score: {:.4f}".format(model_train_f1))
            print("- ROC AUC: {:.4f}".format(train_roc_auc))

            print('----------------------------------')
            
            #REporting the performance for validation set
            print('Model performance for Validation set')
            print("- Accuracy: {:.4f}".format(model_test_accuracy))
            print("- Precision: {:.4f}".format(model_test_precision))
            print("- Recall: {:.4f}".format(model_test_recall))
            print("- F1 score: {:.4f}".format(model_test_f1))
            print("- ROC AUC: {:.4f}".format(test_roc_auc))
            
            data_list.append({'Model': list(models.keys())[i], 'BestAccuracy': model_test_accuracy,'Recall':model_test_recall,'Precision':model_test_precision,'ROC_AUC':test_roc_auc, 'ModelParams': best_params})
            
            print('='*35)
            print('\n')
        
      
        data = pd.DataFrame(data_list)

        # Sort the DataFrame by the 'FloatColumn' in descending order
        data = data.sort_values(by='Recall', ascending=False)

        # Reset the index to have continuous index values
        data.reset_index(drop=True, inplace=True)
        
        return data

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
