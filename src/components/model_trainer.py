import os
import sys
from dataclasses import dataclass

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models, get_model_accuracy

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
            "Logistic Regression": LogisticRegression(),
            "XGBClassifier": XGBClassifier(verbose=0),
            "CatBoost Classifier": CatBoostClassifier(verbose=False),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest Classifier": RandomForestClassifier(),
            "Light GBM Classifier":lgb.LGBMClassifier(),
        }
            
                        
            params = {
                "Logistic Regression": {
                    "C": [0.001, 0.01, 0.1, 1, 10],
                    "penalty": ["l1", "l2"]
                },
                "XGBClassifier": {
                    "learning_rate": [0.01, 0.1, 0.2],
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 4, 5,10,20],
                    "min_child_weight": [1, 2, 3],
                    "subsample": [0.8, 0.9, 1.0],
                    "colsample_bytree": [0.8, 0.9, 1.0]
                },
                "CatBoost Classifier": {
                    "iterations": [100, 200, 300],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "depth": [3, 4, 5,10,20],
                    "l2_leaf_reg": [1, 3, 5, 7, 9]
                },
                "Decision Tree": {
                    "max_depth": [10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "criterion": ["gini", "entropy"]
                },
                "Random Forest Classifier": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "criterion": ["gini", "entropy"]
                },
                "Light GBM Classifier": {
                    "learning_rate": [0.01, 0.1, 0.2],
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 4, 5,10,20],
                    "num_leaves": [31, 63, 127]
                }
            }

            model_report=evaluate_models(X_train=X_train,y_train=y_train,X_val=X_test,y_val=y_test,
                                             models=models,params=params)
            
            ## To get best model score from dict
            best_model_score = model_report.loc[0,'Recall']

            ## To get best model name from dict

            best_model_name = model_report.loc[0,'Model']
            
            best_model = models[best_model_name]
            
            best_params = model_report.loc[0,'ModelParams']
            
            print(best_model)
            
            logging.info(f"Best found model: {best_model_name}")
            logging.info(f"Best found model recall: {best_model_score}")
            logging.info(f"Best found model params: {best_params}")

            model_and_params = {
                            "model": best_model,
                            "params": best_params
                            }

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model_and_params
            )

            model_with_loaded_params = best_model.set_params(**best_params)
            
            predicted = model_with_loaded_params.predict(X_test)

            accuracy,precision,recall,f1_score,roc_auc = get_model_accuracy(y_test, predicted)
            
            return recall
            
            
        except Exception as e:
            raise CustomException(e,sys)