from src.pipeline.predict_pipeline import PredictPipeline
import pandas as pd

if __name__=="__main__":

    obj = PredictPipeline()
    df_clean_final = obj.predict("disinfectant toilet cleaner","bert")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', 100)
    print(df_clean_final)