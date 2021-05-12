from ML_Pipeline.Constants import remove_columns
from ML_Pipeline.utils import null_processing
import pandas as pd



def remove_unused_columns(df,column_names=remove_columns):
    for col in column_names:
        if col in df.columns:
            df = df.drop(column_names,axis=1)
    return df

def clean_datasets(df):
    # remove unused column
    df = remove_unused_columns(df)
    #impute null values
    feature_df = null_processing(df)
    return feature_df

