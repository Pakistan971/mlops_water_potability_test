import pandas as pd
import os
import numpy as np

train_data  = pd.read_csv(r"data\raw\train.csv")
test_data  = pd.read_csv(r"data\raw\test.csv")

def fill_missing_median(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_val = df[column].median()
            df[column].fillna(median_val,inplace=True)
    return df

train_processed_data = fill_missing_median(train_data)
test_processed_data = fill_missing_median(test_data)

data_path = os.path.join("data","processed")

os.makedirs(data_path)

train_processed_data.to_csv(os.path.join(data_path,"train_processed.csv"),index=False)
test_processed_data.to_csv(os.path.join(data_path,"test_processed.csv"),index=False)