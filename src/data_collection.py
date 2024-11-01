import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv("water_potability.csv")

train_data , test_data =train_test_split(data,test_size=0.2,random_state=69)

data_path = os.path.join("data","raw")

os.makedirs(data_path)
train_data.to_csv(os.path.join(data_path,"train.csv"),index=False)
test_data.to_csv(os.path.join(data_path,"test.csv"),index=False)