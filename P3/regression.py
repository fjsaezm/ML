# Regression 
import pandas as pd
import numpy as np 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test,split

np.random.seed(1)

path_data = "data/regresion/train.csv"

def read_data(path = path_data,test_size = 0.3):
    df = pd.read_csv(path)
    X = df.iloc[:,:-1].to_numpy()
    y = df.iloc[:,-1].to_numpy()

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = test_size,random_state = 2022)


print(df.shape)
