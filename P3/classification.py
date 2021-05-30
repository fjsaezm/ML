
# Regression
import pandas as pd
import numpy as np 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sn
from sklearn.manifold import TSNE


np.random.seed(1)

SAVE = True

path_data = "data/clasificacion/Sensorless_drive_diagnosis.txt"

def wait():
    input("--------------------------------------\n \t \t Presiona Enter para continuar... \n--------------------------------------\n")

def read_data(path = path_data,test_size = 0.3):
    
    with open(path) as f:
        X = []
        y = []
        # Pop removes empty line
        lines = f.read().split('\n')
        lines.pop()
        print(len(lines))
        for line in lines:
            vec = line.split(" ")
            X.append([float(v) for v in vec[0:len(vec)-1]])
            y.append(int(vec[-1]))

        print(np.array(X).shape)
        return train_test_split(X,y,test_size = test_size,random_state = 2022)

# Read the data and split in subsets
X_train,X_test,y_train,y_test =  read_data()

palette = sn.color_palette("bright", 10)
X_embedded = TSNE(n_components = 2).fit_transform(X_train)
sn.scatterplot(X_embedded[:,0],X_embedded[:,1],hue = y,legend = 'full', palette = palette)