
# Regression
import pandas as pd
import numpy as np 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

np.random.seed(1)

SAVE = True

path_data = "data/regresion/train.csv"

def wait():
    input("Presiona Enter para continuar...")

def read_data(path = path_data,test_size = 0.3):
    df = pd.read_csv(path)
    X = df.iloc[:,:-1].to_numpy()
    y = df.iloc[:,-1].to_numpy()

    return train_test_split(X,y,test_size = test_size,random_state = 2022)




X_train,X_test,y_train,y_test =  read_data()

green_diamond = dict(markerfacecolor='g', marker='D')
fig3,ax3 = plt.subplots()
ax3.boxplot(y_train,vert = False,showmeans = True,flierprops = green_diamond)
ax3.set_title("Diagrama de caja de las temperaturas en el conjunto de entrenamiento.")
if SAVE:
    plt.savefig("media/boxplot_y.pdf")
plt.show()
wait()
