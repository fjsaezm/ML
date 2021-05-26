
# Regression
import pandas as pd
import numpy as np 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import seaborn as sn

np.random.seed(1)

SAVE = True

path_data = "data/regresion/train.csv"

def wait():
    input("--------------------------------------\n \t \t Presiona Enter para continuar... \n--------------------------------------")

def read_data(path = path_data,test_size = 0.3):
    df = pd.read_csv(path)
    X = df.iloc[:,:-1].to_numpy()
    y = df.iloc[:,-1].to_numpy()

    return train_test_split(X,y,test_size = test_size,random_state = 2022)



# Read the data and split in subsets
X_train,X_test,y_train,y_test =  read_data()


# Box plot of tags
green_diamond = dict(markerfacecolor='g', marker='D')
fig3,ax3 = plt.subplots()
ax3.boxplot(y_train,vert = False,showmeans = True,flierprops = green_diamond)
ax3.set_title("Temperaturas en el conjunto de entrenamiento.")
if SAVE:
    plt.savefig("media/boxplot_y.pdf")
plt.show()
wait()

# Matriz de correlacoines
df = pd.DataFrame(X_train,columns = np.arange(X_train.shape[1]))
corr_matrix = df.corr()

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr_matrix,cmap = 'coolwarm',vmin = 0,vmax = 1)
fig.colorbar(cax)
ticks = np.arange(0,X_train.shape[1],1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
plt.show()
wait()

exit()

sn.heatmap(corr_matrix,annot = False);
if SAVE:
    plt.savefig("media/corr_matrix.pdf")

plt.show()
wait()
