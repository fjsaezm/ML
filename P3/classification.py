
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
from matplotlib import cm


np.random.seed(1)

SAVE = True
SEED = 2022

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
        for line in lines:
            vec = line.split(" ")
            X.append([float(v) for v in vec[0:len(vec)-1]])
            y.append(int(vec[-1]))

        return train_test_split(X,y,test_size = test_size,random_state = SEED)



def scatter_plot(X, y, axis, ws = None, labels = None, title = None,
                 figname = "", cmap = 'coolwarm', save_figures = False, img_path = ""):
    """Muestra un scatter plot de puntos (opcionalmente) etiquetados por clases,
       eventualmente junto a varias rectas de separación.
         - X: matriz de características de la forma [x1, x2].
         - y: vector de etiquetas o clases. Puede ser None.
         - axis: nombres de los ejes.
         - ws: lista de vectores 2-dimensionales que representan las rectas
           (se asumen centradas).
         - labels: etiquetas de las rectas.
         - title: título del plot.
         - figname: nombre para guardar la gráfica en fichero.
         - cmap: mapa de colores."""

    # Establecemos tamaño, colores e información del plot
    plt.figure(figsize = (8, 6))
    plt.xlabel(axis[0])
    plt.ylabel(axis[1])
    if title is not None:
        plt.title(title)

    # Establecemos los límites del plot
    xmin, xmax = np.min(X[:, 0]), np.max(X[:, 0])
    ymin, ymax = np.min(X[:, 1]), np.max(X[:, 1])
    scale_x = (xmax - xmin) * 0.01
    scale_y = (ymax - ymin) * 0.01
    plt.xlim(xmin - scale_x, xmax + scale_x)
    plt.ylim(ymin - scale_y, ymax + scale_y)

    # Mostramos scatter plot con leyenda
    scatter = plt.scatter(X[:, 0], X[:, 1], c = y, cmap = cmap)
    if y is not None:
        legend1 = plt.legend(
            *scatter.legend_elements(),
            title = "Clases",
            loc = "upper right")

    # Pintamos las rectas con leyenda
    if ws is not None:
        # Elegimos los mismos colores que para los puntos
        mask = np.ceil(np.linspace(0, len(cmap.colors) - 1, len(np.unique(y)))).astype(int)
        colors = np.array(cmap.colors)[mask]

        for w, l, c in zip(ws, labels, colors):
            x = np.array([xmin - scale_x, xmax + scale_x])
            plt.plot(x, (-w[0] * x) / w[1], label = l, lw = 2, ls = "--", color = c)
        plt.legend(loc = "lower right")

    # Añadimos leyenda sobre las clases
    if y is not None:
        plt.gca().add_artist(legend1)

    if SAVE:
        plt.savefig("media/" + figname + ".pdf")
    
    plt.show()

    wait()

    
# Read the data and split in subsets
X_train,X_test,y_train,y_test =  read_data()
X_train,X_test,y_train,y_test = np.array(X_train),np.array(X_test),np.array(y_train),np.array(y_test)
print(X_train.shape)
pca = PCA(n_components = 30 )
X_pca = pca.fit_transform(X_train)
tsne = TSNE(n_components = 2, verbose = 1, perplexity = 500, n_iter = 1000, learning_rate  = 200)
scatter_plot(tsne.fit_transform(X_pca),y_train,axis = ["x", "y"],title = "Proyección 2-dimensional con TSNE",figname = "tsne")
#print(X_train)
#print(y_train)

