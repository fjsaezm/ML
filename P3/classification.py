
# Regression
import pandas as pd
import numpy as np 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import cm
from sklearn.feature_selection import VarianceThreshold
import seaborn as sn
from sklearn.pipeline import Pipeline
from timeit import default_timer
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error


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

        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = test_size,random_state = SEED)
        return np.array(X_train),np.array(X_test),np.array(y_train),np.array(y_test)


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

# Plot class distribution in train and test
def plot_class_distribution(y_train, y_test, n_classes, save_figures = False, img_path = ""):

    fig, axs = plt.subplots(1, 2, figsize = (12, 6))
    plt.suptitle("Distribución de clases", y = 0.96)
    
    # Diagrama de barras en entrenamiento
    axs[0].bar(np.unique(y_train), [y_train.tolist().count(i) for i in np.unique(y_train)],
        color = cm.get_cmap('plasma',n_classes).colors)
    axs[0].title.set_text("Entrenamiento")
    axs[0].set_xlabel("Clases")
    axs[0].set_ylabel("Número de ejemplos")
    axs[0].set_xticks(range(n_classes))

    # Diagrama de barras en test
    axs[1].bar(np.unique(y_test), [y_test.tolist().count(i) for i in np.unique(y_test)],
        color = cm.get_cmap('plasma',n_classes).colors)
    axs[1].title.set_text("Test")
    axs[1].set_xlabel("Clases")
    axs[1].set_ylabel("Número de ejemplos")
    axs[1].set_xticks(range(n_classes))

    if SAVE:
        plt.savefig(img_path + "class_distr.pdf")
        
    plt.show()
    
    wait()
    
# Correlation Matrix of a set of data
def corr_matrix(data,name_fig):
    df = pd.DataFrame(data,columns = np.arange(data.shape[1]))
    corr_m = df.corr().abs()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(corr_m,cmap = 'coolwarm',vmin = 0,vmax = 1)
    fig.colorbar(cax)

    if SAVE:
        plt.savefig("media/" + name_fig + ".pdf")
    plt.show()
    wait()


    
# Read the data and split in subsets
X_train,X_test,y_train,y_test =  read_data()

# TSNE code commented since execution time is too high and the results are meaningless
#X_train,X_test,y_train,y_test = np.array(X_train),np.array(X_test),np.array(y_train),np.array(y_test)
#tsne = TSNE()
#scatter_plot(tsne.fit_transform(X_train.copy()),y_train,axis = ["x", "y"],title = "Proyección 2-dimensional con TSNE",figname = "tsne")


df = pd.DataFrame(X_train,columns = np.arange(X_train.shape[1]))
sum = 0
for col in df.columns:
    if df.var()[col] < 0.01:
        sum += 1
print("There are {} cols with variance lesser than 0.01".format(sum))

preprocess = [
    ("standardize",StandardScaler()),
    ("var-threshold", VarianceThreshold(0.01))
]

preprocess_pipeline = Pipeline(preprocess)

print("Before pipeline: {}".format(X_train.shape))
X_train_pre = preprocess_pipeline.fit_transform(X_train)
print("After pipeline: {}".format(X_train_pre.shape))





plot_class_distribution(y_train, y_test, n_classes = 11, img_path = "media/")


corr_matrix(X_train_pre,"corr-standarized-classification")

