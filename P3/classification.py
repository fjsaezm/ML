
# Regression
import pandas as pd
import numpy as np 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import cm
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
import seaborn as sn
from sklearn.pipeline import Pipeline
from timeit import default_timer
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.metrics import classification_report

from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_score,confusion_matrix,recall_score,f1_score
from sklearn import svm


np.random.seed(1)

SAVE = False
SEED = 2022
show = True

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
    
# Searches for the model that fits the best the data
def gridSearch(X,y,model_pipe,search_space,csv_title):
    start = default_timer()
    best_reg = GridSearchCV(model_pipe, search_space, scoring = "accuracy", cv = 5, n_jobs = -1)
    best_reg.fit(X_train_rem,y_train_rem)
    end = default_timer() - start
    
    print("Terminado en {}.".format(end))


    if SAVE:
        df = pd.concat([pd.DataFrame(best_reg.cv_results_["params"]),pd.DataFrame(best_reg.cv_results_["mean_test_score"])],axis = 1)
        df.to_csv(csv_title,float_format='%.5f')

    return best_reg

def print_best(grid):

    print(" ----- Mejor clasificador lineal encontrado ------")
    print(" - Parámetros:")
    print(grid.best_params_['clf'])
    print(" - Accuracy en Cross Validation")
    print(grid.best_score_)
    print("----------------------------------------------")

# Plot confusion matrix using certain classifier and given a dataset.
def confusion_matrix(clf, X, y):

    fig, ax = plt.subplots(1, 1, figsize = (8, 6))
    disp = plot_confusion_matrix(clf, X, y, cmap = cm.Blues, values_format = 'd', ax = ax)
    disp.ax_.set_title("Matriz de confusión")
    disp.ax_.set_xlabel("Etiqueta predicha")
    disp.ax_.set_ylabel("Etiqueta real")

    if SAVE:
        plt.savefig("media/confusion.pdf")

    plt.show()
    wait()

# -------------------------------------------------------------------
# ------------------------ VISUALIZATION ----------------------------
# -------------------------------------------------------------------



# Read the data and split in subsets
X_train,X_test,y_train,y_test =  read_data()

if show:

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


#Outlier detection
from sklearn.ensemble import IsolationForest
clf = IsolationForest(n_estimators = 1000, max_samples = 'auto',random_state = SEED).fit(X_train)
val = clf.predict(X_train)
print("There are outliers: ")
out = np.where(val == -1)[0]
print("Outliers represent:{}% so they will be eliminated".format((len(out)/X_train.shape[0])*100))
print(len( np.where(val == -1)[0]))
X_train_rem = np.array([X_train[i] for i in range(X_train.shape[0]) if i not in out])
y_train_rem = np.array([y_train[i] for i in range(y_train.shape[0]) if i not in out])
print("Shape before and after removing outliers")
print(X_train.shape)
print(X_train_rem.shape)



# -------------------------------------------------------------------
# ------------------------ TRAINING MODEL ---------------------------
# -------------------------------------------------------------------

# Re-read the data
#X_train,X_test,y_train,y_test =  read_data()


preprocess = [
    ("var-thresh",VarianceThreshold())
]

# Define preprocessors
preprocess = [
    ("standardize",StandardScaler())
]

preprocess_pca = [ 
    ("pre-standardize", StandardScaler()),
    ("PCA", PCA(n_components = 0.95)),
    ("standardize",StandardScaler()),
    ("var-thresh",VarianceThreshold()),
]

preprocess_anova = [
    ("pre-standardize", StandardScaler()),
    ("ANOVA", SelectKBest(score_func=f_classif, k=24)),
    ("standardize",StandardScaler()),
    ("var-thresh",VarianceThreshold()),
]

preps = [preprocess, preprocess_pca,preprocess_anova]

search_space = [
        {"clf": [LogisticRegression(multi_class = 'ovr',
                                    penalty = 'l2')],
         "clf__C": [10**(-i) for i in range(-1,4)],
         "clf__max_iter":[5000,10000]},
        {"clf": [svm.SVC(kernel='linear',
                         decision_function_shape = 'ovr')],
         "clf__C": [10**(-i) for i in range(-1,4)],
         "clf__max_iter":[5000,10000],
         "clf__tol":[1e-4]
        }
]

names = ["solo estandarización","estandarización y PCA", "estandarización y ANOVA"]
csv_names = ["clf-standardized","clf-pca","clf-anova"]
best_performers = []
# Iterate through preprocessors
for prep,name,csv_name in zip(preps,names,csv_names):
    # Classificator is a placeholder, will be changed
    model_pipe = Pipeline(prep + [('clf',LogisticRegression())])
    print("Realizando la búsqueda en el espacio dado de modelos lineales con {}...".format(name), flush = True)
    best = gridSearch(X_train_rem,y_train_rem,model_pipe,search_space,csv_name)
    print_best(best)
    best_performers.append(best)

b = 0
for i in range(1,len(best_performers)):
    if best_performers[i].best_score_ > best_performers[b].best_score_:
        b = i



best_model = best_performers[b].best_estimator_
#best_model = Pipeline(preprocess + [('clf',svm.SVC(kernel='linear',decision_function_shape = 'ovr',C = 1))])

print("Entrenando el mejor modelo en el conjunto de entrenamiento completo...")
best_model.fit(X_train,y_train)
print("Prediciendo etiquetas en el conjunto de test...")


print("-----------------------------------------")
print("-----------------------------------------")
print("------ RESULTADOS FINALES EN TEST -------\n")
pred_y_test = best_model.predict(X_test)
print(classification_report(y_test,pred_y_test))




confusion_matrix(best_model, X_test, y_test)

