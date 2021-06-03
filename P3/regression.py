# Regression
import pandas as pd
import numpy as np 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sn
from sklearn.pipeline import Pipeline
from timeit import default_timer
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error


np.random.seed(1)

SAVE = True
SEED = 2022

path_data = "data/regresion/train.csv"

def wait():
    input("--------------------------------------\n \t \t Presiona Enter para continuar... \n--------------------------------------\n")

# Function to read the data from CSV file. Split in train and test.
def read_data(path = path_data,test_size = 0.3):
    df = pd.read_csv(path)
    X = df.iloc[:,:-1].to_numpy()
    y = df.iloc[:,-1].to_numpy()

    return train_test_split(X,y,test_size = test_size,random_state = SEED)

# Box plot of tags
def box_plot_values(values,title = " ",save_name = "boxplot_y"):
    green_diamond = dict(markerfacecolor='g', marker='D')
    fig3,ax3 = plt.subplots()
    ax3.boxplot(values,vert = False,showmeans = True,flierprops = green_diamond)
    ax3.set_title(title)
    if SAVE:
        plt.savefig("media/" + save_name + ".pdf")
    plt.show()
    wait()

# Standardize data per columns to a N(0,1)
def standardize_data(data):
    Xc = data.copy()
    Xc = StandardScaler().fit_transform(Xc)
    return Xc

# Apply standardization + PCA to data
def preprocess_data(data):
    Xp = standardize_data(data)
    pca = PCA(n_components = 0.95)
    pca.fit(Xp)
    Xp = pca.transform(Xp)
    
    return Xp,pca



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


# Given Model Pipe and parameter space, apply gridsearch
# Return the result of GridSearchCV
def gridSearch(model_pipe,search_space,csv_title):
    start = default_timer()
    best_reg = GridSearchCV(model_pipe, search_space, scoring = "neg_mean_squared_error", cv = 5, n_jobs = -1)
    best_reg.fit(X_train,y_train)
    end = default_timer() - start
    
    print("Terminado en {}.".format(end))


    if SAVE:
        df = pd.concat([pd.DataFrame(best_reg.cv_results_["params"]),pd.DataFrame(-best_reg.cv_results_["mean_test_score"])],axis = 1)
        df.to_csv(csv_title,float_format='%.5f')

    return best_reg


# Prints best model in the grid
def print_best(grid):

    print(" ----- Mejor regresor lineal encontrado ------")
    print(" - Parámetros:")
    print(grid.best_params_['reg'])
    print(" - Error en Cross Validation")
    print(-grid.best_score_)
    print("----------------------------------------------")

#Prints all models in the grid: params and scores
def print_all_scores(grid):
    print("Grid scores")
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


# Plot y_predicted vs y_true
def plot_residues_error(y_true, y_pred):

    fig, axs = plt.subplots()

    reg = LinearRegression()
    
    # Prediction error
    x = y_true.reshape(-1, 1)
    y = y_pred
    axs.scatter(x, y, facecolors = 'none', edgecolors = 'tab:blue', marker = '.')
    axs.set_xlabel("y")
    axs.set_ylabel("y_pred")
    axs.set_title("Error de predicción")

    # Adjust linear regressor and show identity
    reg.fit(x, y)
    m = reg.coef_[0]
    b = reg.intercept_
    xx = np.array([x.min() - 0.1, x.max() + 0.1])
    axs.plot(xx, m * xx + b, color = 'blue', ls = "--", lw = 2, label = "Mejor ajuste")
    axs.plot(xx, xx, color = 'red', ls = "--", lw = 2, label = "Identidad")
    axs.legend()

    if SAVE:
        plt.savefig("media/" + "residues_error.pdf")
    
    plt.show()
    wait()



# -------------------------------------------------------------------
# ------------------------ VISUALIZATION ----------------------------
# -------------------------------------------------------------------


# Read the data and split in subsets
X_train,X_test,y_train,y_test =  read_data()

# Find if there are columns with low std
df = pd.DataFrame(X_train,columns = np.arange(X_train.shape[1]))
print("Printing columns which variance is smaller than 0.05...")
for col in df.columns:
    if df.var()[col] <= 0.05:
        print(col)
print("Done")


# Box plot Y values in train
#box_plot_values(y_train,title = "Temperaturas en el conjunto de entrenamiento.")

# Average std on datas features
avg_std = np.mean([X_train[i].std() for i in range(0,X_train.shape[0]) ])
print("Average Standard deviation of the features of the dataset per row: {}".format(avg_std))

# Data preprocessing
print("Preprocessing...")



# Plot standardized data's correlation matrix
Xp = standardize_data(X_train)
#corr_matrix(Xp,"corr-normalized")

# Find high correlated variables
dfp = pd.DataFrame(Xp,columns = np.arange(Xp.shape[1]))
corr_m = dfp.corr().abs()
# Select upper triangle of correlation matrix
upper = corr_m.where(np.triu(np.ones(corr_m.shape), k=1).astype(bool))
# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] >= 0.95)]


#X_train_rem = X_train.copy()
#X_train_rem = np.delete(X_train_rem, np.array(to_drop), axis = 1)
print("There are {} variables which correlation with another is greater than 0.95".format(len(to_drop)))






# -------------------------------------------------------------------
# ------------------------ TRAINING MODEL ---------------------------
# -------------------------------------------------------------------

# Define pipelines for only Scaler and PCA
preprocess = [
    ("standardize",StandardScaler())
]

preprocess_pca = [
    ("pre-standardize", StandardScaler()),
    ("PCA", PCA(n_components = 0.95)),
    ("standardize",StandardScaler())
]
preprocess_pipeline = Pipeline(preprocess)
preprocess_pca_pipeline = Pipeline(preprocess_pca)

# Apply pipeline to Xtrain to fit it for the test
X_train_pre = preprocess_pipeline.fit_transform(X_train)
X_train_pre_pca = preprocess_pca_pipeline.fit_transform(X_train)



# By using LinearRegression(), we can use later any linear regression
model_pipe = Pipeline(preprocess + [("reg",LinearRegression())])
model_pca_pipe = Pipeline(preprocess_pca + [("reg", LinearRegression())])

search_space = [
    {"reg": [SGDRegressor(penalty = 'l2',
                          random_state = SEED)],
     'reg__alpha' : [1/10.0**i for i in range(1,5)],
     'reg__learning_rate':['constant','optimal','adaptive'],
     'reg__max_iter':[5000,10000],
    },
    {'reg': [Ridge()],
     'reg__alpha': [1/10.0**i for i in range(1,5)],
     'reg__max_iter': [5000,10000]
    
    }

]

#-------------------------------------------------
# Search only standarizing
print("Realizando la búsqueda en el espacio dado de modelos lineales solo con estandarización...", flush = True)

only_standarization = gridSearch(model_pipe,search_space,"only-standarization-results.csv")





#-------------------------------------------------
# Search using PCA
print("Realizando la búsqueda en el espacio dado de modelos lineales aplicando PCA...", flush = True)
pca_best = gridSearch(model_pca_pipe,search_space,"pca-results.csv")

print("Mejor en solo estandarización")
print_best(only_standarization)
print("Mejor usando PCA")
print_best(pca_best)


# -------------------------------------------------------------------
# ---------------------- EVALUATE IN TEST ---------------------------
# -------------------------------------------------------------------


best_model = only_standarization.best_estimator_

print("Entrenando el mejor modelo en el conjunto de entrenamiento completo...")
best_model.fit(X_train,y_train)
print("Prediciendo etiquetas en el conjunto de test...")
pred_y_test = best_model.predict(X_test)
mse_test = mean_squared_error(y_test,pred_y_test)
r_squared = best_model.score(X_test,y_test)


print("-----------------------------------------")
print("-----------------------------------------")
print("------ RESULTADOS FINALES EN TEST -------\n")
print("\t- MSE: {}".format(mse_test))
print("\t- R^2: {} ".format(r_squared))


plot_residues_error(y_test, pred_y_test)





