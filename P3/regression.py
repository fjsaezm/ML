e
# Regression
import pandas as pd
import numpy as np 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sn

np.random.seed(1)

SAVE = True

path_data = "data/regresion/train.csv"

def wait():
    input("--------------------------------------\n \t \t Presiona Enter para continuar... \n--------------------------------------\n")

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

# Average std on datas features
avg_std = np.mean([X_train[i].std() for i in range(0,X_train.shape[0]) ])
print("Average Standard deviation of the features of the dataset per row: {}".format(avg_std))

# Data preprocessing



# Correlation Matrix
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

print("Preprocessing...")

def standardize_data(data):
    Xc = data.copy()
    Xc = StandardScaler().fit_transform(Xc)
    return Xc

Xp = standardize_data(X_train)
#corr_matrix(Xp,"corr-normalized")

# Find high correlated variables
dfp = pd.DataFrame(Xp,columns = np.arange(Xp.shape[1]))
corr_m = dfp.corr().abs()
# Select upper triangle of correlation matrix
upper = corr_m.where(np.triu(np.ones(corr_m.shape), k=1).astype(np.bool))
# Find index of feature columns with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

print("There are {} variables which correlation with another is greater than 0.95".format(len(to_drop)))


def preprocess_data(data):
    Xp = standardize_data(data)
    pca = PCA(n_components = 0.95)
    pca.fit(Xp)
    Xp = pca.transform(Xp)
    
    return Xp,pca

X_t_pca,transformer = preprocess_data(X_train)
print("Data's shape after PCA: {}".format(X_t_pca.shape))

corr_matrix(X_t_pca,"corr-pca")