# Machine Learning
# Francisco Javier Sáez Maldonado
# Worksheet 0

# Part 1

import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
import matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split

# Obtain iris database
iris = datasets.load_iris()
# Create a dataframe with the iris data, and create new column with
# the labels. Obtain the class names in c_names
iris_df = pd.DataFrame(iris['data'],columns = iris['feature_names'])
iris_df['labels'] = iris['target']
c_names = iris.target_names


# Obtain 1st and 3rd features
sub_df = iris_df[[iris.feature_names[0],iris.feature_names[2]]]
sub_df['labels'] = iris_df['labels']
print(sub_df)
# Set colors
colors = ['orange','black','green']
# Plot the scatter, maping each color to a class
scatter = plt.scatter(
    sub_df[iris.feature_names[0]],
    sub_df[iris.feature_names[2]],
    c = sub_df['labels'],
    cmap = matplotlib.colors.ListedColormap(colors)
)

plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[2])
plt.legend(
    scatter.legend_elements()[0],
    labels = c_names,
    loc = "upper left",
    title = "Classes"
)
plt.show()

exit()

# Part 2
# sklearn function to stratifiedly split 
# In train and test

# Split
X_train, X_test, y_train, y_test = train_test_split(selected,y,stratify = y, test_size = 0.25)

# Part 3
import math
# Create 100 values between 0 and 4*pi
l = np.linspace(0,4*math.pi,num = 100)
# Obtain the valued asked, using sin and cos from math
# Obtain tanh values from np.tanh
sin_x = [math.sin(x) for x in l]
cos_x = [math.cos(x) for x in l]
tanh  = [np.tanh(sin_x[i] + cos_x[i]) for i in range(len(l))]

plt.plot(l,sin_x,label = "sin",linestyle = 'dashed', color = 'green')
plt.plot(l,cos_x,label = "cos",linestyle = 'dashed', color = 'black')
plt.plot(l,tanh,label = "tanh",linestyle = 'dashed', color = 'red')
plt.show()