# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Ejercicio 2.
Francisco Javier Sáez Maldonado
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(1)

# Constants for the data
label5 = 1
label1 = -1

# Auxiliar functions to help with the exercises

def wait():
	input("\n--- Pulsar tecla para continuar ---\n")

def to_numpy(func):
  """Decorador para convertir funciones a versión NumPy"""
  def numpy_func(w):
    return func(*w)

  return numpy_func

# Data reading
#def readData(file_x, file_y):
#	# Read data	from file
#	datax = np.load(file_x)
#	datay = np.load(file_y)
#
#	df = pd.DataFrame({'H':1,'Intensidad Promedio' : datax[:,0], 'Simetria' : datax[:,1], 'Y':datay})
#	df = df[(df['Y'] == 1) | (df['Y'] == 5)]
#	# Change 1 by -1
#	df.loc[df['Y'] == 1, 'Y'] = -1
#	# Change 5 by 1
#	df.loc[df['Y'] == 5, 'Y'] = 1
#	# Get data to np vectors
#	x = df[['H', 'Intensidad Promedio', 'Simetria']].to_numpy()
#	y = df[['Y']].to_numpy()
#	
#	return x, y

## Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(label5)
			else:
				y.append(label1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Funcion para calcular el error
def MSE(x,y,w):
	""" Calcula el error cuadrático medio para un modelo lineal"""
	return (np.linalg.norm(x.dot(w) - y)**2)/len(x)

def dMSE(x,y,w):
	""" Calcula la derivada del error cuadrático medio para un modelo lineal"""
	return 2*(x.T.dot(x.dot(w) - y))/len(x)

# Pseudoinverse xT = (x^T x)^-1 x^T 
def pseudoinverse(x,y):
	return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)

# Stochastic Gradient Descent
def sgd(x,y,eta=0.01,max_iterations = 2000,batch_size = 32):

	# Initialize w
	w = np.zeros((x.shape[1],))

	# Create the index for selecting the batches
	index = np.random.permutation(np.arange(len(x)))
	current_index_pos = 0

	for i in range(max_iterations):

		# Select the index that will be used
		iteration_index = index[current_index_pos : current_index_pos + batch_size]
		current_index_pos += batch_size

		# Update w and all_w
		w = w - eta*dMSE(x[iteration_index, :], y[iteration_index],w)

		# Re-do the index if we have used all the data
		if current_index_pos > len(x) or current_index_pos + batch_size > len(x):
			index = np.random.permutation(np.arange(len(x)))
			current_index_pos = 0

	return w

def scatter(x,y = None,ws = None,labels = None,reg_titles = None ,xlabel_title = None , ylabel_title = None, title = "",save = True):
	"""
	Funcion que permite pintar puntos en el plano
	- x: datos
	- y: etiquetas (opcional)
	- w: vector con pesos de regresion (opcional)
	- labels: etiquetas para los puntos segun el modelo de regresion(opcional)
	
	Se crean dos casos para los modelos de regresión que no sean lineales, se utiliza el caso de 6 características
	"""
	# Init subplot
	_, ax = plt.subplots()
	ax.set_xlabel(xlabel_title)
	ax.set_ylabel(ylabel_title)
	# Set plot margins
	xmin, xmax = np.min(1.1*x[:, 1]), np.max(1.1*x[:, 1])
	ax.set_xlim(xmin, xmax)
	ax.set_ylim(1.1*np.min(x[:, 2]), np.max(1.1*x[:, 2]))
	# No classes given
	if y is None:
		ax.scatter(x[:, 1], x[:, 2],marker=".")

	# Classes given
	else:
		colors = {-1: 'green', 1: 'red'}
		# For each class
		for cls, name in [(-1, "Clase -1"), (1, "Clase 1")]:
      		# Get points of that class
			class_points = x[np.where(y == cls)[0]]
			# Plot them
			ax.scatter(	class_points[:, 1],
                		class_points[:, 2],
                		c = colors[cls],
                		label = name,
						marker=".")

		# Plot regressions
		if ws is not None:
			# Get plot limits
			xmin, xmax = ax.get_xlim()
			ymin, ymax = ax.get_ylim()
			x = np.array([xmin, xmax])

			if labels is None:
				# Plot regression results
				for w in ws:
					# Linear regression
					if len(w) == 3:
						ax.plot(x, (-w[1]*x - w[0])/w[2])
					# Non linear regression
					else:
						X, Y = np.meshgrid(np.linspace(xmin-0.2, xmax+0.2, 100), np.linspace(ymin-0.2, ymax+0.2, 100))
						F = w[0] + w[1]*X + w[2]*Y + w[3]*X*Y + w[4]*X*X + w[5]*Y*Y
						plt.contour(X, Y, F, [0])

					
			else:
				for w,a in zip(ws,labels):
					# Linear regression
					if len(w) == 3:
						ax.plot(x, (-w[1]*x - w[0])/w[2],label=a)
						# Non linear regression
					else:
						X, Y = np.meshgrid(np.linspace(xmin-0.2, xmax+0.2, 100), np.linspace(ymin-0.2, ymax+0.2, 100))
						F = w[0] + w[1]*X + w[2]*Y + w[3]*X*Y + w[4]*X*X + w[5]*Y*Y
						plt.contour(X, Y, F, [0]).collections[0].set_label(a)

	
	if y is not None or ws is not None:
		ax.legend()
  	
	ax.set_title(title)

	if save:
		plt.savefig("media/"+title+".pdf")
	plt.show() 



# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')


print('EJERCICIO SOBRE REGRESION LINEAL\n')
print('Ejercicio 2\n')


eta = 0.01
max_iterations = 2000
batch_size = 32

w = sgd(x,y,eta,max_iterations,batch_size)
print ('Bondad del resultado para grad. descendente estocastico en {} iteraciones:\n'.format(max_iterations))
print ("\tEin: ", MSE(x,y,w))
print ("\tEout: ", MSE(x_test, y_test, w))

scatter(x,y,xlabel_title = "Intensidad promedio", ylabel_title = "Simetría",title = "Dibujo de los datos con etiquetas")
wait()
scatter(x,y,[w],labels = ["SGD"],xlabel_title = "Intensidad promedio", ylabel_title = "Simetría",title = "Regresión SGD en train")
wait()
scatter(x_test,y_test,[w],labels = ["SGD"],xlabel_title = "Intensidad promedio", ylabel_title = "Simetría",title = "Regresión SGD en test")
wait()

w_pseudo = pseudoinverse(x, y)
print ('Bondad del resultado para pseudoinversa:')
print ("\tEin: ", MSE(x,y,w_pseudo))
print ("\tEout: ", MSE(x_test, y_test, w_pseudo))

scatter(x,y,[w_pseudo],labels = ["Pseudoinverse"],xlabel_title = "Intensidad promedio", ylabel_title = "Simetría",title = "Regresión pseudoinversa en train")
wait()
scatter(x_test,y_test,[w_pseudo],labels = ["Pseudoinverse"],xlabel_title = "Intensidad promedio", ylabel_title = "Simetría",title = "Regresión pseudoinversa en test")
wait()

scatter(x,y,[w,w_pseudo],labels = ["SGD","Pseudoinverse"],xlabel_title = "Intensidad promedio", ylabel_title = "Simetría",title="Ambas regresiones en train")
wait()
scatter(x_test,y_test,[w,w_pseudo],labels = ["SGD","Pseudoinverse"],xlabel_title = "Intensidad promedio", ylabel_title = "Simetría",title="Ambas regresiones en test")
wait()

print("-------------------------------------- \n")
print('Ejercicio 2.2\n')

# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

def sign(x):
	if x >= 0:
		return 1
	return -1

@to_numpy
def f(x1, x2):
	"""Function used in 2.2 exercise"""
	return sign((x1-0.2)**2 + x2**2 - 0.6) 


N = 1000
size = 1
dimensions = 2

def generate_data(noise = True):
	"""
	Generates data adding 1 in the first col
	"""
	x = simula_unif(N,dimensions,size)
	## Generate tags
	y = np.array([f(a) for a in x])
	if noise:
		y[900:] = np.random.choice([-1,1],100)
	
	# Añade columna de 1s
	x = np.hstack((np.ones((1000, 1)), x))
	return x, y

# Definition of two functions to simulate feature augmentation and experiment
# Create new features for the data

def generate_features(x):
	x1x2 =  np.multiply(x[:,1],x[:,2])[:, None]
	x1x1 =  np.multiply(x[:,1],x[:,1])[:, None]
	x2x2 =  np.multiply(x[:,2],x[:,2])[:, None]
	return np.hstack((x, x1x2, x1x1, x2x2))



# Generate N=1000 points in the space
x,y = generate_data()
# Scatter plot it:
scatter(x,xlabel_title = "x1", ylabel_title = "x2",title= "1000 datos generados")
wait()
scatter(x,y,xlabel_title = "x1", ylabel_title = "x2",title= "1000 datos generados, con etiquetas y ruido")
wait()

print("Ejecución simple. Ajuste de modelo lineal sobre los datos.")
w = sgd(x,y,eta,max_iterations,batch_size)
print ('Bondad del resultado para grad. descendente estocastico en {} iteraciones:\n'.format(max_iterations))
print ("\tEin: ", MSE(x,y,w))

scatter(x,y,[w],labels = ["SGD"],xlabel_title = "x1", ylabel_title = "x2",title = "Regresión SGD en train")
wait()


# Repeat 'iterations'
def experiment(iterations = 1000, more_features = False):
	"""
	Function that repeats the experiment of regression in random data 'iterations'
	times and returns the results :
	- iterations: number of iterations, default = 1000
	- more_features: default=False, indicate true for generating more features
	"""
	E_in = 0
	E_out = 0
	for i in range(iterations):
		# Generate train, test data
		x_train,y_train = generate_data()
		x_test,y_test = generate_data()
		if more_features:
			x_train = generate_features(x_train)
			x_test  = generate_features(x_test)

		# Find w using sgd
		#w = sgd(x_train,y_train)
		w = sgd(x_train,y_train)
		# Evaluate w in both sets
		E_in = E_in + MSE(x_train,y_train,w)
		E_out = E_out + MSE(x_test,y_test,w)

	return E_in/iterations,E_out/iterations


print('Ejecución de N=1000 experimentos de ajuste de modelo lineal en curso ... \n')
# 1000 Experiments
E_in,E_out = experiment()
print('Bondad media del resultado en 1000 experimentos para SGD con 3 características')
print("\t Average Ein: ",E_in)
print("\t Average Eout: ",E_out)



print("\n Ajuste de los datos usando un modelo no lineal")
x,y = generate_data()
x = generate_features(x)
w = sgd(x,y)
print ('Bondad del resultado en modelo de 6 características para grad. descendente estocastico en {} iteraciones:\n'.format(max_iterations))
print ("\tEin: ", MSE(x,y,w))
scatter(x,y,[w],labels = ["Non linear SGD"],xlabel_title = "x1", ylabel_title = "x2",title = "Regresión SGD usando modelo no lineal")
wait()

print("Ejecucion de N=1000 experimentos de ajuste de modelo no lineal en curso ... \n")
E_in,E_out = experiment(more_features=True)
print('Bondad media del resultado en 1000 experimentos para SGD con 6 características')
print("\t Average Ein: ",E_in)
print("\t Average Eout: ",E_out)
