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


# Funcion para leer los datos
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
def sgd(x,y,eta=0.01,max_iterations = 500,batch_size = 32):

	# Initialize w
	w = np.zeros((x.shape[1],))
	all_w = [w]

	# Create the index for selecting the batches
	index = np.random.permutation(np.arange(len(x)))
	current_index_pos = 0

	for i in range(max_iterations):

		# Select the index that will be used
		iteration_index = index[current_index_pos : current_index_pos + batch_size]
		current_index_pos += batch_size

		# Update w and all_w
		w = w - eta*dMSE(x[iteration_index, :], y[iteration_index],w)
		all_w.append(w)

		# Re-do the index if we have used all the data
		if current_index_pos > len(x) or current_index_pos + batch_size > len(x):
			index = np.random.permutation(np.arange(len(x)))
			current_index_pos = 0

	return w,all_w

def scatter(x,y = None,ws = None,labels = None, title = ""):
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
	ax.set_xlabel('Intensidad Promedio')
	ax.set_ylabel('Simetría')
	# Set plot margins
	xmin, xmax = np.min(x[:, 1]), np.max(x[:, 1])
	ax.set_xlim(xmin, xmax)
	ax.set_ylim(np.min(x[:, 2]), np.max(x[:, 2]))
	# No classes given
	if y is None:
		ax.scatter(x[:, 1], x[:, 2])

	# Classes given
	else:
		colors = {-1: 'green', 1: 'red'}
		# For each class
		for cls, name in [(-1, "Clase -1"), (1, "Clase 1")]:
      		# Get points of that class
			class_points = x[y==cls]
			# Plot them
			ax.scatter(	class_points[:, 1],
                		class_points[:, 2],
                		c = colors[cls],
                		label = name)

		if ws is not None:
			# Get plot limits
			x = np.array([xmin, xmax])

			if labels is None:
				for w in ws:
					ax.plot(x, -w[1]*x - w[0]/w[2])
				#for a in w:
				#	ax.plot(x, (-a[1]*x - a[0])/a[2])
					
			else:
				for w in ws:
					ax.plot(x, -w[1]*x - w[0]/w[2],label=name)
				#for a, name in zip(w, labels):
				#	ax.plot(x, (-a[1]*x - a[0])/a[2], label=name)
	
	if y is not None or w is not None:
		ax.legend()
  	
	ax.set_title(title)
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

w,all_w = sgd(x,y,eta,max_iterations,batch_size)
print ('Bondad del resultado para grad. descendente estocastico en {} iteraciones:\n'.format(max_iterations))
print ("\tEin: ", MSE(x,y,w))
print ("\tEout: ", MSE(x_test, y_test, w))
scatter(x,y,[w],title = "Regresión SGD en train")
scatter(x_test,y_test,[w],title = "Regresión SGD en test")


wait()

w_pseudo = pseudoinverse(x, y)
print ('Bondad del resultado para pseudoinversa:')
print ("\tEin: ", MSE(x,y,w_pseudo))
print ("\tEout: ", MSE(x_test, y_test, w_pseudo))

scatter(x,y,[w_pseudo],title = "Regresión pseudoinversa en train")
scatter(x_test,y_test,[w_pseudo],title = "Regresión pseudoinversa en test")
wait()

exit()

#Seguir haciendo el ejercicio...

print('Ejercicio 2\n')
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

def sign(x):
	if x >= 0:
		return 1
	return -1

def f(x1, x2):
	return sign((x1-0.2)**2 + x2**2 - 0.6) 

#Seguir haciendo el ejercicio...