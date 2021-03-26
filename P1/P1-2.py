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





print('EJERCICIO SOBRE REGRESION LINEAL\n')
print('Ejercicio 2\n')


# Funcion para calcular el error
def MSE(x,y,w):
    return  ((x.dot(w) - y)**2).mean(axis = ax)

def dMSE(x,w,y):
	return 2*(x.T.dot(x.dot(w) -y ))/len(x)

# Stochastic Gradient Descent
def sgd(x,y,eta=0.01,max_iterations = 500,batch_size = 32):

	# Initialize w
	w = np.zeros((x.shape[1],1))
	all_w = [w]

	# Create the index for selecting the batches
	index = np.random.permutation(np.arange(len(x)))
	current_index_pos = 0

	for i in range(max_iterations):

		# Select the index that will be used
		iteration_index = index[current_index_pos : current_index_pos + batch_size]
		current_index_pos += batch_size

		# Update w and all_w
		w = w - eta*dMSE(x[iteration_index,:], y[iteration_index],w)
		all_w += [w]

		# Re-do the index if we have used all the data
		if current_pos > len(x):
			index = np.random.permutation(np.arange(len(x)))
			current_index_pos = 0

	return all_w, w





   return w

# Pseudoinverse xT = (x^T x)^-1 x^T 
def pseudoinverse(x,y):
    return np.matmul(np.matmul(np.linalg.inv(np.matmul(x,x.T)),x.T),y)


# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')


w = sgd(?)
print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", MSE(x,y,w))
print ("Eout: ", MSE(x_test, y_test, w))

input("\n--- Pulsar tecla para continuar ---\n")

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
	#return sign(?) 
	return 0
#Seguir haciendo el ejercicio...