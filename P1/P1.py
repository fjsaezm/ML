# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Francisco Javier Sáez Maldonado
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(1)

# Auxiliar functions to help with the exercises

def wait():
	input("\n--- Pulsar tecla para continuar ---\n")

def to_numpy(func):
  """Decorador para convertir funciones a versión NumPy"""

  def numpy_func(w):
    return func(*w)

  return numpy_func

def print_output_e1(str_f,initial_point,eta,it,w):
	print("Gradiente descendente sobre la función: " +  str_f)
	print("Punto inicial: {}".format(initial_point))
	print("Tasa de aprendizaje: {}".format(eta))
	print ('Numero de iteraciones: ', it)
	print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')

# DISPLAY FIGURE
from mpl_toolkits.mplot3d import Axes3D

def plot_e1(fun):
	x = np.linspace(-30, 30, 50)
	y = np.linspace(-30, 30, 50)
	X, Y = np.meshgrid(x, y)
	Z = fun([X, Y]) #E_w([X, Y])
	fig = plt.figure()
	ax = Axes3D(fig)
	surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
							cstride=1, cmap='jet')
	min_point = np.array([w[0],w[1]])
	min_point_ = min_point[:, np.newaxis]
	ax.plot(min_point_[0], min_point_[1], fun([min_point_[0], min_point_[1]]), 'r*', markersize=10)
	ax.set(title='Ejercicio 1.1. Función sobre la que se calcula el descenso de gradiente')
	ax.set_xlabel('u')
	ax.set_ylabel('v')
	ax.set_zlabel('E(u,v)')

	plt.show()


print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 1.1\n')

@to_numpy
def E(u,v):
    return (u**3 * np.exp(v-2) - 2 * v**2 * np.exp(-u))**2  

#Derivada parcial de E con respecto a u
def dEu(u,v):
	return 2 * (u**3 * np.exp(v-2) -2 * v**2 * np.exp(-u)) * (3 * u**2 * np.exp(v-2) + 2 * v**2 * np.exp(-u))
    
#Derivada parcial de E con respecto a v
def dEv(u,v):
	return 2 * (u**3 * np.exp(v-2) - 2 * v**2 * np.exp(-u)) * (u**3 * np.exp(v-2) - 4 * v * np.exp(-u))

#Gradiente de E
@to_numpy
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])

def gradient_descent(eta,E,gradE,maxIter,error2get,initial_point):
	iterations = 0
	w_t = initial_point
	all_w = np.array(w_t)

	while iterations < maxIter and E(w_t) > error2get:
		# All gradient descent in 1 line
		w_t = w_t - eta*gradE(w_t)
		all_w = np.append(all_w,w_t)
		# sum iterations
		iterations += 1

    
	return all_w,w_t, iterations


eta = 0.1
maxIter = 10000000000
error2get = 1e-14
initial_point = np.array([1.0,1.0])
_ , w, it = gradient_descent(eta,E,gradE,maxIter,error2get,initial_point)


print_output_e1("E(u,v) = (u^3 e^(v-2) - 2v^2 e^(-u))^2",initial_point,eta,it,w)
plot_e1(E)
wait()



@to_numpy
def f(x,y):
	return (x + 2)**2 + 2*(y - 2)**2 + 2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

def dfx(x,y):
	return 2 * (x+2) + 4 * np.pi * np.cos(2* np.pi * x) * np.sin(2* np.pi * y)

def dfy(x,y):
	return 4 * (y-2) + 4 * np.pi * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) 

@to_numpy
def grad_f(x,y):
    return np.array([dfx(x,y), dfy(x,y)])


print("\n")
print("Ejercicio 1.2.")

eta = 0.01
maxIter = 50
error2get = 1e-14
initial_point = np.array([-1.0,1.0])
all_w ,w, it = gradient_descent(eta,f,grad_f,maxIter,error2get,initial_point)

print_output_e1("f(x,y) = (x+2)^2 + 2(y-2)^2 + 2 cos(2pi x) cos(2pi y)",initial_point,eta,it,w)
plot_e1(f)
wait()

eta = 0.1
all_w ,w, it = gradient_descent(eta,f,grad_f,maxIter,error2get,initial_point)
print_output_e1("f(x,y) = (x+2)^2 + 2(y-2)^2 + 2 cos(2pi x) cos(2pi y)",initial_point,eta,it,w)
plot_e1(f)
wait()


starting_points = np.array([[-0.5,0.5],[1.0,1.0],[2.1,-2.1],[-3.0,3.0],[-2.0,2.0]])
data = {}
for s in starting_points:
	_,w,it = gradient_descent(eta,f,grad_f,maxIter,error2get,s)
	data[str(s)] = {'(x,y)':w, 'f(w)':f(w),'iterations':it}
	#print("{} \t - \t {} \t - \t {}".format(s,w,f(w)))
	
df = pd.DataFrame.from_dict(data,orient='index')

print(df)

exit()





###############################################################################
###############################################################################
###############################################################################
###############################################################################
print('EJERCICIO SOBRE REGRESION LINEAL\n')
print('Ejercicio 2\n')

label5 = 1
label1 = -1

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
def Err(x,y,w):
    return 

# Gradiente Descendente Estocastico
#def sgd(?):
#    #
#    return w

# Pseudoinversa	
#def pseudoinverse(?):
#    #
#    return w


# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')


#w = sgd(?)
print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

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