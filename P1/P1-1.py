# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Ejercicio 1.
Francisco Javier Sáez Maldonado
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
# DISPLAY FIGURE
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(1)

# Auxiliar functions to help with the exercises

def wait():
	input("\n--- Pulsar tecla para continuar ---\n")

def to_numpy(func):
  """Decorador para convertir funciones a versión NumPy"""
  def numpy_func(w):
    return func(*w)

  return numpy_func

def print_output_e1(str_f,initial_point,eta,it,w,fun):
	print("Gradiente descendente sobre la función: " +  str_f)
	print("Punto inicial: {}".format(initial_point))
	print("Tasa de aprendizaje: {}".format(eta))
	print ('Numero de iteraciones: ', it)
	print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
	print('Valor de la función de error en el mínimo : ' , fun(w) )



def plot_min_point_e1(fun,min_point_arg):
	x = np.linspace(-30, 30, 50)
	y = np.linspace(-30, 30, 50)
	X, Y = np.meshgrid(x, y)
	Z = fun([X, Y]) #E_w([X, Y])
	fig = plt.figure()
	ax = Axes3D(fig)
	surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
							cstride=1, cmap='jet')
	min_point = min_point_arg
	min_point_ = min_point[:, np.newaxis]
	ax.plot(min_point_[0], min_point_[1], fun(min_point), 'r*', markersize=10)
	#ax.set(title='Ejercicio 1.1. Función sobre la que se calcula el descenso de gradiente')
	ax.set_xlabel('u')
	ax.set_ylabel('v')
	ax.set_zlabel('E(u,v)')

	plt.show()

def plot_all_e1(fun,min_point_arg,all_points):
	
	f_s = np.array([fun(p) for p in all_points ])
	#x = np.linspace(-30, 30, 50)
	#y = np.linspace(-30, 30, 50)
	x = np.linspace(np.min(all_points[:,0]), np.max(all_points[:,0]), 50)
	y = np.linspace(np.min(all_points[:,1]), np.max(all_points[:,1]), 50)
	X, Y = np.meshgrid(x, y)
	Z = fun([X, Y]) #E_w([X, Y])
	fig = plt.figure()
	ax = Axes3D(fig)
	surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
							cstride=1, cmap='jet',alpha=0.2)
	min_point = np.array([w[0],w[1]])
	min_point_ = min_point[:, np.newaxis]
	ax.plot(all_points[:,0], all_points[:,1],f_s,'g+',markersize=10)
	ax.plot(min_point_[0], min_point_[1], fun(min_point), 'r*', markersize=10)
	#ax.set(title='Ejercicio 1.1. Función sobre la que se calcula el descenso de gradiente')
	ax.set_xlabel('u')
	ax.set_ylabel('v')
	ax.set_zlabel('E(u,v)')

	plt.show()


print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 1.1\n')

@to_numpy
def E(u,v):
    return (u**3 * (np.exp(v-2)) - 2 * (v**2) * (np.exp(-u)))**2  

#Derivada parcial de E con respecto a u
def dEu(u,v):
	return 2 * (u**3 * (np.exp(v-2)) - 2 * (v**2) * (np.exp(-u))) * (3 * (u**2) * (np.exp(v-2)) + 2 * (v**2) * (np.exp(-u)))
    
#Derivada parcial de E con respecto a v
def dEv(u,v):
	return 2 * ((u**3) * (np.exp(v-2)) - 2 * (v**2) * (np.exp(-u))) * ((u**3) * (np.exp(v-2)) - 4 * v * (np.exp(-u)))

#Gradiente de E
@to_numpy
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])

def gradient_descent(eta,fun,grad_fun,maxIter,error2get,initial_point):
	iterations = 0
	w_t = initial_point
	all_w = []
	all_w.append(w_t)

	while iterations < maxIter and fun(w_t) > error2get:
		# All gradient descent in 1 line
		w_t = w_t - eta*grad_fun(w_t)
		all_w.append(w_t)
		# sum iterations
		iterations += 1

	return np.array(all_w),w_t, iterations


eta = 0.1
maxIter = 10000000000
error2get = 1e-14
initial_point = np.array([1.0,1.0])
all_w , w, it = gradient_descent(eta,E,gradE,maxIter,error2get,initial_point)

print_output_e1("E(u,v) = (u^3 e^(v-2) - 2v^2 e^(-u))^2",initial_point,eta,it,w,E)
#plot_min_point_e1(E,w)

plot_all_e1(E,w,all_w)
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
plot_min_point_e1(f,w)
wait()

eta = 0.1
all_w ,w, it = gradient_descent(eta,f,grad_f,maxIter,error2get,initial_point)
print_output_e1("f(x,y) = (x+2)^2 + 2(y-2)^2 + 2 cos(2pi x) cos(2pi y)",initial_point,eta,it,w)
plot_min_point_e1(f,w)
wait()


starting_points = np.array([[-0.5,0.5],[1.0,1.0],[2.1,-2.1],[-3.0,3.0],[-2.0,2.0]])
data = {}
for s in starting_points:
	_,w,it = gradient_descent(eta,f,grad_f,maxIter,error2get,s)
	data[str(s)] = {'(x,y)':w, 'f(w)':f(w),'iterations':it}
	#print("{} \t - \t {} \t - \t {}".format(s,w,f(w)))
	
df = pd.DataFrame.from_dict(data,orient='index')

print(df)





