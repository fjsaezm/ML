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

print("\U0001F618")

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



def plot_min_point_e1(fun,min_point_arg,v1_title,v2_title,f_title,name="media/fail.pgf"):
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
	ax.set_xlabel(v1_title)
	ax.set_ylabel(v2_title)
	ax.set_zlabel(f_title)

	# Next line was used in development to create the memory
	#plt.savefig(name)
	plt.show()

def plot_all_e1(fun,min_point_arg,all_points,v1_title,v2_title,f_title,name="media/fail.pdf"):
	
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
	#ax.plot(all_points[:,0], all_points[:,1],f_s,'g',markersize=10)
	ax.scatter(all_points[:,0], all_points[:,1],f_s,'g')
	ax.plot(min_point_[0], min_point_[1], fun(min_point), 'r*', markersize=10)
	#ax.set(title='Ejercicio 1.1. Función sobre la que se calcula el descenso de gradiente')
	ax.set_xlabel(v1_title)
	ax.set_ylabel(v2_title)
	ax.set_zlabel(f_title)

	# Next line was used in development to create the memory
	#plt.savefig(name)
	plt.show()

def plot_fun_evolution(fun,all_points,f_title,name="media/fail.pdf"):

	# Create figure
	fig = plt.figure()
	ax = ax = fig.add_subplot(1, 1, 1)
	xs = np.array([i for i in range(all_points.shape[0])])
	ys = np.array([fun(x) for x in all_points])
	# Plot f values
	ax.plot(xs,ys, linestyle='--', marker='o', color='b')
	ax.grid(True)
	ax.set_ylabel(f_title)
	# Save fig for memory purposes
	#plt.savefig(name)
	plt.show()




# Error function for exercise 1.1
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

@to_numpy
# Function for exercise 1.2
def f(x,y):
	return (x + 2)**2 + 2*(y - 2)**2 + 2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
# Partial respect to x
def dfx(x,y):
	return 2 * (x+2) + 4 * np.pi * np.cos(2* np.pi * x) * np.sin(2* np.pi * y)
# Partial respect to u
def dfy(x,y):
	return 4 * (y-2) + 4 * np.pi * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) 

# Gradient of the function for exercise 1.2
@to_numpy
def grad_f(x,y):
    return np.array([dfx(x,y), dfy(x,y)])

#Function that implements the gradient descent algorithm
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



print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 1.1\n')


eta = 0.1
maxIter = 10000000000
error2get = 1e-14
initial_point = np.array([1.0,1.0])
all_w , w, it = gradient_descent(eta,E,gradE,maxIter,error2get,initial_point)

print_output_e1("E(u,v) = (u^3 e^(v-2) - 2v^2 e^(-u))^2",initial_point,eta,it,w,E)
plot_min_point_e1(E,w,v1_title= 'u', v2_title= 'v', f_title='E(u,v)', name = "media/E1-1.pdf")
wait()

print("\n")
print("Dibujo de todos los puntos que el algoritmo ha ido considerando en la búsqueda del mínimo")
plot_all_e1(E,w,all_w,v1_title= 'u', v2_title= 'v', f_title='E(u,v)',name = "media/E1-1-all.pdf")
wait()

print("Valores de la función según las iteraciones")
plot_fun_evolution(E, all_points = all_w, f_title = "E(u,v)",name = "media/f_evolution_e1-1.pdf")
wait()


print("\n")
print("Ejercicio 1.2.")

print("Eta = 0.01")

eta = 0.01
maxIter = 50
error2get = 1e-14
initial_point = np.array([-1.0,1.0])
# Here we set error to -inf to let the algorithm converge for the 50 iterations
all_w ,w, it = gradient_descent(eta,f,grad_f,maxIter,-math.inf,initial_point)

print_output_e1("f(x,y) = (x+2)^2 + 2(y-2)^2 + 2 sin(2pi x) sin(2pi y)",initial_point,eta,it,w,f)
#plot_min_point_e1(f,w)
plot_all_e1(f,w,all_w,v1_title='x',v2_title='y',f_title='f(x,y)',name= "media/E1-2-all.pdf")
wait()


print("Valores de la función según las iteraciones")
plot_fun_evolution(f, all_points = all_w, f_title = "f(x,y)",name = "media/f_evolution_e1-2-001.pdf")
wait()


print("Eta = 0.1")
eta = 0.1
all_w ,w, it = gradient_descent(eta,f,grad_f,maxIter,-math.inf,initial_point)
print_output_e1("f(x,y) = (x+2)^2 + 2(y-2)^2 + 2 sin(2pi x) sin(2pi y)",initial_point,eta,it,w,f)
#plot_min_point_e1(f,w)
plot_all_e1(f,w,all_w,v1_title='x',v2_title='y',f_title='f(x,y)',name= "media/E1-2-loweta-all.pdf")
wait()

print("Valores de la función según las iteraciones")
plot_fun_evolution(f, all_points = all_w, f_title = "f(x,y)",name = "media/f_evolution_e1-2-01.pdf")
wait()


print("\n")
print("Ejercicio 1.2,b) Tabla de valores ")
starting_points = np.array([[-0.5,0.5],[1.0,1.0],[2.1,-2.1],[-3.0,3.0],[-2.0,2.0]])
data = {}

eta = 0.01
for s in starting_points:
	_,w,it = gradient_descent(eta,f,grad_f,maxIter,-math.inf,s)
	data[str(s)] = {'(x,y)':w, 'f(w)':f(w),'iterations':it}
	#print("{} \t - \t {} \t - \t {}".format(s,w,f(w)))
	
df = pd.DataFrame.from_dict(data,orient='index')

# Used to save table to csv
#df.to_csv("ej1-2.csv",sep=',')


points = np.array([[p[0],p[1]] for p in df['(x,y)']])

plot_all_e1(f,w,points,v1_title='x',v2_title='y',f_title='f(x,y)',name= "media/E1-2-minimums.pdf")

print(df)





