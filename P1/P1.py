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
	input("\n--- Pulsar ENTER para continuar ---\n")

def to_numpy(func):
  """Decorador para convertir funciones a versión NumPy"""
  def numpy_func(w):
    return func(*w)

  return numpy_func



# ------------------------------------------
# ------------- EJERCICIO 1 ----------------
# ------------------------------------------


def print_output_e1(str_f,initial_point,eta,it,w,fun):
	print("Gradiente descendente sobre la función: " +  str_f)
	print("Punto inicial: {}".format(initial_point))
	print("Tasa de aprendizaje: {}".format(eta))
	print ('Numero de iteraciones: ', it)
	print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
	print('Valor de la función de error en el mínimo : ' , fun(w) )



def plot_min_point_e1(fun,min_point_arg,v1_title,v2_title,f_title,name="media/fail.pgf"):
	""" Plots the point that is the minimum that the algorithm found """
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
	""" Function that plots the value of a function f over the surface that the function f defines """

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
	#min_point = np.array([w[0],w[1]])
	min_point = min_point_arg
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

def plot_fun_evolution(fun,all_points,f_title,save = False ,name="media/fail.pdf"):
	""" Function that plots the value of a function f in each iteration in R^2 """

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
	if save:
		plt.savefig(name)
	plt.show()


@to_numpy
def E(u,v):
	"""Error function for exercise 1.1"""
	return (u**3 * (np.exp(v-2)) - 2 * (v**2) * (np.exp(-u)))**2  

def dEu(u,v):
	""" Partial derivative of E respect u"""
	return 2 * (u**3 * (np.exp(v-2)) - 2 * (v**2) * (np.exp(-u))) * (3 * (u**2) * (np.exp(v-2)) + 2 * (v**2) * (np.exp(-u)))
    
def dEv(u,v):
	""" Partial derivative of E respect v"""
	return 2 * ((u**3) * (np.exp(v-2)) - 2 * (v**2) * (np.exp(-u))) * ((u**3) * (np.exp(v-2)) - 4 * v * (np.exp(-u)))

@to_numpy
def gradE(u,v):
	""" Gradient of function E"""
	return np.array([dEu(u,v), dEv(u,v)])

@to_numpy
def f(x,y):
	""" Function defined for exercise 1.2"""
	return (x + 2)**2 + 2*(y - 2)**2 + 2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

def dfx(x,y):
	"""Partial respect to x of f(x,y)"""
	return 2 * (x+2) + 4 * np.pi * np.cos(2* np.pi * x) * np.sin(2* np.pi * y)

def dfy(x,y):
	""" Partial respect to y of f(x,y)"""
	return 4 * (y-2) + 4 * np.pi * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y) 

@to_numpy
def grad_f(x,y):
	"""Gradient of the function for exercise 1.2"""
	return np.array([dfx(x,y), dfy(x,y)])

def gradient_descent(eta,fun,grad_fun,maxIter,error2get,initial_point):
	"""
	Implementation of the gradient descent algorithm.Parameters:
	- eta : learning rage
	- fun: function to descend the algorithm
	- grad_fun: gradient of the function
	- maxIter: maximum of iterations
	- error2get: error that must be achieved
	- initial_point: starting point of the algorithm
	"""
	iterations = 0
	w_t = initial_point.copy()
	all_w = []
	all_w.append(w_t)

	while iterations < maxIter and fun(w_t) > error2get:
		# All gradient descent in 1 line
		w_t = w_t - eta*grad_fun(w_t)
		all_w.append(w_t)
		# sum iterations
		iterations += 1

	return np.array(all_w),w_t, iterations






def ej1():

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
	plot_all_e1(f,w,all_w,v1_title='x',v2_title='y',f_title='f(x,y)',name= "media/E1-2-all.pdf")
	wait()


	print("Valores de la función según las iteraciones")
	plot_fun_evolution(f, all_points = all_w, f_title = "f(x,y)",name = "media/f_evolution_e1-2-001.pdf")
	wait()

	eta = 0.1
	all_w ,w, it = gradient_descent(eta,f,grad_f,maxIter,-math.inf,initial_point)
	print_output_e1("f(x,y) = (x+2)^2 + 2(y-2)^2 + 2 sin(2pi x) sin(2pi y)",initial_point,eta,it,w,f)
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

	dataframe = pd.DataFrame.from_dict(data,orient='index')

	# Used to save table to csv
	#dataframe.to_csv("ej1-2.csv",sep=',')


	points = np.array([[p[0],p[1]] for p in dataframe['(x,y)']])

	plot_all_e1(f,w,points,v1_title='x',v2_title='y',f_title='f(x,y)',name= "media/E1-2-minimums.pdf")

	print(dataframe)


# ------------------------------------------
# ------------ EJERCICIO 2 -----------------
# ------------------------------------------


np.random.seed(1)

# Constants for the data
label5 = 1
label1 = -1

N = 1000
size = 1
dimensions = 2

# Auxiliar functions to help with the exercises

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

def scatter(x,y = None,ws = None,labels = None,reg_titles = None ,xlabel_title = None , ylabel_title = None, title = "",save = False):
	"""
	Funcion que permite pintar puntos en el plano
	- x: datos
	- y: etiquetas (opcional)
	- w: vector con pesos de regresion (opcional)
	- labels: etiquetas para los puntos segun el modelo de regresion(opcional)
	- reg_titles : titulos para las rectas de regresion (opcional)
	- xlabel_title :  titulo para la etiqueta del eje x (opcional)
	- ylabel_title : titulo para la etiqueta del eje y (opcional)
	- title: Titulo del gráfico (opcional)
	- save : indica si se guardará el gráfico en memoria.
	
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

def generate_data(noise = True):
		"""
		Generates data adding 1 in the first col
		"""
		x = simula_unif(N,dimensions,size)
		## Generate tags
		y = np.array([f_2(a) for a in x])
		if noise:
			idxs = np.random.choice(N, int(0.1 * N), replace = False)
			y[idxs] = -y[idxs]

		# Homogenize
		x = np.hstack((np.ones((1000, 1)), x))
		return x, y

	# Definition of two functions to simulate feature augmentation and experiment
	# Create new features for the data

def generate_features(x):
	x1x2 =  np.multiply(x[:,1],x[:,2])[:, None]
	x1x1 =  np.multiply(x[:,1],x[:,1])[:, None]
	x2x2 =  np.multiply(x[:,2],x[:,2])[:, None]
	return np.hstack((x, x1x2, x1x1, x2x2))


def simula_unif(N, d, size):
	"""Simula datos en un cuadrado [-size,size]x[-size,size]"""
	return np.random.uniform(-size,size,(N,d))
def sign(x):
	if x >= 0:
		return 1
	return -1


@to_numpy
def f_2(x1, x2):
    """Function used in 2.2 exercise"""
    return sign((x1-0.2)**2 + x2**2 - 0.6) 


def ej2():

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
    print("\t Pesos obtenidos:",w)  
    print("Datos con etiquetas")
    scatter(x,y,xlabel_title = "Intensidad promedio", ylabel_title = "Simetría",title = "Dibujo de los datos con etiquetas")
    wait()
    print("Regresión SGD en Train")
    scatter(x,y,[w],labels = ["SGD"],xlabel_title = "Intensidad promedio", ylabel_title = "Simetría",title = "Regresión SGD en train")
    wait()
    print("Regresión SGD en test")
    scatter(x_test,y_test,[w],labels = ["SGD"],xlabel_title = "Intensidad promedio", ylabel_title = "Simetría",title = "Regresión SGD en test")
    wait()  
    w_pseudo = pseudoinverse(x, y)
    print ('Bondad del resultado para pseudoinversa:')
    print ("\tEin: ", MSE(x,y,w_pseudo))
    print ("\tEout: ", MSE(x_test, y_test, w_pseudo))
    print("\t Pesos obtenidos:",w_pseudo)   
    print("Regresión Pseudoinversa en train")
    scatter(x,y,[w_pseudo],labels = ["Pseudoinverse"],xlabel_title = "Intensidad promedio", ylabel_title = "Simetría",title = "Regresión pseudoinversa en train")
    wait()
    print("Regresión Pseudoinversa en test")
    scatter(x_test,y_test,[w_pseudo],labels = ["Pseudoinverse"],xlabel_title = "Intensidad promedio", ylabel_title = "Simetría",title = "Regresión pseudoinversa en test")
    wait()  
    print("Ambas regresiones en train en train")
    scatter(x,y,[w,w_pseudo],labels = ["SGD","Pseudoinverse"],xlabel_title = "Intensidad promedio", ylabel_title = "Simetría",title="Ambas regresiones en train")
    wait()
    print("Ambas regresiones en test")
    scatter(x_test,y_test,[w,w_pseudo],labels = ["SGD","Pseudoinverse"],xlabel_title = "Intensidad promedio", ylabel_title = "Simetría",title="Ambas regresiones en test")
    wait()  
    print("-------------------------------------- \n")
    print('Ejercicio 2.2\n')    

    # Generate N=1000 points in the space
    x,y = generate_data()
    # Scatter plot it:
    print("Datos generados")
    scatter(x,xlabel_title = "x1", ylabel_title = "x2",title= "1000 datos generados")
    wait()
    print("Datos con etiquetas")
    scatter(x,y,xlabel_title = "x1", ylabel_title = "x2",title= "1000 datos generados, con etiquetas y ruido")
    wait()  
    print("Ejecución simple. Ajuste de modelo lineal sobre los datos.")
    w = sgd(x,y,eta,max_iterations,batch_size)
    print ('Bondad del resultado para grad. descendente estocastico en {} iteraciones:\n'.format(max_iterations))
    print ("\tEin: ", MSE(x,y,w))   
    scatter(x,y,[w],labels = ["SGD"],xlabel_title = "x1", ylabel_title = "x2",title = "Regresión SGD en train en datos aleatorios")
    wait()  
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





# ------------------------------------------
# ----------------- BONUS ------------------
# ------------------------------------------



def hfxx(x, y):
    """Two times partial derivative respect x of f(x, y)."""
    return 2 - 8 * np.pi ** 2 * np.sin(2 * np.pi * y) * np.sin(2 * np.pi * x)

def hfyy(x, y):
    """Two times partial derivative respect x of f(x, y)."""
    return 4 - 8 * np.pi ** 2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

def hfxy(x, y):
    """Partial derivative respect x and then respect y of f(x,y)."""
    return 8 * np.pi ** 2 * np.cos(2 * np.pi * y) * np.cos(2 * np.pi * x)

@to_numpy
def hf(x, y):
    """f(x, y) Hessian."""
    return np.array([[hfxx(x, y), hfxy(x, y)],
					[hfxy(x, y), hfyy(x, y)]])


def print_output_bonus(str_f,initial_point,eta,it,w,fun):
	print("Método de Newton sobre la función: " +  str_f)
	print("Punto inicial: {}".format(initial_point))
	print("Tasa de aprendizaje: {}".format(eta))
	print ('Numero de iteraciones: ', it)
	print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')
	print('Valor de la función de error en el mínimo : ' , fun(w) )


def plot_bonus_comparison(function,ws,titles,iterations,title,save = False):
	""" Function that draws a set of functions given the ws that take in each iteration"""

	plt.xlabel("Iteraciones")
	plt.ylabel("f(x,y)")
	for w,t in zip(ws,titles):
		plt.plot(range(iterations+1),[f(a) for a in w], '.',label = str(t), linestyle = '--')
	
	plt.title(title)
	plt.legend()

	if save:
		plt.savefig("media/"+ title+".pdf")

	plt.show()


def newton(eta,dfun,hfun,maxIter,initial_point):
	"""
	Newton's method for function optimization. Parameters:
	- eta: learning rate
	- dfun: derivative of the function to optimize
	- hfun: hessian of the function to optimize
	- maxIter : maximum of iterations to perform
	- initial_point : initial point of the optimization
	"""
	# Initialize vectors
	w = initial_point.copy()
	all_w = [w]
	
	# Iterate maxIter times
	for i in range(maxIter):
		# Update weights
		w = w - eta*np.linalg.inv(hfun(w)).dot(dfun(w))
		all_w.append(w)

	return np.array(all_w),w

def bonus():

	print("BONUS: Implementación del método de Newton")
	eta = 0.01
	iterations = 50
	initial_point = np.array([-1.0,1.0])

	# First execution, eta = 0.01
	all_w, w = newton(eta,grad_f,hf,iterations,initial_point)
	print_output_bonus("f(x,y) = (x+2)^2 + 2(y-2)^2 + 2 sin(2pi x) sin(2pi y)", initial_point, eta, iterations, w, f)
	plot_bonus_comparison(f,[all_w],["Newton , Eta = 0.01"],iterations, title = "Newton's first execution, eta = 0.01")
	wait()

	#Augment Eta
	eta = 0.1
	all_w_2,w_2 = newton(eta,grad_f,hf,iterations,initial_point)
	print_output_bonus("f(x,y) = (x+2)^2 + 2(y-2)^2 + 2 sin(2pi x) sin(2pi y)", initial_point, eta, iterations, w_2, f)



	# Plot comparison of etas
	plot_bonus_comparison(f,[all_w,all_w_2] , ["Newton Eta = 0.01", "Newton Eta = 0.1"], iterations, title = "Comparison Newton's method different etas")
	wait()


	print("Comparativa de método de Newton con Gradiente Descendente")

	# Perform gradient descent to compare
	eta_gd = 0.01
	all_w_gd, w_gd , _ = gradient_descent(eta_gd,f,grad_f,iterations,-math.inf,initial_point) 


	plot_bonus_comparison(f, [all_w,all_w_2,all_w_gd], ["Newton Eta = 0.01", "Newton Eta = 0.1","Gradient Descent, Eta = 0.01"], iterations, title = "Comparision Newton's method with gradient descent")
	wait()


	print("BONUS: Ejecución del método de newton sobre una serie de puntos.")
	starting_points = np.array([[-0.5,0.5],[1.0,1.0],[2.1,-2.1],[-3.0,3.0],[-2.0,2.0]])
	data = {}

	ws = []
	eta = 0.01
	for s in starting_points:
		all_w, w = newton(eta,grad_f,hf,iterations,s)
		ws.append(all_w)
		data[str(s)] = {'(x,y)':w, 'f(w)':f(w),'iterations':iterations}

	dataframe = pd.DataFrame.from_dict(data,orient='index')

	# Used to save table to csv
	#dataframe.to_csv("BONUS.csv",sep=',')
	print(dataframe)

	# Perform gradient descent to compare
	eta_gd = 0.01
	all_w_gd, w_gd , _ = gradient_descent(eta_gd,f,grad_f,iterations,-math.inf,initial_point) 
	initial_point = np.array([-2.1,2.1])
	all_w_gd_2,w_gd_2 ,_ = gradient_descent(eta_gd,f,grad_f,iterations,-math.inf,initial_point) 

	ws.append(all_w_gd)
	ws.append(all_w_gd_2)
	plot_bonus_comparison(f, ws , ["[-0.5,0.5]","[1.0,1.0]","[2.1,-2.1]","[-3.0,3.0]","[-2.0,2.0]", "GD in [1.0,1.0] ", "GD in [-2.1,2.1] "], iterations, title = "Starting in all points compared to GD")
	#plot_bonus_comparison(f, [all_w,all_w_2,all_w_gd], ["Newton Eta = 0.01", "Newton Eta = 0.1","Gradient Descent, Eta = 0.01"], iterations, title = "Comparison Newton's method with gradient descent method starting in (1,1)")
	wait()

ej1()
print("----------------------------------\n----------------------------------\n----------------------------------\n")
ej2()
print("----------------------------------\n----------------------------------\n----------------------------------\n")
bonus()