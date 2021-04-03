# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: Adolfo Soto Werner
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from time import time

#La semilla se utiliza porque en el segundo ejercicio se utiliza mucho
#la función random para barajar resultados y para generar las muestras aleatorias
#y de este modo los resultados que se obtengan coincidiran con lo plasmado 
#en la memoria de la práctica
np.random.seed(1)

#Función E del segundo ejercicio
def E(u,v):
    return ((u**3)*math.e**(v-2)-2*(v**2)*(math.e**(-u)))**2

#Derivada parcial de E con respecto a u
def dEu(u,v):
    return 2*((u**3)*(math.e**(v-2))-2*(v**2)*(math.e**(-u)))*(3*(u**2)*(math.e**(v-2)+(math.e**(-u))*2*(v**2)))
    
#Derivada parcial de E con respecto a v
def dEv(u,v):
    return 2*((u**3)*(math.e**(v-2))-2*(v**2)*(math.e**(-u)))*((u**3)*(math.e**(v-2))-4*v*(math.e**(-u)))

#Gradiente de E
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])

#Funcion f del tercer ejercicio
def f(u,v):
     return (((u+2)**2)+2*((v-2)**2)+2*math.sin(2*math.pi*u)*math.sin(2*math.pi*v))

#Derivada parcial de f con respecto a u
def dfu(u,v):
    return 2*(2*math.pi*math.cos(2*math.pi*u)*math.sin(2*math.pi*v)+u+2)

#Derivada parcial de f con respecto a v
def dfv(u,v):
    return 4*(math.pi*math.sin(2*math.pi*u)*math.cos(2*math.pi*v)+v-2)

#Gradiente de f
def gradf(u,v):
    return np.array([dfu(u,v),dfv(u,v)])
    
#Calcula la norma de un vector
def module(x):
    return math.sqrt(x[0]**2+x[1]**2)
    

#Algoritmo de gradiente descendiente
def gradient_descent(init_pos, eta, maxIter,error,func,funcgrad):
    iterations=1
    #Se crea un vector numpy con el punto en el que se inicia la busqueda del mínimo
    coord=np.array([[init_pos[0],init_pos[1],func(init_pos[0],init_pos[1])]],dtype=np.float64)
    curr_pos=np.array([init_pos[0],init_pos[1]])
    #Se itera mientras no se alcance el límite de iteraciones establecido en maxIter
    #y mientras no se encuente una solución menor que el valor mínimo que hemos indicado en error
    if(error == '-'):
        while(iterations < maxIter):# and func(curr_pos[0],curr_pos[1]) > error):
            #Se actualiza la posición de la solución calculada
            curr_pos=curr_pos-eta*funcgrad(curr_pos[0],curr_pos[1])
            #Se añade al vector de coordenadas la solución calculada en esta iteración
            coord=np.append(coord,[[curr_pos[0],curr_pos[1],func(curr_pos[0],curr_pos[1])]],axis=0)
            #se incrementa el contador de iteraciones
            iterations+=1
    else:
        while(iterations < maxIter and func(curr_pos[0],curr_pos[1]) > error):
            #Se actualiza la posición de la solución calculada
            curr_pos=curr_pos-eta*funcgrad(curr_pos[0],curr_pos[1])
            #Se añade al vector de coordenadas la solución calculada en esta iteración
            coord=np.append(coord,[[curr_pos[0],curr_pos[1],func(curr_pos[0],curr_pos[1])]],axis=0)
            #se incrementa el contador de iteraciones
            iterations+=1
    #Devuelve la posición en la que se encuentra la solución, el numero de iteraciones
    #y el vector con las soluciones que se han ido calculando        
    return curr_pos, iterations,coord     

#Funciones para inicializar los parametros con los que trabajar en los ejercicios
#Los valores que se devuelven son de izquierda a derecha:
#1. Funcion con la que se va a trabajar
#2. Gradiente d ela función con la que se va a trabajar
#3. Ratio de aprendizaje (eta)
#4. Máximo permitido de iteraciones
#5. Error mínimo para el que consideramos aceptable una solución
#   En este caso al estar trajando con funciones este error es el
#   valor mas pequeño que queremos. Otro uso interesante para este valor 
#   Es como condicion de parada del algoritmo ya que si la norma del 
#   gradiente cae por debajo de este límite significa que se ha quedado
#   muy cerca de un mínimo local pero precisamente por ser tan pequeño el
#   gradiente las soluciones se estancaran en el entorno del mínimo local
#   sin llegar a alcanzarlo
#6. Coordenadas del punto de partida en la busqueda
def initE12():
    return E,gradE,0.1,50,1e-14,np.array([1,1])
def initE13a1():
    return f,gradf,0.01,50,'-',np.array([-1,1])
def initE13a2():
    return f,gradf,0.1,50,'-',np.array([-1,1])
def initE13b1():
    return f,gradf,0.01,50,1e-14,np.array([-0.5,-0.5])
def initE13b2():
    return f,gradf,0.01,50,1e-14,np.array([1,1])
def initE13b3():
    return f,gradf,0.01,50,1e-14,np.array([2,1])
def initE13b4():
    return f,gradf,0.01,50,1e-14,np.array([-2,1])
def initE13b5():
    return f,gradf,0.01,50,1e-14,np.array([-3,3])
#En este caso inicializamos un error máximo ya que se encuentra una
#solución factible en la primera itreación en la que el algoritmo
#se queda atascado las 50 iteraciones
def initE13b6():
    return f,gradf,0.01,50,1e-14,np.array([-2,2])

def draw(coord,func,w,fig,ax,it):
    #Extracción de la solución inicial generada
    cxi=coord[0,0]
    cyi=coord[0,1]
    czi=coord[0,2]
    #Última de las soluciones calculadas por el algoritmo
    cx=coord[1:it,0]
    cy=coord[1:it,1]
    cz=coord[1:it,2]
    
    #Se pinta la figura con un poco de margen entre el máximo y el mínimo
    #de los valores de las soluciones encontradas para que la superficie
    #se vea mejor siempre y cuando no se obtenga una solución aceptable
    #en la posición inicial
    if(cx.size!=0 and cy.size!=0 and cz.size!=0):
        restx=(np.max(cx)- np.min(cx))*1.3
        resty=(np.max(cy)- np.min(cy))*1.3
        x = np.linspace(np.min(cx)-restx, np.max(cx)+restx, 50)
        y = np.linspace(np.min(cy)-resty, np.max(cy)+resty, 50)
    else:
        #Este caso está hecho especialmente para el caso del apartado b del ejercicio 1.3
        #en el que se empieza en el punto [-2,2] para que en el gráfico se vea bien
        x = np.linspace(-4, 1, 50)
        y = np.linspace(-1, 4, 50)
        
    
    X, Y = np.meshgrid(x, y)

    #Tenemos que utilizar esta orden para que no haya problema cunado
    #pasamos vectores numpy a las funciones en lugar de valores individuales
    func2=np.vectorize(func)
    Z = func2(X, Y)
    min_point = np.array([w[0],w[1]])
    min_point_ = min_point[:, np.newaxis]
    #Ponemos en la gráfica el mínimo de la función con una estrella roja
    ax.plot(min_point_[0], min_point_[1], func2(min_point_[0], min_point_[1]), 'r*', markersize=10)
    #Se ponen todos las soluciones calculadas por el algoritmo como crcues verdes
    #siempre y cuando no se haya encontrado una solución en el punto inicial
    if(cx.size!=0 and cy.size!=0 and cz.size!=0):
       ax.plot(cx,cy,cz,'ko',markersize=3)
    #Se pone la solución inicial de la que parte el algoritmo como una bola roja
    ax.plot(cxi,cyi,czi,'bo',markersize=5)
    #Se pone en la gráfica la superficie que define la función en cuestión
    #con el parámetro alpha se define la opacidad de la misma para que las
    #distintas soluciones sean visibles sobre la misma
    ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet',alpha=0.2)
  
    
print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 2\n')
#Inicialización de los valores    
func,gradfunc,eta,maxIter,error2get,initial_point = initE12()
#Llamada al algoritmo 
w, it,coord = gradient_descent(initial_point,eta,maxIter,error2get,func,gradfunc)

print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')  Valor de la función: ', coord[it-1,2])

#Creamos la figura sobre la que incluiremos el gráfico 
anch,alt=plt.figaspect(3.)
fig = plt.figure(figsize=(anch+5,alt))
ax = fig.add_subplot(2,1,1,projection='3d')

#Usamos una función para pintar el gráfico
draw(coord,func,w,fig,ax,it)

#Añadimos título y nombramos los ejes
ax.set(title='E(u,v), eta=0.1 , init_pos=[1,1]')
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('E(u,v)')

ax=fig.add_subplot(2,1,2)
ax.set(title="Valores de la fucnión en cada iteración")
t1=np.arange(0,it,1,dtype='int')
ax.plot(t1,coord[t1,2],'-o')
plt.show()

input("\n--- Pulsar tecla para continuar al siguiente ejercicio ---\n")
print('Ejercicio 3 apartado a con eta=0.01\n')
func,gradfunc,eta,maxIter,error2get,initial_point = initE13a1()
w, it,coord = gradient_descent(initial_point,eta,maxIter,error2get,func,gradfunc)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')  Valor de la función: ', coord[it-1,2])
anch,alt=plt.figaspect(3.)
fig = plt.figure(figsize=(anch+5,alt))
ax = fig.add_subplot(2,1,1,projection='3d')
draw(coord,func,w,fig,ax,it)
ax.set(title='f(x,y), eta=0.01 , init_pos=[-1,1]')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')

ax=fig.add_subplot(2,1,2)
ax.set(title="Valores de la fucnión en cada iteración")
t1=np.arange(0,it,1,dtype='int')
ax.plot(t1,coord[t1,2],'-o')
plt.show()

input("\n--- Pulsar tecla para continuar al siguiente ejercicio ---\n")
print('Ejercicio 3 apartado a con eta = 0.1\n')  
func,gradfunc,eta,maxIter,error2get,initial_point = initE13a2()
w, it,coord = gradient_descent(initial_point,eta,maxIter,error2get,func,gradfunc)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')  Valor de la función: ', coord[it-1,2])
anch,alt=plt.figaspect(3.)
fig = plt.figure(figsize=(anch+5,alt))
ax = fig.add_subplot(2,1,1,projection='3d')
draw(coord,func,w,fig,ax,it)
ax.set(title='f(x,y), eta=0.1 , init_pos=[-1,1]')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')

ax=fig.add_subplot(2,1,2)
ax.set(title="Valores de la fucnión en cada iteración")
t1=np.arange(0,it,1,dtype='int')
ax.plot(t1,coord[t1,2],'-o')
plt.show()

input("\n--- Pulsar tecla para continuar al siguiente ejercicio ---\n")
print('Ejercicio 3 apartado b con eta = 0.01 y posición inicial [-0.5,-0.5]\n')
func,gradfunc,eta,maxIter,error2get,initial_point = initE13b1()
w, it,coord = gradient_descent(initial_point,eta,maxIter,error2get,func,gradfunc)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')  Valor de la función: ', coord[it-1,2])
anch,alt=plt.figaspect(3.)
fig = plt.figure(figsize=(anch+5,alt))
ax = fig.add_subplot(2,1,1,projection='3d')
draw(coord,func,w,fig,ax,it)
ax.set(title='f(x,y), eta=0.01 , init_pos=[-0.5,-0.5]')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')

ax=fig.add_subplot(2,1,2)
ax.set(title="Valores de la fucnión en cada iteración")
t1=np.arange(0,it,1,dtype='int')
ax.plot(t1,coord[t1,2],'-o')
plt.show()


input("\n--- Pulsar tecla para continuar al siguiente ejercicio ---\n")
print('Ejercicio 3 apartado b con eta = 0.01 y posición inicial [1,1]\n')
func,gradfunc,eta,maxIter,error2get,initial_point = initE13b2()
w, it,coord = gradient_descent(initial_point,eta,maxIter,error2get,func,gradfunc)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')  Valor de la función: ', coord[it-1,2])
anch,alt=plt.figaspect(3.)
fig = plt.figure(figsize=(anch+5,alt))
ax = fig.add_subplot(2,1,1,projection='3d')
draw(coord,func,w,fig,ax,it)
ax.set(title='f(x,y), eta=0.01 , init_pos=[1,1]')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')

ax=fig.add_subplot(2,1,2)
ax.set(title="Valores de la fucnión en cada iteración")
t1=np.arange(0,it,1,dtype='int')
ax.plot(t1,coord[t1,2],'-o')
plt.show()

input("\n--- Pulsar tecla para continuar al siguiente ejercicio ---\n")
print('Ejercicio 3 apartado b con eta = 0.01 y posición inicial [2,1]\n')
func,gradfunc,eta,maxIter,error2get,initial_point = initE13b3()
w, it,coord = gradient_descent(initial_point,eta,maxIter,error2get,func,gradfunc)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')  Valor de la función: ', coord[it-1,2])
anch,alt=plt.figaspect(3.)
fig = plt.figure(figsize=(anch+5,alt))
ax = fig.add_subplot(2,1,1,projection='3d')
draw(coord,func,w,fig,ax,it)
ax.set(title='f(x,y), eta=0.01 , init_pos=[2,1]')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')

ax=fig.add_subplot(2,1,2)
ax.set(title="Valores de la fucnión en cada iteración")
t1=np.arange(0,it,1,dtype='int')
ax.plot(t1,coord[t1,2],'-o')
plt.show()

input("\n--- Pulsar tecla para continuar al siguiente ejercicio ---\n")
print('Ejercicio 3 apartado b con eta = 0.01 y posición inicial [-2,1]\n')
func,gradfunc,eta,maxIter,error2get,initial_point = initE13b4()
w, it,coord = gradient_descent(initial_point,eta,maxIter,error2get,func,gradfunc)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')  Valor de la función: ', coord[it-1,2])
anch,alt=plt.figaspect(3.)
fig = plt.figure(figsize=(anch+5,alt))
ax = fig.add_subplot(2,1,1,projection='3d')
draw(coord,func,w,fig,ax,it)
ax.set(title='f(x,y), eta=0.01 , init_pos=[-2,1]')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')

ax=fig.add_subplot(2,1,2)
ax.set(title="Valores de la fucnión en cada iteración")
t1=np.arange(0,it,1,dtype='int')
ax.plot(t1,coord[t1,2],'-o')
plt.show()

input("\n--- Pulsar tecla para continuar al siguiente ejercicio ---\n")
print('Ejercicio 3 apartado b con eta = 0.01 y posición inicial [-3,3]\n')
func,gradfunc,eta,maxIter,error2get,initial_point = initE13b5()
w, it,coord = gradient_descent(initial_point,eta,maxIter,error2get,func,gradfunc)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')  Valor de la función: ', coord[it-1,2])
anch,alt=plt.figaspect(3.)
fig = plt.figure(figsize=(anch+5,alt))
ax = fig.add_subplot(2,1,1,projection='3d')
draw(coord,func,w,fig,ax,it)
ax.set(title='f(x,y), eta=0.01 , init_pos=[-3,3]')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')

ax=fig.add_subplot(2,1,2)
ax.set(title="Valores de la fucnión en cada iteración")
t1=np.arange(0,it,1,dtype='int')
ax.plot(t1,coord[t1,2],'-o')
plt.show()

input("\n--- Pulsar tecla para continuar al siguiente ejercicio ---\n")
print('Ejercicio 3 apartado b con eta = 0.01 y posición inicial [-2,2]\n')
func,gradfunc,eta,maxIter,error2get,initial_point = initE13b6()
w, it,coord = gradient_descent(initial_point,eta,maxIter,error2get,func,gradfunc)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')  Valor de la función: ', coord[it-1,2])
anch,alt=plt.figaspect(3.)
fig = plt.figure(figsize=(anch+5,alt))
ax = fig.add_subplot(2,1,1,projection='3d')
draw(coord,func,w,fig,ax,it)
ax.set(title='f(x,y), eta=0.01 , init_pos=[-2,2]')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')

ax=fig.add_subplot(2,1,2)
ax.set(title="Valores de la fucnión en cada iteración")
t1=np.arange(0,it,1,dtype='int')
ax.plot(t1,coord[t1,2],'-o')
plt.show()

input("\n--- Pulsar tecla para continuar al siguiente ejercicio ---\n")

print('EJERCICIO SOBRE REGRESION LINEAL\n')
print('Ejercicio 2.1')

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
    return (np.linalg.norm(x.dot(w) - y)**2)/len(x)

# Derivada del error
def dErr(x,y,w):
    
    return 2*(x.T.dot(x.dot(w) - y))/len(x)

# Gradiente Descendente Estocastico, recibe como parámetros el vector de
# características, el vector de etiquetas y el tamaño de los minibatches
def sgd(x,y,eta,b_size):
    epochs=0
    #Creamos el vector de pesos inicializado a 0
    w = np.zeros_like(x[0,:])
    
    #Barajamos los datos
    x_shuf,y_shuf=shuffle_two(x,y)
    #Creasmos el primer minibatch con los datos barajados
    x_batch,y_batch  = minibatches(x_shuf,y_shuf,b_size)
    data=len(x_shuf)
    #En cada iteración de este bucle se generan todos los minibatches 
    #de b_size posibles en el conjunto de datos
    while(epochs<50):
        #En cada iteración de este bucle se genera un minibatch
        while(data>b_size):
            data-=b_size
            #Actualizamos los pesos
            w=w-eta*dErr(x_batch,y_batch,w)
            #Extraemos un nuevo minibatch
            x_batch,y_batch  = minibatches(x_shuf,y_shuf,b_size)
        epochs+=1
        #print("E: ",epochs,"W: ",w, "Error: ",Err(x, y, w))
        #Cuando se han generado todos los minibatches posibles se
        #vuelven a barajar los datos para poder volver a crear los minibatches
        x_shuf,y_shuf=shuffle_two(x,y)
        x_batch,y_batch  = minibatches(x_shuf,y_shuf,b_size)
        data=len(x_shuf)   
    return w

# Pseudoinversa	
def pseudoinverse(x,y):
    return (np.linalg.inv(x.T.dot(x)).dot(x.T)).dot(y)

#Funcion para barajar los datos junto con sus etiquetas
#creando una nueva columna en la matriz de los datos en la 
#que se introduce su etiqueta, se barajan las filas de la matriz
#y se vuelven a dividir en matriz de datos y vector de etiquetas
def shuffle_two(x,y):
    c=np.array(x)
    c = np.insert(c,len(x[0]),y,axis=1)
    np.random.shuffle(c)
    x_aux=np.array(c[:,0:len(x[0])])
    y_aux=np.array(c[:,len(x[0])])
    
    return x_aux, y_aux

#De los datos introducidos se extraen los size primeros vectores
#y se eleiminan del parámetro para poder llamar a esta funcion siempre
#que queden al emnos size filas en la matriz que se pasa como argumento
def minibatches(x,y,size):
    x_aux=x[0:size,:]
    y_aux=y[0:size]
    x=np.delete(x, (range(size)),axis=0)
    y=np.delete(y, (range(size)))    
    return x_aux, y_aux


# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

#Ejecución del algoritmo de gradiente descendiente estocástico
w = sgd(x,y,0.01,32)
print()
print ('Bondad del resultado para grad. descendente estocastico:')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

plt.scatter(x[:,1],x[:,2],c=y)
t=np.arange(0,0.6,0.01)
plt.plot(t,(-w[0]-w[1]*t)/w[2],"-",label="sgd")

#Ejecución del algoritmo de la pseudoinversa
w2=pseudoinverse(x,y)
plt.plot(t,(-w2[0]-w2[1]*t)/w2[2],"-",label="pseudoinversa")
print()
print ('Bondad del resultado para pseudoinversa:')
print ("Ein: ", Err(x,y,w2))
print ("Eout: ", Err(x_test, y_test, w2))
plt.legend()
plt.show()

input("\n--- Pulsar tecla para continuar al siguiente ejercicio ---\n")
print('Ejercicio 2.2')

# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
 	return np.random.uniform(-size,size,(N,d))

# Función para el ejercicio 2 adaptaa para que devuelva el
# vector de etiquetas
def f2(x1, x2):
    vect=np.sign(((x1-0.2)**2)+x2**2-0.6)
    for i in range(len(vect)):
        if vect[i]==0:
            vect[i]=1
    return vect
 
# Fucnión para asignar las etiquetas al conjunto de puntos que se simulan
# en el ejercicio 2 incluyendo el 10% de ruido
def asignarEtiquetas(datos):
    y=np.array(f2(datos[:,0],datos[:,1]))
    ruido=np.array([])
    for i in range(int((len(datos)*0.1))):
        k=np.random.randint(0,len(y))
        if(k not in ruido):
            ruido=np.append(ruido,k)
        else:
            i-=1
    for j in ruido:
        y[int(j)]=-y[int(j)]
    return y
    
#Creación de vectores para almacenar los errores
errorsin=np.array([])
errorsout=np.array([])

#Generación de la muestra aleatoria
muestra=simula_unif(1000,2,1)
plt.scatter(muestra[:,0],muestra[:,1])
plt.title("Muestra uniforme de puntos")
print("Apartado a)")
plt.show()
input("\n--- Pulsar tecla para continuar al siguiente apartado ---\n")
#Asignación de etiquetas
labels=asignarEtiquetas(muestra)
plt.scatter(muestra[:,0],muestra[:,1],c=labels)
plt.title("Muestra uniforme de puntos con etiquetas y ruido")
print("Apartado b)")
plt.show()
input("\n--- Pulsar tecla para continuar al siguiente apartado ---\n")
muestra=np.insert(muestra,0,1,axis=1)

#Cálculo de pesos
w = sgd(muestra,labels,0.01,32)

#Almacenamos el error cometido en la muestra en la
#que se han calculado los pesos
errorsin=np.append(errorsin,[Err(muestra,labels,w)])
print ('Apartado c)')
print('Bondad del resultado para gradiente descendente estocástico:')
print ("Ein: ", errorsin[0])

input("\n--- Pulsar tecla para continuar al siguiente apartado ---\n")
print("Apartado d)")
init=time()
maxIter=1000
for i in range(maxIter):
    print("\r"+"Porcentaje de iteraciones completadas: " + str(round((i/maxIter)*100,1)) +" %", end="")
    #Generamos una nueva muestra
    muestra=simula_unif(1000,2,1)
    labels=asignarEtiquetas(muestra)
    muestra=np.insert(muestra,0,1,axis=1)
    #Utilizamos la nueva muestra como conjunto de test
    #para los pesos calculados previamente
    errorsout=np.append(errorsout,Err(muestra,labels,w))
    #print("Eout: ",errorsout[i])
    #Como se ha calculado un primer Ein fuera del bucle el último
    #no se calcula para que haya exactamente 1000 cálculos de 
    #errores de cada tipo
    if(i!=maxIter-1):
        w = sgd(muestra,labels,0.01,32)
        #print("Iteration : ",i+1)
        errorsin=np.append(errorsin,Err(muestra,labels,w))
        #print("Ein: ",errorsin[i+1])
print("\rPorcentaje de iteraciones completadas: 100 %")
#Cálculo de errores medios    
avg_ein=errorsin.mean()
avg_eout=errorsout.mean()
end=time()
print("Ein medio: ",avg_ein)
print("Eout medio: ",avg_eout)
print("Tiempo transcurrido en ejecutar el algoritmo 1000 veces: ", end-init)
#Mostramos los datos junto con el ajuste lineal
plt.scatter(muestra[:,1],muestra[:,2],c=labels)
t=np.linspace(-1,1, 100)
plt.plot(t,(-w[0]-w[1]*t)/w[2],"-",color='red')
plt.title("Ajuste Lineal")
plt.ylim(-1,1)
plt.show()
input("\n--- Pulsar tecla para continuar al siguiente ejercicio ---\n")
print("Ejercicio 2.2")

init=time()
#Se vuelve a crear una nueva muestra aleatoria
muestra=simula_unif(1000,2,1)
labels=asignarEtiquetas(muestra)
muestra=np.insert(muestra,0,1,axis=1)

#Se crea la nueva matriz de datos que indica el enunciado
charvect=np.array(muestra)
charvect=np.insert(charvect,3,charvect[:,1]*charvect[:,2],axis=1)
charvect=np.insert(charvect,4,charvect[:,1]**2,axis=1)
charvect=np.insert(charvect,5,charvect[:,2]**2,axis=1)

#Se crean los vectores para almacenar los errores
errorsin=np.array([])
errorsout=np.array([])

#Ejecución del algoritmo
w = sgd(charvect,labels,0.01,32)

#Almacenamos el error cometido en la muestra en la
#que se han calculado los pesos
errorsin=np.append(errorsin,[Err(charvect,labels,w)])

#print ('Iteration : 0')
#print ("Ein: ", errorsin[0])
maxIter=1000
for i in range(maxIter):
    print("\r"+"Porcentaje de iteraciones completadas: " + str(round((i/maxIter)*100,1)) +" %", end="")
    #Generamos una nueva muestra
    muestra=simula_unif(1000,2,1)
    labels=asignarEtiquetas(muestra)
    muestra=np.insert(muestra,0,1,axis=1)
    #Completamos la amtriz de datos
    charvect=np.array(muestra)
    charvect=np.insert(charvect,3,charvect[:,1]*charvect[:,2],axis=1)
    charvect=np.insert(charvect,4,charvect[:,1]**2,axis=1)
    charvect=np.insert(charvect,5,charvect[:,2]**2,axis=1)
    #Utilizamos la nueva muestra como conjunto de test
    #para los pesos calculados previamente
    errorsout=np.append(errorsout,Err(charvect,labels,w))
   # print("Eout: ",errorsout[i])
    #Como se ha calculado un primer Ein fuera del bucle el último
    #no se calcula para que haya exactamente 1000 cálculos de 
    #errores de cada tipo
    if(i!=maxIter-1):
        w = sgd(charvect,labels,0.01,32)
        #print("Iteration : ",i+1)
        errorsin=np.append(errorsin,Err(charvect,labels,w))
        #print("Ein: ",errorsin[i+1])

print("\rPorcentaje de iteraciones completadas: 100 %")  
#Cálculo de errores medios    
avg_ein=errorsin.mean()
avg_eout=errorsout.mean()
end=time()

print("Ein medio: ",avg_ein)
print("Eout medio: ",avg_eout)
print("Tiempo transcurrido en ejecutar el algoritmo 1000 veces: ", end-init)
#Mostramos el resultado del ajuste con el nuevo vector de características
plt.scatter(muestra[:,1],muestra[:,2],c=labels)
X, Y = np.meshgrid(np.linspace(-1.2,1.2, 100), np.linspace(-1.2, 1.2, 100))
F = w[0] + w[1]*X + w[2]*Y + w[3]*X*Y + w[4]*X*X + w[5]*Y*Y
plt.contour(X, Y, F, [0],colors=['red'])
plt.title("Ajuste No Lineal")
plt.show()


def hessianf(u,v):
    return np.linalg.inv(np.array([[2-8*(math.pi**2)*math.sin(2*u*math.pi)*math.sin(2*math.pi*v),
                      8*(math.pi**2)*math.cos(2*u*math.pi)*math.cos(2*math.pi*v)],
                     [8*(math.pi**2)*math.cos(2*u*math.pi)*math.cos(2*math.pi*v),
                      4-8*(math.pi**2)*math.sin(2*u*math.pi)*math.sin(2*math.pi*v)]]))

def newton(init_pos, maxIter,error,func,funcgrad,hess):
    iterations=1
    #Se crea un vector numpy con el punto en el que se inicia la busqueda del mínimo
    coord=np.array([[init_pos[0],init_pos[1],func(init_pos[0],init_pos[1])]],dtype=np.float64)
    curr_pos=np.array([init_pos[0],init_pos[1]])
    #Se itera mientras no se alcance el límite de iteraciones establecido en maxIter
    #y mientras no se encuente una solución menor que el valor mínimo que hemos indicado en error
    if(error == '-'):
        while(iterations < maxIter):# and func(curr_pos[0],curr_pos[1]) > error):
            #Se actualiza la posición de la solución calculada
            curr_pos=curr_pos-np.matmul(hess(curr_pos[0],curr_pos[1]),(funcgrad(curr_pos[0],curr_pos[1])))
            #Se añade al vector de coordenadas la solución calculada en esta iteración
            coord=np.append(coord,[[curr_pos[0],curr_pos[1],func(curr_pos[0],curr_pos[1])]],axis=0)
            #se incrementa el contador de iteraciones
            iterations+=1
    else:
        while(iterations < maxIter and func(curr_pos[0],curr_pos[1]) > error):
            #Se actualiza la posición de la solución calculada
            curr_pos=curr_pos-np.matmul(hess(curr_pos[0],curr_pos[1]),funcgrad(curr_pos[0],curr_pos[1]))
            #Se añade al vector de coordenadas la solución calculada en esta iteración
            coord=np.append(coord,[[curr_pos[0],curr_pos[1],func(curr_pos[0],curr_pos[1])]],axis=0)
            #se incrementa el contador de iteraciones
            iterations+=1
    #Devuelve la posición en la que se encuentra la solución, el numero de iteraciones
    #y el vector con las soluciones que se han ido calculando        
    return curr_pos, iterations,coord 

def initE3a():
    return f,gradf,50,'-',np.array([-1,1])
def initE3b1():
    return f,gradf,50,1e-14,np.array([-0.5,-0.5])
def initE3b2():
    return f,gradf,50,1e-14,np.array([1,1])
def initE3b3():
    return f,gradf,50,1e-14,np.array([2,1])
def initE3b4():
    return f,gradf,50,1e-14,np.array([-2,1])
def initE3b5():
    return f,gradf,50,1e-14,np.array([-3,3])
#En este caso inicializamos un error máximo ya que se encuentra una
#solución factible en la primera itreación en la que el algoritmo
#se queda atascado las 50 iteraciones
def initE3b6():
    return f,gradf,50,1e-14,np.array([-2,2])
    

input("\n--- Pulsar tecla para continuar al siguiente ejercicio ---\n")
print('Ejercicio 3 apartado a con eta=0.01\n')
hess=hessianf
func,gradfunc,maxIter,error2get,initial_point = initE3a()
w, it,coord = newton(initial_point,maxIter,error2get,func,gradfunc,hess)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')  Valor de la función: ', coord[it-1,2])
anch,alt=plt.figaspect(3.)
fig = plt.figure(figsize=(anch+5,alt))
ax = fig.add_subplot(2,1,1,projection='3d')
draw(coord,func,w,fig,ax,it)
ax.set(title='f(x,y), eta=0.01 , init_pos=[-1,1]')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')

ax=fig.add_subplot(2,1,2)
ax.set(title="Valores de la fucnión en cada iteración")
t1=np.arange(0,it,1,dtype='int')
ax.plot(t1,coord[t1,2],'-o')
plt.show()

input("\n--- Pulsar tecla para continuar al siguiente ejercicio ---\n")
print('Ejercicio 3 apartado b con eta = 0.01 y posición inicial [-0.5,-0.5]\n')
func,gradfunc,maxIter,error2get,initial_point = initE3b1()
w, it,coord = newton(initial_point,maxIter,error2get,func,gradfunc,hess)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')  Valor de la función: ', coord[it-1,2])
anch,alt=plt.figaspect(3.)
fig = plt.figure(figsize=(anch+5,alt))
ax = fig.add_subplot(2,1,1,projection='3d')
draw(coord,func,w,fig,ax,it)
ax.set(title='f(x,y), eta=0.01 , init_pos=[-0.5,-0.5]')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')

ax=fig.add_subplot(2,1,2)
ax.set(title="Valores de la fucnión en cada iteración")
t1=np.arange(0,it,1,dtype='int')
ax.plot(t1,coord[t1,2],'-o')
plt.show()


input("\n--- Pulsar tecla para continuar al siguiente ejercicio ---\n")
print('Ejercicio 3 apartado b con eta = 0.01 y posición inicial [1,1]\n')
func,gradfunc,maxIter,error2get,initial_point = initE3b2()
w, it,coord = newton(initial_point,maxIter,error2get,func,gradfunc,hess)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')  Valor de la función: ', coord[it-1,2])
anch,alt=plt.figaspect(3.)
fig = plt.figure(figsize=(anch+5,alt))
ax = fig.add_subplot(2,1,1,projection='3d')
draw(coord,func,w,fig,ax,it)
ax.set(title='f(x,y), init_pos=[1,1]')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')

ax=fig.add_subplot(2,1,2)
ax.set(title="Valores de la fucnión en cada iteración")
t1=np.arange(0,it,1,dtype='int')
ax.plot(t1,coord[t1,2],'-o')
plt.show()

input("\n--- Pulsar tecla para continuar al siguiente ejercicio ---\n")
print('Ejercicio 3 apartado b con eta = 0.01 y posición inicial [2,1]\n')
func,gradfunc,maxIter,error2get,initial_point = initE3b3()
w, it,coord = newton(initial_point,maxIter,error2get,func,gradfunc,hess)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')  Valor de la función: ', coord[it-1,2])
anch,alt=plt.figaspect(3.)
fig = plt.figure(figsize=(anch+5,alt))
ax = fig.add_subplot(2,1,1,projection='3d')
draw(coord,func,w,fig,ax,it)
ax.set(title='f(x,y), init_pos=[2,1]')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')

ax=fig.add_subplot(2,1,2)
ax.set(title="Valores de la fucnión en cada iteración")
t1=np.arange(0,it,1,dtype='int')
ax.plot(t1,coord[t1,2],'-o')
plt.show()

input("\n--- Pulsar tecla para continuar al siguiente ejercicio ---\n")
print('Ejercicio 3 apartado b con eta = 0.01 y posición inicial [-2,1]\n')
func,gradfunc,maxIter,error2get,initial_point = initE3b4()
w, it,coord = newton(initial_point,maxIter,error2get,func,gradfunc,hess)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')  Valor de la función: ', coord[it-1,2])
anch,alt=plt.figaspect(3.)
fig = plt.figure(figsize=(anch+5,alt))
ax = fig.add_subplot(2,1,1,projection='3d')
draw(coord,func,w,fig,ax,it)
ax.set(title='f(x,y), init_pos=[-2,1]')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')

ax=fig.add_subplot(2,1,2)
ax.set(title="Valores de la fucnión en cada iteración")
t1=np.arange(0,it,1,dtype='int')
ax.plot(t1,coord[t1,2],'-o')
plt.show()

input("\n--- Pulsar tecla para continuar al siguiente ejercicio ---\n")
print('Ejercicio 3 apartado b con eta = 0.01 y posición inicial [-3,3]\n')
func,gradfunc,maxIter,error2get,initial_point = initE3b5()
w, it,coord = newton(initial_point,maxIter,error2get,func,gradfunc,hess)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')  Valor de la función: ', coord[it-1,2])
anch,alt=plt.figaspect(3.)
fig = plt.figure(figsize=(anch+5,alt))
ax = fig.add_subplot(2,1,1,projection='3d')
draw(coord,func,w,fig,ax,it)
ax.set(title='f(x,y), init_pos=[-3,3]')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')

ax=fig.add_subplot(2,1,2)
ax.set(title="Valores de la fucnión en cada iteración")
t1=np.arange(0,it,1,dtype='int')
ax.plot(t1,coord[t1,2],'-o')
plt.show()

input("\n--- Pulsar tecla para continuar al siguiente ejercicio ---\n")
print('Ejercicio 3 apartado b con eta = 0.01 y posición inicial [-2,2]\n')
func,gradfunc,maxIter,error2get,initial_point = initE3b6()
w, it,coord = newton(initial_point,maxIter,error2get,func,gradfunc,hess)
print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')  Valor de la función: ', coord[it-1,2])
anch,alt=plt.figaspect(3.)
fig = plt.figure(figsize=(anch+5,alt))
ax = fig.add_subplot(2,1,1,projection='3d')
draw(coord,func,w,fig,ax,it)
ax.set(title='f(x,y), init_pos=[-2,2]')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')

ax=fig.add_subplot(2,1,2)
ax.set(title="Valores de la fucnión en cada iteración")
t1=np.arange(0,it,1,dtype='int')
ax.plot(t1,coord[t1,2],'-o')
plt.show()