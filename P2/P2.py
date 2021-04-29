#!/usr/bin/env python
# coding: utf-8

# In[9]:


# -*- coding: utf-8 -*-
"""
TRABAJO 2
Nombre Estudiante: Francisco Javier Sáez Maldonado
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


SAVE = False

# Fijamos la semilla
np.random.seed(1)

def wait():
    input("\n--- Pulsar tecla para continuar ---\n")
    
def to_numpy(func):
  """Decorador para convertir funciones a versión NumPy"""
  def numpy_func(w):
    return func(*w)

  return numpy_func

def generate_uniform_data(N, dim, ran):
    return np.random.uniform(ran[0],ran[1],(N,dim))

def generate_gaussian_data(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0])) 
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    
    return out


def generate_line(interval):
    points = np.random.uniform(interval[0], interval[1], size=(2, 2))
    x1,x2 = points[0,0],points[1,0]
    y1,y2 = points[0,1],points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b

def separador_ejercicio(ej):
    print("---------------------------------")
    print("\t {}".format(ej))
    print("---------------------------------")
    
def scatter_multiple(x,y = None,ws = None,labels = None,reg_titles = None ,xlabel_title = None , ylabel_title = None, title = ""):
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
    xmin, xmax = np.min(1.05*x[:, 1]), np.max(1.05*x[:, 1])
    ymin, ymax = np.min(1.05*x[:,2]), np.max(1.05*x[:,2])
    #ax.set_xlim(xmin, xmax)
    #ax.set_ylim(1.1*np.min(x[:, 2]), np.max(1.1*x[:, 2]))
    plt.xlim(xmin - 0.1, xmax + 0.1)
    plt.ylim(ymin - 0.1, ymax + 0.1)
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
            ax.scatter(class_points[:, 1],
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

            if reg_titles is None:
            # Plot regression results
                for w in ws:
                # Linear regression
                    if len(w) == 3:
                        ax.plot(x, (-w[1]*x - w[0])/w[2])
            else:
                for w,a in zip(ws,reg_titles):
                    # Linear regression
                    if len(w) == 3:
                        ax.plot(x, (-w[1]*x - w[0])/w[2],label=a)

    
    if y is not None or ws is not None:
        ax.legend()
  
    ax.set_title(title)

    if SAVE:
        plt.savefig("media/"+title+".pdf")
    plt.show() 


# In[10]:


def scatter(X, y = None, fun = None, title='Gráfico', axis_labels = ['x','y'],regions = False,label = "Clasificador",name = "def-name"):
    # Init graphic
    fig = plt.figure(figsize = (8, 6))
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    plt.title(title)
    cmap = ListedColormap(['r', 'green'])
    
    # Set Plot limits
    xmin, xmax = np.min(X[:, 1]), np.max(X[:, 1])
    ymin, ymax = np.min(X[:, 2]), np.max(X[:, 2])
    
    scale_x = (xmax - xmin) * 0.01
    scale_y = (ymax - ymin) * 0.01
    plt.xlim(xmin - scale_x, xmax + scale_x)
    plt.ylim(ymin - scale_y, ymax + scale_y)
    
    # Labels
    if y is None:
        c = [1 for _ in range(len(X))]
    else:
        c = y
        
    # Plot points
    scatter = plt.scatter(X[:, 1], X[:, 2],s = 60,cmap = cmap, c = c,marker='.')
    # If we got labels, create label legend
    if y is not None:
        legend1 = plt.legend(
            *scatter.legend_elements(),
            title = "Clases",
            loc = "upper right")
    
    # If function is given
    if fun is not None:
        #Generate predictions grid
        xs, ys = np.meshgrid(np.linspace(xmin - scale_x, xmax + scale_x, 100),
                    np.linspace(ymin - scale_y, ymax + scale_y, 100))
        zs = fun(xs,ys)
        
        # Draw regions divided by fun
        if regions:
            reg = plt.contourf(xs,ys,zs,cmap = cmap, levels = 0,alpha = 0.1)
            #fig.colorbar(reg,aspect = 30,label = "f(x,y)")
            
        # Draw fun
        plt.contour(xs,ys,zs,levels = [0],colors = ['blue']).collections[0].set_label(label)
        # Add legend
        plt.legend()
        
        # Insert function legend on the plot
        if label is not None:
            plt.legend(loc = "lower right")
    
    # Insert label functions on the plot
    if y is not None:
        plt.gca().add_artist(legend1)
        
    
    # Save, only for report purposes.
    if SAVE:
        plt.savefig("media/"+name+".pdf")
        
    plt.show(block = False)

  


# In[11]:


# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente

separador_ejercicio("Ejercicio 1.1")

x = generate_uniform_data(50, 2, [-50,50])

x = np.hstack((np.ones((50, 1)), x))
#def scatter(X, y = None, fz = None, title='Gráfico', axis_labels = ['x','y'],regions = False,label = "Clasificador"):
 
scatter(x,title= "50 datos generados usando la distribución uniforme", name = "50-datos-uniforme")

wait()


# In[12]:


x = generate_gaussian_data(50, 2, np.array([5,7]))
x = np.hstack((np.ones((50, 1)), x))
scatter(x,title= "50 datos generados usando la distribución gaussiana",name = "50-datos-gauss")


wait()


# In[5]:


###############################################################################
###############################################################################
###############################################################################


# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente

separador_ejercicio("Ejercicio 1.2")

# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def sign(x):
    if x >= 0:
        return 1
    return -1

# Función usada para el ejercicio 1
def f(x, y, a, b):
    return sign(y - a*x - b)

    

# Generate data
X = generate_uniform_data(100,2,[-50,50])
a,b = generate_line([-50,50])
y = [f(x[0],x[1],a,b) for x in X]


X = np.hstack((np.ones((100, 1)), X))
# Inline function using a and b obtained by generate_line
line = lambda x,y: y-a*x-b

scatter(X,y,line,title= "100 datos uniformes clasificados según el signo de la función",label = "y = {:0.3f}x + ({:0.3f})".format(a, b), name = "100-datos-separados")




wait()


# In[13]:



separador_ejercicio("Ejercicio 1.2-b")

# 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello
# Array con 10% de indices aleatorios para introducir ruido



# Function that generates noise on a percentage of the labels
def generate_noise(y,per = 0.1):
    
    y_copy = np.copy(y)
    
    for label in {-1,1}:
        # Get index of labels
        idx = np.where(y==label)[0]
        # Random selection of percentage*length labels
        changes = np.random.choice(idx,int(per*len(idx)),replace=False)
        # Change labels
        y_copy[changes] = -label
        
    return y_copy

X = generate_uniform_data(100,2,[-50,50])
a,b = generate_line([-50,50])
y = np.array([f(x[0],x[1],a,b) for x in X])

y_noise = generate_noise(y,0.1)  



X = np.hstack((np.ones((100, 1)), X))
# Inline function using a and b obtained by generate_line
line = lambda x,y: y-a*x-b

scatter(X,y_noise,line,title= "100 datos con ruido clasificados según el signo de la función",label = "y = {:0.3f}x + ({:0.3f})".format(a, b), name = "100-datos-ruido")
wait()


# In[14]:


###############################################################################
###############################################################################
###############################################################################

separador_ejercicio("Ejercicio 1.3")

# EJERCICIO 1.3: Supongamos ahora que las siguientes funciones definen la frontera de clasificación de los puntos de la muestra en lugar de una recta
functions = [line,
             lambda x,y: (x-10)**2 + (y-20)**2 -400,
             lambda x,y: 0.5*(x+10)**2 + (y-20)**2 -400,
             lambda x,y: 0.5*(x-10)**2 - (y+20)**2 - 400,
             lambda x,y: y-20*x**2 -5*x + 3]
names = ["y = {:0.3f}x + ({:0.3f})".format(a,b),
         "f(x,y) = (x-10)^2 + (y-20)^2 -400 = 0",
         "f(x,y) = 0.5(x+10)^2 + (y-20)^2 -400 = 0",
         "f(x,y) = 0.5(x-10)^2 - (y+20)^2 - 400 = 0",
         "f(x,y) = y-20x^2 -5x + 3 = 0"]

geo_names = ["Recta","Elipse 1", "Elipse 2", "Hipérbola", "Parábola"]


for func,name,geo_name in zip(functions,names,geo_names):
    scatter(X,y_noise,func,regions = True,title= "Frontera de clasificación de {}".format(geo_name),label = name,name = geo_name)
    wait()
    



# In[16]:



from sklearn.metrics import accuracy_score,balanced_accuracy_score
for clas,name,geo_name in zip(functions,names,geo_names):
    y_predicted = [sign(clas(x[1],x[2])) for x in X]
    score = 100*accuracy_score(y_noise,y_predicted)
    balanced_score = 100*balanced_accuracy_score(y_noise,y_predicted)
    print("Usando la {} dada por {}, las precisiones son:".format(geo_name,name))
    print("\tAccuracy : {}".format(score))
    print("\tBalanced Accuracy : {}".format(balanced_score))


# In[17]:


###############################################################################
###############################################################################
###############################################################################

separador_ejercicio("Ejercicio 2-a: Perceptron")

MAX_ITER = 1000

def PLA(X,y,initial_w,max_iter = MAX_ITER):
    
    w_t = initial_w.copy()
    all_w = [w_t]
    
    iterations = 0
    # Repeat max_it times
    for it in range(0,max_iter):
        
        # Control if w has changed
        update = False
        # Iteration in all the data
        for x,label in zip(X,y):
            
            # Predict using linear model
            pred = sign(x.dot(w_t))
            # Check label and update weights if neccesary
            
            if pred != label:
                w_t = w_t + label*x
                update = True
                
        if not update:
            break
                
        all_w.append(w_t.copy())
        iterations +=1
            

    
    return w_t,all_w,iterations

def predict(X,y,w):
    y_pred = [sign(x.dot(w)) for x in X]
    score = 100*accuracy_score(y,y_pred)
    return score

def plot_pla_evolution(X,y,all_w,f_title ,name="media/fail.pdf"):
    """ Function that plots the value of a function f in each iteration in R^2 """

    # Create figure
    fig = plt.figure()
    ax = ax = fig.add_subplot(1, 1, 1)
    xs = np.array([i for i in range(all_w.shape[0])])
    preds = np.array([[sign(x.dot(w)) for x in X] for w in all_w])
    ys = np.array([accuracy_score(y,y_pred) for y_pred in preds])
    # Plot f values
    ax.plot(xs,ys, linestyle='-', marker='.', color='b',linewidth=0.5)
    ax.grid(True)
    ax.set_ylabel(f_title)
    # Save fig for memory purposes
    if SAVE:
        plt.savefig("media/"+name+".pdf")
    plt.show()

    


# In[18]:



# [0,0,0] vector
w,all_w,it = PLA(X,y,[0,0,0])
print("Pesos obtenidos para peso inicial [0,0,0]: {}".format(w))
print("Iteraciones para peso inicial [0,0,0]: {}".format(it))
print("Accuracy = {}".format(predict(X,y,w)))
print("Realizando experimento de 10 inicializaciones de pesos aleatorias ....")

# Random inicializations
initial_weights = []
iterations = []
v_accuracy = []
for i in range(0,10):
    w0 = np.random.uniform(0,1,3)
    initial_weights.append(w0)
    #print("Pesos Iniciales: {}".format(w0))
    w_t,all_w,it = PLA(X,y,w0)
    iterations.append(it)
    v_accuracy.append(predict(X,y,w_t))
    

    
print('Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.array(iterations))))
print('Valor medio de accuracy en pesos aleatorios: {}'.format(np.mean(np.array(v_accuracy))))
print('La desviación típica en las iteraciones ha sido: {}, pues las iteraciones son: {}'.format(np.std(iterations),iterations))


# In[19]:


# Ahora con los datos del ejercicio 1.2.b
separador_ejercicio("Ejercicio 2-a-2 : Datos con ruido")

# [0,0,0] vector
w,all_w_zero,it = PLA(X,y_noise,[0,0,0])
print("Iteraciones para peso inicial [0,0,0]: {}".format(it))
print("Accuracy = {}".format(predict(X,y_noise,w)))

# Random inicializations
initial_weights = []
iterations = []
v_accuracy = []
print("Realizando experimento de 10 inicializaciones de pesos aleatorias en datos con ruido ....")
for i in range(0,10):
    w0 = np.random.uniform(0,1,3)
    initial_weights.append(w0)
    #print("Pesos Iniciales: {}".format(w0))
    w_t,all_w,it = PLA(X,y_noise,w0)
    iterations.append(it)
    v_accuracy.append(predict(X,y_noise,w_t))
    

    
   
print('Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.array(iterations))))
print('Valor medio de accuracy en pesos aleatorios: {}'.format(np.mean(np.array(v_accuracy))))
print('La desviación típica en las iteraciones ha sido: {}, pues las iteraciones son: {}'.format(np.std(iterations),iterations))


# In[20]:


# plot_pla_evolution(X,y,all_w,f_title ,name="media/fail.pdf"):
plot_pla_evolution(X,y_noise,all_w = np.array(all_w_zero),f_title = "Acierto en PLA",name = "PLA-ruido")
wait()

# In[21]:


###############################################################################
###############################################################################
###############################################################################

# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT

separador_ejercicio("Ejercicio 3: Regresión logística")

def err(X, y, w):
    """Error cometido en regresión logística"""

    return np.mean(np.log(1 + np.exp(-y * X.dot(w))))

def d_err(x, y, w):
    """Gradiente del error de regresión logística"""
    return -y * x / (1 + np.exp(y * x.dot(w)))



def sgdRL(X,y,eta = 0.01,epsilon = 0.01):
    # Neccesary vars
    w_t = np.zeros(X.shape[1])
    index = np.arange(X.shape[0])
    it = 0
    converged = False
    
    while not converged:
        w = w_t.copy()
        
        index = np.random.permutation(index)
        for idx in index:
            w_t -= eta*d_err(X[idx],y[idx],w_t)
            
        if np.linalg.norm(w_t - w) < epsilon:
            converged = True

        it +=1

    return w,it


    
def experiment(N_experiments = 100,N_train = 100,N_test = 1000, epsilon = 0.01):
    
    avg_iterations = []
    avg_error = []
    avg_error_train = []
    all_w = []
    for i in range(0,N_experiments):
        # Generate data and line
        a,b = generate_line([0,2])
        X = generate_uniform_data(N_train,2,[0,2])
        # Create tags using the line
        y = np.array([f(x[0],x[1],a,b) for x in X])
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        # Find w using sgdRL
        w,iterations = sgdRL(X,y,epsilon = epsilon)
        all_w.append(w)
        # Create data for eout
        X_test = generate_uniform_data(N_test,2,[0,2])
        y_test = np.array([f(x[0],x[1],a,b) for x in X_test])
        X_test = np.hstack((np.ones((X_test.shape[0],1)),X_test))
        # Evaluate w on test
        error = err(X_test,y_test,w)
        error_train = err(X,y,w)
        
        # Append results
        avg_iterations.append(iterations)
        avg_error.append(error)
        avg_error_train.append(error_train)
        
        # Save 
        if N_experiments == 1:
            scatter_multiple(X,y,ws = [[-b,-a,1],w],reg_titles = ["Original line", "LR line"], title = "LR vs Original-Train-eps{}".format(epsilon) )
            wait()
            print("El error en la muestra es de {}".format(avg_error_train))
            # def scatter_multiple(x,y = None,ws = None,labels = None,reg_titles = None ,xlabel_title = None , ylabel_title = None, title = ""):
            scatter_multiple(X_test,y_test,ws = [[-b,-a,1],w],reg_titles = ["Original line", "LR line"], title = "LR vs Original-eps{}".format(epsilon) )
            wait()

    
        
    return np.array(avg_iterations),np.array(avg_error)



# In[22]:


# Experimento único epsilon  = 0.01
print("Ejecución de SGDRL con epsilon = 0.01, como se indica")
N_experiments = 1
iters,error = experiment(N_experiments = N_experiments)
print("Average results in {} experiments:".format(N_experiments))
print("\t Iterations: {}".format(np.mean(iters)))
print("\t Error E_out: {}".format(np.mean(error)))


# In[23]:


# Experimento único epsilon  = 0.005
print("Ejecución de SGDRL con epsilon = 0.005, como experimento")
N_experiments = 1
iters,error = experiment(N_experiments = N_experiments,epsilon = 0.005)
print("Average results in {} experiments:".format(N_experiments))
print("\t Iterations: {}".format(np.mean(iters)))
print("\t Error E_out: {}".format(np.mean(error)))


# In[24]:


# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).
print("Ejecutando 100 experimentos de regresión logística y calculando medias...")
N_experiments = 100
iters,error = experiment(N_experiments = N_experiments)
print("Average results in {} experiments:".format(N_experiments))
print("\t Iterations: {}".format(np.mean(iters)))
print("\t Error out sample: {}".format(np.mean(error)))


# In[133]:


###############################################################################
###############################################################################
###############################################################################
#BONUS: Clasificación de Dígitos

separador_ejercicio("BONUS")


# Funcion para leer los datos
def readData(file_x, file_y, digits, labels):
    # Leemos los ficheros
    datax = np.load(file_x)
    datay = np.load(file_y)
    y = []
    x = []
    # Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
    for i in range(0,datay.size):
        if datay[i] == digits[0] or datay[i] == digits[1]:
            if datay[i] == digits[0]:
                y.append(labels[0])
            else:
                y.append(labels[1])
            x.append(np.array([1, datax[i][0], datax[i][1]]))
            
    x = np.array(x, np.float64)
    y = np.array(y, np.float64)
    
    return x, y

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])


#mostramos los datos
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.legend()
if SAVE:
    plt.savefig("media/"+"Digitos-Manuscritos-Train"+".pdf")
plt.show()
wait()

fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.legend()
if SAVE:
    plt.savefig("media/"+"Digitos-Manuscritos-Test"+".pdf")
plt.show()

wait()


# In[134]:


#LINEAR REGRESSION FOR CLASSIFICATION 

def pseudoinverse(x,y):
    return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)

def error(X, y, w):
    incorrect = [sign(x.dot(w)) for x in X] != y
    return np.mean(incorrect)

w_pseudo = pseudoinverse(x,y)
scatter_multiple(x,y,ws = [w_pseudo],reg_titles = ["Pseudoinverse"], title = "Pseudoinverse-train") 
wait()
scatter_multiple(x_test,y_test,ws = [w_pseudo],reg_titles = ["Pseudoinverse"], title = "Pseudoinverse-test") 
wait()

print("El porcentaje de error de pseudoinversa es:")
print("\t Train: {:0.5f}%".format(error(x,y,w_pseudo)*100))
print("\t Test: {:0.5f}%".format(error(x_test,y_test,w_pseudo)*100))


# In[135]:


MAX_ITER = 1000


    

def PLA_Pocket(X,y,initial_w,max_iter = MAX_ITER):
    
    w_t = initial_w.copy()
    best_w = w_t.copy()
    err_best = error(X,y,best_w)
    
    # Repeat max_it times
    for it in range(0,max_iter):
        w_old = w_t.copy()
        
        # Iteration in all the data
        for el,label in zip(X,y):
            # Check if label != prediction and update weights if neccesary
            if sign(el.dot(w_t)) != label:
                w_t = w_t + label*el
                
        curr_error = error(X,y,w_t)
        if curr_error < err_best:
            err_best = curr_error
            best_w = w_t.copy()
        # Early stop    
        if np.all(w_old == w_t):
            break
                

        all_w.append(w_t.copy())

    return best_w,all_w


# In[140]:


#POCKET ALGORITHM
w_pla_rand,_ = PLA_Pocket(x,y,np.random.rand(3))
  

scatter_multiple(x,y,ws = [w_pla_rand],reg_titles = ["PLA-Pocket-Random-Ini"], title = "PLA-Pocket-Random-Ini-train") 
wait()
scatter_multiple(x_test,y_test,ws = [w_pla_rand],reg_titles = ["PLA-Pocket-Random-Ini"], title = "PLA-Pocket-Random-Ini-test") 
wait()

print("El porcentaje de error de PLA-Pocket pesos iniciales aleatorios es:")
print("\t Train: {:0.5f}%".format(error(x,y,w_pla_rand)*100))
print("\t Test: {:0.5f}%".format(error(x_test,y_test,w_pla_rand)*100))


# In[141]:


#POCKET ALGORITHM Starting with Pseudoinverse
w_pla_pseudo,_ = PLA_Pocket(x,y,w_pseudo)
  

scatter_multiple(x,y,ws = [w_pla_pseudo],reg_titles = ["PLA-Pocket-Pseudo-Ini"], title = "PLA-Pocket-Pseudo-Ini-train") 
wait()
scatter_multiple(x_test,y_test,ws = [w_pla_pseudo],reg_titles = ["PLA-Pocket-Pseudo-Ini"], title = "PLA-Pocket-Pseudo-Ini-test") 
wait()
 

print("El porcentaje de error de PLA-Pocket pesos iniciales dados por pseudoinversa es:")
print("\t Train: {:0.5f}%".format(error(x,y,w_pla_pseudo)*100))
print("\t Test: {:0.5f}%".format(error(x_test,y_test,w_pla_pseudo)*100))


# In[142]:


print("Pesos pseudoinversa: {}".format(w_pseudo))
print("Pesos PLA: {}".format(w_pla_rand))
print("Pesos Pseudo+PLA :{}".format(w_pla_pseudo))


# In[143]:


scatter_multiple(x,y,ws = [w_pseudo,w_pla_rand,w_pla_pseudo],reg_titles = ["Pseudoinversa", "PLA-Pocket-Rand-Ini","PLA-Pocket-Pseudo-Ini" ], title = "Comparacion-Bonus-train") 
wait()
scatter_multiple(x_test,y_test,ws = [w_pseudo,w_pla_rand,w_pla_pseudo],reg_titles = ["Pseudoinversa", "PLA-Pocket-Rand-Ini","PLA-Pocket-Pseudo-Ini"], title = "Comparacion-Bonus-test") 
wait()


# In[144]:


#COTA SOBRE EL ERROR

def Hoeffding_bound(in_sample_error,N,card_h = 2**(64*3),delta = 0.05):
    return in_sample_error + np.sqrt((1 / (2 * N)) * np.log((2 * card_h) / delta))


def VC_bound(in_sample_error, N, d_vc,delta = 0.05):
    return in_sample_error + np.sqrt((8 / N) * np.log(4 * ((2 * N) ** d_vc + 1) / delta))



# In[145]:


N = len(x)
hoeff_pseudo = Hoeffding_bound(error(x,y,w_pseudo),N)
hoeff_pla_rand =Hoeffding_bound(error(x,y,w_pla_rand),N)
hoeff_pla_pseudo = Hoeffding_bound(error(x,y,w_pla_pseudo),N)

print("Cota del error E_out por desigualdad de Hoeffding usando E_in según algoritmo:")
print("\t Pseudoinversa: E_out <= {:0.5f}".format(hoeff_pseudo))
print("\t PLA-Pocket-Random inicialización: E_out <= {:0.5f}".format(hoeff_pla_rand))
print("\t PLA-Pocket-Pseudoinversa inicialización: E_out <= {:0.5f}".format(hoeff_pla_pseudo))


# In[147]:


N = len(x_test)
card_h = 1
hoeff_pseudo_t = Hoeffding_bound(error(x_test,y_test,w_pseudo),N,card_h)
hoeff_pla_rand_t =Hoeffding_bound(error(x_test,y_test,w_pla_rand),N,card_h)
hoeff_pla_pseudo_t = Hoeffding_bound(error(x_test,y_test,w_pla_pseudo),N,card_h)

print("Cota del error E_out por desigualdad de Hoeffding usando E_test  según algoritmo:")
print("\t Pseudoinversa: E_out <= {:0.5f}".format(hoeff_pseudo_t))
print("\t PLA-Pocket-Random inicialización: E_out <= {:0.5f}".format(hoeff_pla_rand_t))
print("\t PLA-Pocket-Pseudoinversa inicialización: Cota <= {:0.5f}".format(hoeff_pla_pseudo_t))


# In[106]:


d_VC_PLA = 3
bound_pseudo = VC_bound(error(x,y,w_pseudo),N,d_VC_PLA)
bound_pla_rand = VC_bound(error(x,y,w_pla_rand),N,d_VC_PLA)
bound_pla_pseudo =VC_bound(error(x,y,w_pla_pseudo),N,d_VC_PLA)

print("Cota VC del error E_out según algoritmo:")
print("\t Pseudoinversa: E_out <= {:0.5f}".format(bound_pseudo))
print("\t PLA-Pocket-Random inicialización: E_out <= {:0.5f}".format(bound_pla_rand))
print("\t PLA-Pocket-Pseudoinversa inicialización: E_out <= {:0.5f}".format(bound_pla_pseudo))





