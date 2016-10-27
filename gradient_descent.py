#Script for experiments with gradient descent

#----------------------------------------------------------------------
#------------------------ Script Imports ------------------------------
#----------------------------------------------------------------------
#To make division easier
from __future__ import division
#For matrix and vector structures
import numpy as np
#For math operations
import math as math
#For system requirements
import sys
#For time measurement
import time

#For Graphing. The methods export the grapgics on plotly, the user only needs
# to enter his/her username and api-key
import plotly.plotly as py
import plotly.graph_objs as go
py.sign_in('minigonche', '8cjqqmkb4o') 

#----------------------------------------------------------------------
#------------------------ Global Variables ----------------------------
#----------------------------------------------------------------------

n = 500
m = 200
c = np.random.rand(1,n)
#Puts each a_j as a column of the following matrix
A = np.random.rand(n,m)
#Global constant alpha
global_alpha = 0.001
#GLobal epsilon for treshold
global_eps = 0.001
#global difference measure for gradient
global_dif = 0.000001
#Measure how many iterations to print pogress
print_counter = 20


#Final experiment variables
#-----------------------------------------------
c_e = np.random.rand(1,2)
c_e[0,0] = 1
c_e[0,1] = 1

#Puts each a_j as a column of the following matrix
A_e = np.random.rand(2,4)
A_e[0,0] = 1
A_e[0,1] = -1
A_e[0,2] = 0
A_e[0,3] = 0

A_e[1,0] = 0
A_e[1,1] = 0
A_e[1,2] = 1
A_e[1,3] = -1

cube = 100

global_eps_e = 0.000001


#----------------------------------------------------------------------
#------------------------ Main Methods --------------------------------
#----------------------------------------------------------------------


#NOTE: Vectors are assumed as matrix of dimension 1 x n
#Runs the gradient descent with the given parameters
#Serves as a unified method
def run_gradient_descent(dim, fun, gradient, alpha, B_matrix, eps, inverse = True, initial = None):
    """
        Parameters
        ----------
        dim : int
            The dimension of the vector that the function and gradient receive.
            The domain's dimension of the function we wish to minimize
        fun : function(numpy.vector)
            The function we wish to minimize
        gradient : functon(numpy.vector)
            A functon that receives a numpy.vecotr (real vector: x_k) and 
            returns the gradient (as a numpy.vector) of the given function 
            evaluated at the real number received as parameter
        alpha : function(numpy.vector, numpy.vector)
            A functon that receives two numpy.vecotors (real vectors: 
            x_k and p_k ) and returns the next alpha step
        B_matrix : function(np.matrix, numpy.vector)
            A function that receives a numpy.matrix (the previous matrix) and 
            numpy.vecotr (real vector) and returns the next multiplication
            np.matrix 
        eps : float
            The epsylon that serves as a stopping criteria for the algorithm
        solve : boolean
            Indicates if the B_matrix method gives the B or the B^-1 matrix
        Initial : np:vector
            The initial vector. If None is received, then the procedure strarts
            at zero.
    """
    #Starts the timer
    start_time = time.time()

    #Initial values
    #The first alpha and B matrix are initialized at None
    
    x = initial
    if x is None:
        x = np.zeros((1,dim))
        
    x_last = np.zeros((1,dim))
    grad_last = np.zeros((1,dim))
    B = None
    a = None
    p = None
    
    #Treshold
    treshold = False
    
    #printing variables
    count = 1
    global_count = 0
    
    #Graphing variables
    x_variables = []
    function_values = []
    
    
    #Becomes true when |f(x_n+1) - f(x_n)| < eps
    while(not treshold):
        #Saves the Value
        x_variables.append(x)
        function_values.append(fun(x))
        
        #Calculates the necesarry advancing parameters
        x_actual = x
        
        B = B_matrix(B, x_actual,x_last)
        grad = gradient(x_actual)

        #Calcultaes the next value
        if inverse:
            p = (-1)*B.dot(grad.T).T
        else:
            p = (-1)*np.linalg.solve(B, grad.T).T
            
        
        #raw_input('espere')    
        
        a = alpha(x_actual, p)
        x = x_actual + a*p
        x_last = x_actual
        
        #Checks the the treshold
        treshold = np.linalg.norm(grad) < eps  or np.linalg.norm(grad - grad_last) < global_dif
        
        if count == print_counter:
            print(np.linalg.norm(grad))
            count = 0
        
        count = count + 1
        global_count = global_count +1
        grad_last = grad
        
    
    x_final = x
    value_final = fun(x)
    
    
    return [x_final, value_final, x_variables, function_values, global_count, time.time() - start_time]
#end of run_gradient_descent

#Graphing method
def plot_log(function_values, value_final):
    """
        Parameters
        ----------
        function_values : np.array
            An array of the value of the function at the given iteration
        value_final : float
            The mimimum value achieved in the optimization
    """
    #Graphs the plot
    dif = map(lambda y: math.log(y - value_final ),function_values)
    #Draws the initial trace
    trace = go.Scatter(x = range(len(dif)), y =  dif)
    
    #Export graph
    plot_url = py.plot([trace], auto_open=False)  
    
#end plot_log

#Graphing method
def plot(x_values, y_values):


    #Draws the initial trace
    trace = go.Scatter(x = x_values, y =  y_values)
    
    #Export graph
    plot_url = py.plot([trace], auto_open=False)  
    
#end plot_log


#----------------------------------------------------------------------
#------------------------ Experiment Start ----------------------------
#----------------------------------------------------------------------


#CENTRAL FUNCTION

#Declares the global function, its gradient and its Hessian
def main_function(x):
    
    first_term = c.dot(x.T)[0,0]
    second_term = (-1)*sum(map(lambda a: math.log(1 - a.dot(x.T)), A.T))
    third_term = (-1)*sum(map(lambda y: math.log(1 - y**2), x.T))

    return(first_term + second_term + third_term)
#end of main_function   

def main_gradient(x):
    first_term = np.array(c)
    #print(first_term.shape)
    
    #Calculates the common vector in each coordinate
    temp_vec = np.array(map(lambda a_column: 1/(1 - a_column.dot(x.T)), A.T))

    second_term = np.array(map(lambda a_row:  a_row.dot(temp_vec), A)).T
    
    third_term = np.array(map(lambda y: 2*y/(1 - y**2), x))
    

    return(first_term + second_term + third_term)
#end of main_gradient

def main_hessian(x):
    
    #Calculates the common vector in each coordinate
    temp_vec = np.array(map(lambda a_column: 1/(1 - a_column.dot(x.T))**2, A.T)).T


    #There is probably a faster way to do this, but since for this case the
    # Hessian matrix corresponds to a symetric matrix:
    hessian = np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            row = np.matrix(A[i,:]*A[j,:])
            value = np.dot(row,temp_vec.T)
            if(i == j):
                value = value + 2*(1+(x[0,i]**2))/((1-(x[0,i]**2))**2)

            hessian[i,j] = value
            hessian[j,i] = value
            
    return(hessian)
#end of main_hessian

#-----------------------------------
#EXPERIMENT FUNCTION


#Declares the global function, its gradient 
def exp_function(x):

    first_term = c_e.dot(x.T)[0,0]
    second_term = (-1)*sum(map(lambda a: math.log(cube - a.dot(x.T)), A_e.T))

    return(first_term + second_term )
#end of exp_function   

def exp_gradient(x):
    first_term = np.array(c_e)
    
    #Calculates the common vector in each coordinate
    temp_vec = np.array(map(lambda a_column: 1/(cube - a_column.dot(x.T)), A_e.T))
    second_term = np.array(map(lambda a_row:  a_row.dot(temp_vec), A_e)).T
    
    return(first_term + second_term )
#end of exp_gradient

#--------------------------
#-----Gradient Descent-----
#--------------------------

#First declares the global constant and backtracking method for alpha
#Calculates the max alpha given the restriction of the logarithms
def max_alpha():
    cons = c + np.matrix(sum(A.T))
    theta_1 = max(map(lambda a_column: -1/np.dot(a_column,cons.T) , A.T))
    theta_2 = min(map(lambda v: 1/math.fabs(v), cons.T))
    if(theta_2 < theta_1):
        raise ValueError('No suitable alphas exist')
    return theta_2/(1+0.01)

constant_alpha = max_alpha()

def alpha_constant(x, p):
    return constant_alpha
#end of  alpha_constant

def alpha_global(x, p):
    return global_alpha
#end of  alpha_global

def alpha_backtracking(x, p):
    #For the first iteration
    if(p is None):
        return global_alpha
    
    a = 1
    flag = True
    while(flag):
        try:
            main_function(x + a*p)
            flag = False
        except ValueError:
            a = a/2

    rho = 4/5
    c = 4/5
    while(main_function(x + a*p) > main_function(x) + c*a*np.dot(main_gradient(x),p.T) ):
        a = rho*a
    
    return a
# end of alpha_backtracking

#Alpha backtracking for experiment
def alpha_backtracking_exp(x, p):
    #For the first iteration
    if(p is None):
        return global_alpha_e
    
    a = 1
    flag = True
    while(flag):
        try:
            exp_function(x + a*p)
            flag = False
        except ValueError:
            a = a/2

    rho = 4/5
    c = 4/5
    while(exp_function(x + a*p) > exp_function(x) + c*a*np.dot(exp_gradient(x),p.T) ):
        a = rho*a
    
    return a
# end of alpha_backtracking

#-----------------------------------
#---- Constant ------

#Runs the constant example
def run_constant():

    B_const = np.identity(n)
    B_matrix_fun = lambda B, x, x_prev: B_const
    
    result = run_gradient_descent(dim = n,
                                  fun = main_function,
                                  gradient = main_gradient, 
                                  alpha = alpha_constant, 
                                  B_matrix = B_matrix_fun, 
                                  eps = global_eps )
    
    return result
#end of run_constant

#-----------------------------------
#---- Constant  Backtracking ------
#Runs the constant eith backtracking example
def run_constant_backtracking():
    B_const = np.identity(n)
    B_matrix_fun = lambda B, x, x_prev: B_const
    
    result = run_gradient_descent(dim = n,
                                  fun = main_function,
                                  gradient = main_gradient, 
                                  alpha = alpha_backtracking, 
                                  B_matrix = B_matrix_fun, 
                                  eps = global_eps )
    
    return result
#end of run_constant_backtracking


#-------------------------------------
#---- Newton ------
#Runs the constant example
def B_matrix_hessian(B, x, x_prev):
    return main_hessian(x)

def run_newton():
    
    alpha_newton = lambda a,p: 1
    
    result = run_gradient_descent(dim = n,
                                  fun = main_function,
                                  gradient = main_gradient, 
                                  alpha = alpha_newton, 
                                  B_matrix = B_matrix_hessian, 
                                  eps = global_eps,
                                  inverse = False)
    
    return result
#end of run_newton


#-------------------------------------
#---- Newton BFGS ------

condition_numbers = []

#Primero se declara la funcion que se encarga de
def BFGS(B, x_actual, x_last):
    if B is None:
        return np.identity(n) 

    
    #Calculates temporal next value
    grad = main_gradient(x_actual)
    p = (-1)*np.linalg.solve(B, grad.T).T
    x_next = x_actual + global_alpha*p
    
    #x_next = x_actual
    #x_actual = x_last
    
    s = x_next - x_actual
    y = main_gradient(x_next) - main_gradient(x_actual)
    first_term = B
    second_term = (-1)*np.dot(B.dot(s.T), s.dot(B))/(np.dot(s, B.dot(s.T)))
    third_term = np.dot(y.T,y)/np.dot(y, s.T)
    
    final_b = first_term + second_term + third_term
    
    condition_numbers.append(np.linalg.cond(final_b))
    
    return final_b
        

def run_BFGS():
    
    result = run_gradient_descent(dim = n,
                                  fun = main_function,
                                  gradient = main_gradient, 
                                  alpha = alpha_backtracking, 
                                  B_matrix = BFGS, 
                                  eps = global_eps,
                                  inverse = False)
    return result                                  
    

#-----------------------------------
#---- Final Experiment ------

#Runs the constant example
def run_experiment():
    
    
    x = np.zeros((1,2))
    x[0,0] = 25
    x[0,1] = 75

    B_const = np.identity(2)
    B_matrix_fun = lambda B, x, x_prev: B_const
    
    
    result = run_gradient_descent(dim = 2,
                                  fun = exp_function,
                                  gradient = exp_gradient, 
                                  alpha = alpha_backtracking_exp, 
                                  B_matrix = B_matrix_fun, 
                                  eps = global_eps_e, 
                                  initial = x)
    
    return result
#end of run_constant        

    
#----------------------------------------------------------------------
#------------------------ Excecutions ---------------------------------
#----------------------------------------------------------------------

#----------------------------------------------------------------------
#------------------------ Main Experiment -----------------------------
#----------------------------------------------------------------------

'''
#Runs the main experiment

print('Start Constant Gradient')
r_const =  run_constant()
print('Ok')
print('Numero de Iteraciones: ' + str(r_const[4]))
print('Tiempo: ' + str(r_const[5]))
print('Minimo: ' + str(r_const[1]))
print('------------------------------')
print('')
print('------------------------------')

print('Start Constant Gradient With Backtracking')
r_const_b =  run_constant_backtracking()
print('Ok')
print('Numero de Iteraciones: ' + str(r_const_b[4]))
print('Tiempo: ' + str(r_const_b[5]))
print('Minimo: ' + str(r_const_b[1]))
print('------------------------------')
print('')
print('------------------------------')


print('Start Newton')
r_newton =  run_newton()
print('Ok')
print('Numero de Iteraciones: ' + str(r_newton[4]))
print('Tiempo: ' + str(r_newton[5]))
print('Minimo: ' + str(r_newton[1]))
print('------------------------------')
print('')
print('------------------------------')

print('Start Quasi')
r_quasi =  run_BFGS()
print('Ok')
print('Numero de Iteraciones: ' + str(r_quasi[4]))
print('Tiempo: ' + str(r_quasi[5]))
print('Minimo: ' + str(r_quasi[1]))
print('------------------------------')
print('')
print('------------------------------')

'''

'''
#plot_log(resultado[3], resultado[1])

#Graphs the plot for log
dif = map(lambda y: math.log(y - r_const[1] ),r_const[3])
trace_1 = go.Scatter(x = range(len(dif)), y =  dif)

dif = map(lambda y: math.log(y - r_const_b[1] ),r_const_b[3])
trace_2 = go.Scatter(x = range(len(dif)), y =  dif)

dif = map(lambda y: math.log(y - r_newton[1] ),r_newton[3])
trace_3 = go.Scatter(x = range(len(dif)), y =  dif)

dif = map(lambda y: math.log(y - r_quasi[1] ),r_quasi[3])
trace_4 = go.Scatter(x = range(len(dif)), y =  dif)

#Export graph
plot_url = py.plot([trace_1,trace_2,trace_3,trace_4], auto_open=False) 

print('Grafica logaritmica hecha')

#Makes a bubble graph for the time and iterations

tam = 35
trace_1 = go.Scatter( x = [r_const[4]],
                    y = [r_const[5]],
                    marker = dict(color = ['red'], 
                                  size = [tam]), mode = 'markers')
                                  
trace_2 = go.Scatter( x = [r_const_b[4]],
                    y = [r_const_b[5]],
                    marker = dict(color = ['green'], 
                                  size = [tam]), mode = 'markers')
                                  
trace_3 = go.Scatter( x = [r_newton[4]],
                    y = [r_newton[5]],
                    marker = dict(color = ['blue'], 
                                  size = [tam]), mode = 'markers')

trace_4 = go.Scatter( x = [r_quasi[4]],
                    y = [r_quasi[5]],
                    marker = dict(color = ['orange'], 
                                  size = [tam],), mode = 'markers')                                  

plot_url = py.plot([trace_1,trace_2,trace_3,trace_4], auto_open=False) 

print('Grafica tiempo vs iteraciones')
    

sys.exit('Ok')

'''


#----------------------------------------------------------------------
#------------------------ Last Experiment -----------------------------
#----------------------------------------------------------------------



resultado =  run_experiment()
print('Fin de la ejecucion')
print('------------------------------')
print('Valor:' + str(resultado[1]))
print('Valor de X: ' + str(resultado[0]))
print('Numero de Iteraciones: ' + str(resultado[4]))
print('Tiempo: ' + str(resultado[5]))
print('------------------------------')
print(' ')


optimo = resultado[0]

area = range(-400,1)

x_v_1 = [-100]*201
y_v_1 = range(-100,101)

x_h_2 = range(-100,101)
y_h_2 = [100]*201

x_v_2 = [100]*201
y_v_2 = list(reversed(range(-100,101)))

x_h_1 = list(reversed(range(-100,101)))
y_h_1 = [-100]*201



x_area = x_v_1 + x_h_2 + x_v_2 + x_h_1 
y_area = y_v_1 + y_h_2 + y_v_2 + y_h_1


x_values = map(lambda v: v[0,0], resultado[2])
y_values = map(lambda v: v[0,1], resultado[2])


trace1 = go.Scatter(x = x_area , y =  y_area)
trace2 = go.Scatter(x = x_values, y =  y_values)
trace3 = go.Scatter(x = [optimo[0,0]], y = [optimo[0,1]])
trace4 = go.Scatter(x = [25], y = [75])
    
#Export graph
plot_url = py.plot([trace1,trace2,trace3,trace4], auto_open=False)  

print(resultado[4])


sys.exit('Ok')



