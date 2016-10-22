#Script for experiments with gradient descent

#-------- Script Imports ------------
#To make division easier
from __future__ import division
#For matrix and vector structures
import numpy as np
#For math operations
import math as math



#Runs the gradient descent with the given parameters
#Serves as a unified method
def run_gradient_descent(dim, fun, gradient, alpha, B_matrix, eps):
    """
        Parameters
        ----------
        dim : int
            The dimension of the vector that the function and gradient receive.
            The domain's dimension of the function we wish to minimize
        fun : function(numpy.vector)
            The function we wish to minimize
        gradient : functon(numpy.vector)
            A functon that receives a numpy.vecotr (real vector) and returns the
            gradient (as a numpy.vector) of the given function evaluated at
            the real number received as parameter
        alpha : function(float)
            A function that receives the prevoious alpha and returns the next
            one as a float. For the first iteriation, this function receives a
            None value.
        B_matrix : function(numpy.matrix)
            A function that receives the previous B^-1 matrix and returns the next
            one as a numpy.matrix. For the first iteriation, this function 
            receives a None value.
        eps : float
            The epsylon that serves as a stopping criteria for the algorithm
    """
    
    #Initial values
    #The first alpha and B matrix are initialized at None
    x = np.zeros(dim)
    B = None
    a = None
    
    #Becomes true when |f(x_n+1) - f(x_n)| < eps
    treshold = False
    
    while(not treshold):
        #Calculates the necesarry advancing parameters
        x_prev = x
        a = alpha(a)
        B = B_matrix(B)
        grad = gradient(x_prev)
        
        #Calcultaes the next value
        x = x_prev + alpha(a)*(-1)*B.dot(grad.T)
        
        #Checks the the treshold
        #print(np.linalg.norm(grad))
        treshold = np.linalg.norm(grad) < eps
    
    return [x, fun(x)]
#end of run_gradient_descent


#----------------------------------------------------------------------
#------------------------ Experiment Start ----------------------------
#----------------------------------------------------------------------
n = 500
m = 200
c = np.random.rand(1,n)
#Puts each a_j as a column of the following matrix
A = np.random.rand(n,m)
global_eps = 0.01

#Declares the global function and its gradient
def main_function(x):

    first_term = c.dot(x)
    second_term = (-1)*sum(map(lambda a: math.log(1 - a.dot(x)), A.T))
    third_term = (-1)*sum(map(lambda y: math.log(1 - y**2), x))
    
    return(first_term + second_term + third_term)
#end of main_function   

def main_gradient(x):
    first_term = np.array(c)
    #print(first_term)
    second_term = sum(map(lambda a: 1/(1 - a.dot(x)), A.T))*np.array(map(sum, A))
    #print(second_term)
    third_term = (-1)*np.array(map(lambda y: -2*y/(1 - y**2), x))
    #print(third_term)
    
    
    
    return(first_term + second_term + third_term)
#end of main_function

#--------------------------
#-----Gradient Descent-----
#--------------------------


#---- Constant ------
#Runs the constant example
def run_constant():
    alpha_const = 1
    B_const = np.identity(n)
    alpha_fun = lambda a: alpha_const
    B_matrix_fun = lambda b: B_const
    
    result = run_gradient_descent(dim = n,
                                  fun = main_function,
                                  gradient = main_gradient, 
                                  alpha = alpha_fun, 
                                  B_matrix = B_matrix_fun, 
                                  eps = global_eps )
    
    return result
#end of run_constant

print run_constant()

#print(run_gradient_descent(dim = 1,
#                          fun = lambda x: (x-5)**2,
#                          gradient = lambda x: 2*(x-5), 
#                          alpha = lambda a:0.001, 
#                          B_matrix = lambda b: np.matrix([[1]]), 
#                          eps = 0.001 ))    