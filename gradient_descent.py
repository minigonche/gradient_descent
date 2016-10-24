#Script for experiments with gradient descent

#-------- Script Imports ------------
#To make division easier
from __future__ import division
#For matrix and vector structures
import numpy as np
#For math operations
import math as math

import sys

#NOTE: Vectors are assumed as matrix of dimension 1 x n
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
            A functon that receives a numpy.vecotr (real vector: x_k) and returns the
            gradient (as a numpy.vector) of the given function evaluated at
            the real number received as parameter
        alpha : function(numpy.vector, numpy.vector)
            A functon that receives two numpy.vecotors (real vectors: x_k and p_k ) and returns the
            next alpha step
        B_matrix : function(numpy.vector)
            A function that receives a numpy.vecotr (real vector) and returns 
            the next B^-1 np.matrix 
        eps : float
            The epsylon that serves as a stopping criteria for the algorithm
    """
    
    #Initial values
    #The first alpha and B matrix are initialized at None
    x = np.zeros((1,dim))
    B = None
    a = None
    p = None
    
    #printing variables
    treshold = False
    count = 1
    print_counter = 1
    
    #Becomes true when |f(x_n+1) - f(x_n)| < eps
    while(not treshold):
        #Calculates the necesarry advancing parameters
        x_prev = x
        
        B = B_matrix(x_prev)
        grad = gradient(x_prev)

        #Calcultaes the next value
        a = alpha(x_prev, p)
        p = (-1)*B.dot(grad.T).T
        
        x = x_prev + a*p
        
        #Checks the the treshold
        if count == print_counter:
            print(np.linalg.norm(grad))
            count = 0
        count = count + 1    
        treshold = np.linalg.norm(grad) < eps
    
    print(x)
    return [x, fun(x)]
#end of run_gradient_descent


#----------------------------------------------------------------------
#------------------------ Experiment Start ----------------------------
#----------------------------------------------------------------------
n = 100
m = 20
c = np.random.rand(1,n)
#Puts each a_j as a column of the following matrix
A = np.random.rand(n,m)
global_eps = 5

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
    #print(temp_vec.shape)

    second_term = np.array(map(lambda a_row:  a_row.dot(temp_vec), A)).T
    #print(second_term.shape)
    
    third_term = np.array(map(lambda y: 2*y/(1 - y**2), x))
    #print(third_term.shape)
    

    return(first_term + second_term + third_term)
#end of main_gradient

def main_hessian(x):
    
    #Calculates the common vector in each coordinate
    temp_vec = np.array(map(lambda a_column: 1/(1 - a_column.dot(x.T)), A.T)).T

    #There is probably a faster way to do this, but since for this case the
    # Hessian matrix corresponds to a symetric matrix:
    hessian = np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            row = np.matrix(A[i,]*A[j,])
            value = (-1)*np.dot(row,temp_vec.T)
            if(i == j):
                value = value + 2*(1+x[0,i]**2)/(1-x[0,i]**2)
            
            hessian[i,j] = value
            hessian[j,i] = value
            
    return(hessian)
#end of main_hessian


#--------------------------
#-----Gradient Descent-----
#--------------------------

#First declares the global backtracking method
def alpha_backtracking(x, p):
    #For teh first iteration
    if(p is None):
        return 1
        
    a = 1
    p = np.random.rand(1)[0]
    c = np.random.rand(1)[0]
    while(main_function(x + a*p) > main_function(x) + c*a*np.dot(main_gradient(x).T,p) ):
        a = p*a
    
    return a
# end of alpha_backtracking

#---- Constant ------
#Runs the constant example
def run_constant():
    alpha_const = 1
    B_const = np.identity(n)
    alpha_fun = lambda x: alpha_const
    B_matrix_fun = lambda x: B_const
    
    result = run_gradient_descent(dim = n,
                                  fun = main_function,
                                  gradient = main_gradient, 
                                  alpha = alpha_fun, 
                                  B_matrix = B_matrix_fun, 
                                  eps = global_eps )
    
    return result
#end of run_constant


#---- Newton ------
#Runs the constant example
def run_newton():
    alpha_const = 1
    alpha_fun = lambda x: alpha_const
    B_matrix_fun = lambda x: (-1)*np.linalg.inv(main_hessian(x))
    
    result = run_gradient_descent(dim = n,
                                  fun = main_function,
                                  gradient = main_gradient, 
                                  alpha = alpha_fun, 
                                  B_matrix = B_matrix_fun, 
                                  eps = global_eps )
    
    return result
#end of run_newton






#--------Excecutions -----------------------
print run_newton()

#print(run_gradient_descent(dim = 1,
#                          fun = lambda x: (x-5)**2,
#                          gradient = lambda x: 2*(x-5), 
#                          alpha = lambda a:0.001, 
#                          B_matrix = lambda b: np.matrix([[1]]), 
#                          eps = 0.001 ))    