#Script for experiments with gradient descent

#-------- Script Imports ------------
#For matrix and vector structures
import numpy as np
#For math operations
import math as math


#Runs the gradient descent with the given parameters
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
            A function that receives the previous B^^-1 matrix and returns the next
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
    
    while(!treshold):
        x_prev = x
        a = alpha(a)
        B = B_matrix(B)
        treshold
        x = x_prev + alpha(a)*(-1)*
        
        
        
        treshold = math.fabs(fun(x) - fun(x_prev)) < eps
        
    
    
    
    