import numpy as np
import random
import math
from random import randint 
from scipy.sparse import dok_matrix , csc_matrix , identity , linalg





def jacobiRotation(dim):
    l = randint(0,dim-1)
    k = randint(0,dim-1)
    Q = identity(dim,dtype=np.float32,format = "dok")
    theta = random.uniform(0,2*math.pi)
    c = math.cos(theta)
    s = math.sin(theta)
    
    Q[k,k] = c
    Q[l,l] = c
    Q[k,l] = s
    Q[l,k] = -s

    return Q.tocsc()

class testSparse:
    def __init__(self,dim,prob):
        self.dim = dim
        self.prob = prob
        self.A = csc_matrix((dim,dim),dtype=np.float32)
        self.Q = identity(dim,dtype=np.float32,format = "csc")
        self.D = identity(dim,dtype=np.float32,format = "csc")

    def create(self,eigenvalues,isList = False):
        diag = np.zeros(self.dim,dtype=np.float32)
        if isList:
            for i in range(self.dim):
                diag[i] = eigenvalues[i]
        else:
            a,b = eigenvalues
            for i in range(self.dim):
                diag[i] = random.uniform(a,b)
        
        self.D.setdiag(diag)
        self.A = self.D
        
    
        while self.A.count_nonzero()/(float)(self.dim*self.dim) <= self.prob :
            tempQ = jacobiRotation(self.dim)
            self.A = tempQ.transpose() * self.A * tempQ
            self.Q = self.Q * tempQ

    
    def invert(self):
        diagonal = self.D.diagonal()
        tempD = csc_matrix(self.D)
        tempD = linalg.inv(tempD)
        
        for i in range(self.dim):
            tempD[i,i] = 1/diagonal[i]
        return self.Q.transpose() * tempD * self.Q





