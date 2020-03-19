import numpy as np
import random
import math
from random import randint 
from scipy.sparse import dok_matrix , csr_matrix





def jacobiRotation(dim):
    l = randint(0,dim-1)
    k = randint(0,dim-1)
    Q = dok_matrix((dim,dim), dtype=np.float32)
    theta = random.uniform(0,2*math.pi)
    c = math.cos(theta)
    s = math.sin(theta)

    for i in range(dim):
        Q[i,i] = 1
    
    Q[k,k] = c
    Q[l,l] = c
    Q[k,l] = s
    Q[l,k] = -s

    return Q.tocsr()

class testSparse:
    def __init__(self,dim,prob):
        self.dim = dim
        self.prob = prob
        self.A = csr_matrix((dim,dim),dtype=np.float32)
        self.Q = csr_matrix((dim,dim),dtype=np.float32)
        self.D = csr_matrix((dim,dim),dtype=np.float32)

    def create(self,eigenvalues):
        tempD = self.D.todok()
        tempQ = self.Q.todok()
        for i in range(self.dim):
            tempD[i,i] = eigenvalues[i]
            tempQ[i,i] = 1
        self.D = tempD.tocsr()
        self.Q = tempQ.tocsr()
        self.A = self.D
    
        while self.A.count_nonzero()/(float)(self.dim*self.dim) <= self.prob :
            tempQ = jacobiRotation(self.dim)
            self.A = tempQ.transpose() * self.A * tempQ
            self.Q = self.Q * tempQ

    def invert(self):
        diagonal = self.D.diagonal()
        tempD = dok_matrix((self.dim,self.dim), dtype=np.float32)
        for i in range(self.dim):
            tempD[i,i] = diagonal[i]
        return self.Q.transpose() * tempD.tocsr() * self.Q

    


