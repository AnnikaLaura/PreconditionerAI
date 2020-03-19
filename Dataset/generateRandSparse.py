import numpy as np
import random
import math
from random import randint
from scipy.sparse import dok_matrix , csc_matrix , identity , linalg
from my_utils import timeit
import logging

# reference: https://en.wikipedia.org/wiki/Jacobi_rotation
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

# class for artificially created matrices
# start with diagonal matrix with entries/ eigenvalues in certain range or list
# then apply a few Jacobi Rotations until matrix is not too sparse anymore
class CustomSparse:
    def __init__(self,dim,prob):
        self.dim = dim
        self.prob = prob
        self.A = csc_matrix((dim,dim),dtype=np.float32)
        self.Q = identity(dim,dtype=np.float32,format = "csc")
        self.D = identity(dim,dtype=np.float32,format = "csc")

    @timeit
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

        # while matrix is too sparse, apply jacobi rotations
        while self.A.count_nonzero()/(float)(self.dim*self.dim) <= self.prob :
            tempQ = jacobiRotation(self.dim)
            self.A = tempQ.transpose() * self.A * tempQ
            self.Q = self.Q * tempQ

    @timeit
    def invert(self):
        diagonal = self.D.diagonal()
        tempD = csc_matrix(self.D)

        for i in range(self.dim):
            tempD[i,i] = 1/diagonal[i]

        return self.Q.transpose() * tempD * self.Q




if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG , format='[%(asctime)s] - [%(levelname)s] - %(message)s')

    # creating a test object
    custom = CustomSparse(10000,0.001)
    custom.create([1,99999999])
    logging.debug((custom.invert().dot(custom.A)-identity(custom.dim)).max())
