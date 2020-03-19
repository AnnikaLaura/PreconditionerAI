import numpy as np
import random
import math
from random import randint
from scipy.sparse import dok_matrix , csc_matrix , identity , linalg, save_npz, load_npz
from my_utils import timeit
import logging
import os

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
    def __init__(self,dim=1,prob=0.0):
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

    def inverse_D(self):
        diagonal = self.D.diagonal()
        tempD = csc_matrix(self.D)
        for i in range(self.dim):
            tempD[i,i] = 1/diagonal[i]
        return tempD

    @timeit
    def invert(self):
        return self.Q.transpose() * self.inverse_D() * self.Q

    def save(self,foldername):
        try:
            if not os.path.exists(foldername):
                os.mkdir(foldername)
        except OSError:
            logging.error(f"Creation of the directory {foldername} failed!")
        A_path = os.path.join(foldername,"A.npz")
        Q_path = os.path.join(foldername,"Q.npz")
        D_path = os.path.join(foldername,"D.npz")

        save_npz(A_path, self.A)
        save_npz(Q_path, self.Q)
        save_npz(D_path, self.D)

    def load(self,foldername):
        A_path = os.path.join(foldername,"A.npz")
        Q_path = os.path.join(foldername,"Q.npz")
        D_path = os.path.join(foldername,"D.npz")

        self.A = load_npz(A_path)
        self.Q = load_npz(Q_path)
        self.D = load_npz(D_path)

        self.dim = self.A.shape[0]
        # self.prob can't be reconstructed and doesn't need to be

    def small_matrices(self):
        # yield 128 x 128 matrices

        # if dim is not a multiple of 128, the last matrix is
        # "padded by an identity matrix"
        if self.dim % 128 != 0:
            logging.debug("self.dim not divisible by 128.")
            logging.debug("Last matrix will be padded.")

        for k in range(math.ceil(self.dim / 128.0)):
            logging.debug(f"Big matrix indices: [{k*128},{k*128+127}]")
            tmp = np.identity(128)

            upper_limit = min((k+1)*128,self.dim)
            start = k*128
            for i in range(start,upper_limit):
                row = self.A.getrow(i)
                for j in range(start,upper_limit):
                    val = row.getcol(j).todense()[0][0]
                    tmp[i-start,j-start] = val

            yield tmp

    def preconditioned_cond(self,precond,precond_inv):
        condition_num  = linalg.norm(precond_inv * self.A)
        condition_num *= linalg.norm(self.Q * self.inverse_D * self.Q.transpose() * precond)
        return condition_num

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG , format='[%(asctime)s] - [%(levelname)s] - %(message)s')

    # creating a test object
    custom = CustomSparse(1000,0.001)
    custom.create([1,99999999])
    logging.debug(f"Rounding error while inverting: {(custom.invert().dot(custom.A)-identity(custom.dim)).max()}")
    # save a CustomSparse object
    custom.save("test1")

    # test loading a CustomSparse object
    new_custom = CustomSparse()
    new_custom.load("test1")
    logging.debug(f"Loading error: {(custom.invert()-new_custom.invert()).max()}")

    for matrix in custom.small_matrices():
        #print(matrix)
        print("")
