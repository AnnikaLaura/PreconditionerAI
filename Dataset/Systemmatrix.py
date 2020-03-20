import numpy as np
from scipy.sparse import dia_matrix, csr_matrix, identity

class SystemMatrix(object):
    """Implements System matrix saved as Q*D*Q^t, where D is a Diagonal and Q a Jacobi rotation"""
    #D = dia_matrix() - diagonal matrix storing eigenvalues
    #Q = csr_matrix() - stores Jacobi rotations as matrix
    #dim - dimension
    def __init__(self, dim):
        self.dim = dim
        self.D = dia_matrix((dim,dim))
        self.Q = identity(dim, format='csr') #identity matrix

    def fillDiagonal(self,Data=[]):
        """fills matrix with normally distributed random eigenvalues"""
        if Data == []:
            Data = np.random.normal(size = self.dim)
        self.D.setdiag(Data)

    def randomrot(self):
        """applies one random Jacobi rotation to Q"""
        angle = np.random.uniform(low = -np.pi, high = np.pi) #choose random angle
        rotation = identity(self.dim, format='csr')
        [a,b] = np.random.choice(self.dim, size=2, replace=False) #choose two places for rotation

        rotation[a,a] = np.cos(angle)
        rotation[b,b] = np.cos(angle)
        rotation[a,b] = np.sin(angle)
        rotation[b,a] = -np.sin(angle)

        self.Q = rotation*self.Q

    def print(self):
        tmp = self.Q * self.D * self.Q.transpose()
        print(tmp.todense())

    def addNoise():
        """should we maybe add an extra noise matrix?"""
        pass

    def condition(self):
        maxi = max(abs(self.D.diagonal()))
        mini = min(abs(self.D.diagonal()))
        print(maxi / mini)

def example():
    """create an example instance of a random system matrix"""
    pass


A = SystemMatrix(dim = 4)
A.fillDiagonal()
A.print()
A.randomrot()
A.print()
A.condition()
