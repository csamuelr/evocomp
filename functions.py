from importlib import import_module
from numpy import pi
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import arange
from numpy import meshgrid
import matplotlib.pyplot as plt


class Ackley:

    def __init__(self, ndim=2, a=20, b=0.2, c=2*pi, linf=-32.768, lsup=32.768):
        
        self.a = a
        self.b = b
        self.c = c
        self.linf = linf
        self.lsup = lsup
        self.ndim = ndim

    def compute(self, obj):
        
        x1, x2 = obj 
       
        return -self.a * exp( -self.b * sqrt( (1.0/self.ndim) * (x1**2 + x2**2) ) ) - exp( (1.0/self.ndim) * (cos(self.c * x1) + cos(self.c + x2)) ) + self.a + exp(1)


    def graph(self):
        
        x1 = arange(self.linf, self.lsup, 2.0)
        x2 = arange(self.linf, self.lsup, 2.0)

        d, e = meshgrid(x1, x2)

        obj = [d,e]

        r = self.compute(obj)
             
        figure = plt.figure(figsize=(12, 12))
        axis = figure.gca( projection='3d')
        axis.plot_surface(d, e, r, cmap='RdBu')
        plt.show()


class Griewank:

    def __init__(self, ndim=2, linf=-600, lsup=600):
    
        self.linf = linf
        self.lsup = lsup
        self.ndim = ndim

    def compute(self, obj):
        x1, x2 = obj

        return ((x1**2)/4000 + (x2**2)/4000) -  (cos(x1/sqrt(1)) * cos(x1/sqrt(2))) + 1 
        

    def graph(self):

        x1 = arange(self.linf, self.lsup, 2.0)
        x2 = arange(self.linf, self.lsup, 2.0)

        d, e = meshgrid(x1, x2)

        obj = [d,e]

        r = self.compute(obj)
            
        figure = plt.figure(figsize=(12, 12))
        axis = figure.gca( projection='3d')
        axis.plot_surface(d, e, r, cmap='RdBu')
        plt.show()

class Trid:

    def __init__(self, ndim=5, linf=None, lsup=None):

        self.ndim = ndim
        self.linf = -self.ndim**2
        self.lsup = self.ndim**2

    def compute(self, obj):
        s1 = 0
        s2 = 0
        
        for x in obj:
            s1 += (x - 1)**2

        for i in range(1, len(obj)):
            s2 += obj[i] * obj[i-1]

        return s1 - s2

    def graph(self):

        x1 = arange(self.linf, self.lsup, 2.0)
        x2 = arange(self.linf, self.lsup, 2.0)

        d, e = meshgrid(x1, x2)

        obj = [d,e]

        r = self.compute(obj)
            
        figure = plt.figure(figsize=(12, 12))
        axis = figure.gca( projection='3d')
        axis.plot_surface(d, e, r, cmap='RdBu')
        plt.show()

    def global_minimun(self):
        
        ''' f(x*) = -d(d+4)(d-1)/6, at Xi = i(d+1-i), for all i = 1,2,...,d'''

        return -self.ndim * (self.ndim + 4) * (self.ndim - 1) / 6


class Colville:

    def __init__(self, ndim=4, linf=-10, lsup=10):

        self.ndim = ndim
        self.linf = linf
        self.lsup = lsup

    def compute(self, obj):

        x1, x2, x3, x4 = obj
        
        return 100 * (x1**2 - x2)**2 + (x1 - 1)**2 + (x3 - 1)**2 + 90 * (x3**2 - x4)**2 + 10.1 * ( (x2 - 1)**2 + (x4 - 1)**2 ) + 19.8 * (x2 - 1) * (x4 - 1)

    def graph(self):
        
        x1 = arange(self.linf, self.lsup, 2.0)
        x2 = arange(self.linf, self.lsup, 2.0)
        x3 = arange(self.linf, self.lsup, 2.0)
        x4 = arange(self.linf, self.lsup, 2.0)

        d, e, f, g = meshgrid(x1, x2, x3, x4)

        obj = [d,e,f,g]

        r = self.compute(obj)
        
        # Ajustar 

        # figure = plt.figure()
        # axis = figure.gca( projection='3d')
        # axis.plot_surface(d, e, f, c=g, z=r)
        # plt.show()

if __name__ == '__main__':

    Colville().graph()