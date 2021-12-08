from numpy.random import randint
from numpy.random import normal


class Individual:

    def __init__(self, ndim, linf, lsup):
        self.__ndim = ndim
        self.__fit  = None
        self.__gene = []
        self.__linf = linf
        self.__lsup = lsup


    def create(self):
        self.__gene = randint(self.__linf, self.__lsup, self.__ndim)
        
    def get(self):
        return self.__gene
    
    def set(self, gen):
        self.__gene = gen

    def get_fit(self):
        return self.__fit

    def set_fit(self, f):
        self.__fit = f

    def __repr__(self):
        return 'Gene: {} Fitness: {}'.format(self.__gene, self.__fit)

    def __lt__(self, other):
        return self.get_fit() < other.get_fit()

    def __gt__(self, other):
        return other.__lt__(self)

    def __eq__(self, other):
        return self.get_fit() == other.get_fit()
    
    def __ne__(self, other):
        return not self.__eq__(other)

class IndividualStrig:

    def __init__(self, ndim, linf, lsup, nbits):
        self.__ndim = ndim
        self.__fit  = None
        self.__gene = []
        self.__linf = linf
        self.__lsup = lsup
        self.__nbits = nbits


    def create(self):
        
        g = []

        for i in range(self.__ndim):
            gd = '0b'
            for j in range(self.__nbits):
                r = normal(0, 1)
                if r < 0.5:
                    gd += '0'
                else:
                    gd += '1'

            g.append(gd)

        self.__gene = g

        
    def get(self):
        return self.__gene
    
    def set(self, gen):
        self.__gene = gen

    def get_fit(self):
        return self.__fit

    def set_fit(self, f):
        self.__fit = f

    def __repr__(self):
        return 'Gene: {} Fitness: {}'.format(self.__gene, self.__fit)

  

class Population:

    def __init__(self, np, linf, lsup, ndim, string=False, nbits=None):
        self.__np = np
        self.__individuals = []
        self.__linf = linf
        self.__lsup = lsup
        self.__ndim = ndim
        self.__nbits = nbits
        self.__string = string

    def create(self):

            
        if self.__string:
            print("Generatin string populagion. ")

            for _ in range(self.__np):    
                individual = IndividualStrig(linf=self.__linf, lsup=self.__lsup, ndim=self.__ndim, nbits=self.__nbits)
                individual.create()
                self.__individuals.append(individual)


        else:

            for _ in range(self.__np):    
                individual = Individual(linf=self.__linf, lsup=self.__lsup, ndim=self.__ndim)
                individual.create()
                self.__individuals.append(individual)

        return self.__individuals

    def get(self):
        return self.__individuals

    def __repr__(self) -> str:
        return 'Population Size: {}'.format(self.__np)