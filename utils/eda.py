
from inspect import getmembers, isclass
from importlib import import_module
from sys import exit
from numpy import array
from numpy.random import normal

from .population import Population
from bitstring import Bits
from time import time


class CompactGA:

    def __init__(self, ng, np, pmin, pmax, alpha, nbits, evfunc):
        
        self.__ng    = ng
        self.__np    = np
        self.__pmin  = pmin
        self.__pmax  = pmax
        self.__alpha = alpha
        self.__nbits = nbits
        self.__pvect = []
        self.__elite = []
        self.__evfunc = None
        self.__population = []
        self.__best_solution = None
        self.__all_solutions = []
        self.__all_best_solutions = []
        self.__total_time = None

        modules = import_module('functions')
        for name, cls in getmembers(modules, isclass):
            if evfunc == name:
                clsref = getattr(modules, name)
                self.__evfunc = clsref()
                break
        else:
            print ('\nERROR:\n\tEvaluation function \'{}\' is not available.'.format(evfunc))
            exit(-1)


    def evaluate(self):
        
        for individual in self.__population:

            gene = individual.get()

            int_gene = []
            for allel in gene:
                int_gene.append(Bits(allel).int)  
          
            fit = self.__evfunc.compute(array(int_gene))
            individual.set_fit(fit)


    def evolve(self):

        ti = time()

        self.__all_solutions = []
        self.__all_best_solutions = []

        population = Population(
                np=self.__np,
              linf=self.__evfunc.linf,
              lsup=self.__evfunc.lsup,
              ndim=self.__evfunc.ndim,
             nbits=self.__nbits,
            string=True
        )
        
        self.__population = population.create()
        
        for i in range(self.__nbits):
            self.__pvect.append(0.5)
                
        self.evaluate()
        self.__elite = self.get_best()            

        for g in range(self.__ng):
            
            print("\rGERAÇÃO: {}".format(g+1), end='', flush=True)
            
            for individual in self.__population:
                gene = individual.get()
                new_gene = []
                for allel in gene:
                    allel_aux = '0b'
                    for a, p in zip(allel, self.__pvect):
                        r = normal(0, 1)
                        if r < p:
                            allel_aux += '0'
                        else:
                            allel_aux += '1'

                    new_gene.append(allel_aux)
                individual.set(array(new_gene))
            
            self.__population.append(self.__elite)

            #evaluate
            self.evaluate()
            
            #
            best  = self.get_best()
            worst = self.get_worst()

            for individual in self.__population:
                bits_best  = [ Bits(b).bin for b in best.get()  ] 
                bits_worst = [ Bits(w).bin for w in worst.get() ]

                for bb, bw in zip(bits_best, bits_worst):
                    for i, (b, w, p) in enumerate(zip(bb, bw, self.__pvect)):

                        if b != w:
                            if b == '1':
                                self.__pvect[i] = p - self.__alpha
                            else:
                                self.__pvect[i] = p + self.__alpha

                        self.__pvect[i] = max(min(self.__pvect[i], self.__pmax), self.__pmin)
            
            # print(self.__pvect)
            self.__elite = self.get_best()            
            self.__best_solution = self.__elite

            self.__all_best_solutions.append(self.__best_solution)
            self.__all_solutions.append(self.__population)
            
            tf = time()

            self.__total_time = tf - ti


    def get_best(self):
        
        best = self.__population[0]
        for indv in self.__population:
            if indv.get_fit() < best.get_fit():
                best = indv

        return best

    def get_worst(self):
        
        worst = self.__population[0]
        for indv in self.__population:
            if indv.get_fit() > worst.get_fit():
                worst = indv

        return worst

    def get_best_solution(self):
        return self.__best_solution

    def get_all_solutions(self):
        return self.__all_solutions

    def get_all_best_solutions(self):
        return self.__all_best_solutions
    
    def get_execution_time(self):
        return self.__total_time


if __name__ == '__main__':

    cga = CompactGA(
        ng=1100,
        np=150, 
        pmin=0.04, 
        pmax=0.93, 
        alpha=0.03, 
        nbits=5, 
        evfunc='Trid'
    )

    cga.evolve()

    
    print('\nBEST SOLUTION: {} \n'.format(cga.get_best_solution()))
    
    best = cga.get_best_solution().get()
    best = [ (Bits(allel).int) for allel in best]
    print('\nBEST SOLUTION (int): {}\n'.format(best) )