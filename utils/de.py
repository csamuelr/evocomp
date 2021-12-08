from . population import Individual
from . population import Population
from inspect import getmembers
from inspect import isclass
from importlib import import_module
from sys import exit
from numpy.random import randint
from numpy import array
from numpy.random import uniform 
from numpy.random import exponential
from heapq import heappop
from heapq import heappush
from time import time

class DifferentialEvolution:

    def __init__(self, ng, np, cr, f, evfunc, algorithm):
        self.__ng         = ng
        self.__np         = np
        self.__cr         = cr
        self.__f          = f
        self.__algorithm  = None
        self.__evfunc     = None
        self.__population = []
        self.__temporaryp = []
        self.__crosstype  = None
        self.__best_solution = None
        self.__all_solutions = []
        self.__all_best_solutions = []
        self.__total_time = None

        if callable(getattr(self, algorithm, None)):
            self.__algorithm = getattr(self, algorithm, None)
            self.__crosstype = algorithm.split('_')[-1]

        else:
            print('\nERROR:\n\tDE algortihm \'{}\' is not available.'.format(algorithm))
            exit(-1)

        
        modules = import_module('functions')
        for name, cls in getmembers(modules, isclass):
            if evfunc == name:
                clsref = getattr(modules, name)
                self.__evfunc = clsref()
                break
        else:
            print ('\nERROR:\n\tEvaluation function \'{}\' is not available.'.format(evfunc))
            exit(-1)

    
    def heap_sort(self):

        heap = []
        for individual in self.__population:
            heappush(heap, individual)

        ordered = []
        while heap:
            ordered.append(heappop(heap))

        self.__population = ordered
    

    def evaluate(self):
        
        for individual in self.__population:
            fit = self.__evfunc.compute(individual.get())
            individual.set_fit(fit)
        
        for t_individual in self.__temporaryp:
            fit = self.__evfunc.compute(t_individual.get())
            t_individual.set_fit(fit)

    def evolve(self):
        
        ti = time()

        self.__all_solutions = []
        self.__all_best_solutions = []

        population = Population(
              np=self.__np, 
            linf=self.__evfunc.linf, 
            lsup=self.__evfunc.lsup, 
            ndim=self.__evfunc.ndim            
        )

        self.__population = population.create() 

        self.evaluate()

        
        for g in range(self.__ng):
            
            print("\rGERAÇÃO: {}".format(g+1), end='', flush=True)

            for i, individual in enumerate(self.__population):
                # mutation
                mutated_vector = self.mutate(i)

                # crossover
                new_individual = Individual(
                    linf=self.__evfunc.linf, 
                    lsup=self.__evfunc.lsup, 
                    ndim=self.__evfunc.ndim            
                )
                new_individual.create()

                u = self.crossover(individual, mutated_vector)

                new_individual.set(array(u))
                self.__temporaryp.append(new_individual)

            # evaluation
            self.evaluate()

            for i, (indva, indvb) in enumerate(zip(self.__population, self.__temporaryp)):
                if indvb.get_fit() < indva.get_fit():
                    self.__population[i] = indvb

            self.__all_solutions.append(self.__population)
            self.__all_best_solutions.append(self.get_best())

            self.__temporaryp.clear()       

        self.__best_solution = self.get_best()

        tf = time()
      
        self.__total_time =  tf - ti

    def mutate(self, i):
        return self.__algorithm(i)

    def crossover(self, individual, mutated_vector):

        call = None
        if self.__crosstype == 'exp':
            call = exponential
        elif self.__crosstype == 'bin':
            call = uniform 
        
        J = randint(0, self.__np)

        u = []
        for j, value in enumerate(individual.get()):
            r = float(format(call(), '.1f'))
            if r < self.__cr or j == J:
                u.append(mutated_vector[j])
            else:
                u.append(value)

        return u

    def get_best(self):

        best = self.__population[0]
        for indv in self.__population:
            if indv.get_fit() < best.get_fit():
                best = indv

        return best
    
    def get_best_solution(self):
        return self.__best_solution

    def get_all_solutions(self):
        return self.__all_solutions

    def get_all_best_solutions(self):
        return self.__all_best_solutions

    def get_execution_time(self):
        return self.__total_time

    def rand_1_bin(self, i):

        ''' Vi,G = Vr1,G + F (Vr2,G - Vr3,G) '''

        indexes = []
        indexes.append(i)     
        while len(indexes) <= 4:
            r = randint(0, self.__np)
            if not r in indexes:
                indexes.append(r)

        r1, r2, r3 = indexes[1], indexes[2], indexes[3]

        v = self.__population[r1].get() + self.__f * (self.__population[r2].get() - self.__population[r3].get())

        return v


    def rand_2_bin(self, i):

        ''' Vi,G = Vr1,G + F (Vr2,G - Vr3,G + Vr4,G – Vr5,G) '''

        indexes = []
        indexes.append(i)     
        while len(indexes) <= 6:
            r = randint(0, self.__np)
            if not r in indexes:
                indexes.append(r)

        r1, r2, r3, r4, r5 = indexes[1], indexes[2], indexes[3], indexes[4], indexes[5]

        v = self.__population[r1].get() + self.__f * ((self.__population[r2].get() - self.__population[r3].get()) + (self.__population[r4].get() - self.__population[r5].get()))

        return v


    def randtobest_1_bin(self, i):
        
        ''' Vi,G = Vr1,G + F (Vbest,G – Vr1,G + Vr2,G – Vr3,G) '''

        best = self.__population.index(self.get_best())

        indexes = []
        indexes.append(i)   
        indexes.append(best)  

        while len(indexes) <= 5:
            r = randint(0, self.__np)
            if not r in indexes:
                indexes.append(r)

        r1, r2, r3 = indexes[1], indexes[2], indexes[3]  

        v = self.__population[r1].get() + self.__f * (( self.__population[best].get() - self.__population[r1].get()) + (self.__population[r2].get() - self.__population[r3].get()))

        return v


    def rand_2_exp(self):
        pass

    def best_1_bin(self, i):
        
        ''' Vi,G = Vbest,G + F (Vr2,G - Vr3,G) '''

        best = self.__population.index(self.get_best())

        indexes = []
        indexes.append(i)   
        indexes.append(best)  

        while len(indexes) <= 4:
            r = randint(0, self.__np)
            if not r in indexes:
                indexes.append(r)

        r2, r3 = indexes[2], indexes[3]  

        v = self.__population[best].get() + self.__f * (self.__population[r2].get() - self.__population[r3].get())

        return v



    def best_2_exp(self):
        pass

    def currenttobest_1_bin(self, i):

        ''' Vi,G = Vi,G + F (Vbest,G – Vi,G + Vr2,G – Vr3,G) '''

        best = self.__population.index(self.get_best())
        current = i

        indexes = []
        indexes.append(current)   
        indexes.append(best)
        
        while len(indexes) <= 4:
            r = randint(0, self.__np)
            if not r in indexes:
                indexes.append(r)

        r2, r3 = indexes[2], indexes[3]
        
        v = self.__population[current].get() + self.__f * ((self.__population[best].get() - self.__population[current].get()) + (self.__population[r2].get() - self.__population[r3].get()))

        return v

    def currenttorand_1_bin(self, i):
        
        ''' Vi,G = Vi,G + F (Vr1,G – Vi,G + Vr2,G – Vr3,G) '''
        
        current = i

        indexes = []
        indexes.append(current)   
        
        while len(indexes) <= 4:
            r = randint(0, self.__np)
            if not r in indexes:
                indexes.append(r)

        r1, r2, r3 = indexes[1], indexes[2], indexes[3]        

        v = self.__population[current].get() + self.__f * ((self.__population[r1].get() - self.__population[current].get()) + (self.__population[r2].get() - self.__population[r3].get()))

        return v



if __name__ == '__main__':

    de = DifferentialEvolution(
        ng=300,
        np=100,
        cr=0.9, 
        f=0.8, 
        evfunc='Griewank', 
        algorithm='best_1_bin'
    )

    de.evolve()
    print("\nBEST SOLUTION: \n{}".format(de.get_best_solution()))