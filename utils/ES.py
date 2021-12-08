import numpy as np
import random as r
import matplotlib.pyplot as plt
from numpy import random
import time

class ES():
    def __init__(self,model) -> None:
        if type(model.pop) is np.ndarray:
            self.population = model.pop
        else:
            raise AssertionError("Not a valid population!")
        self.Ub = model.UB
        self.Lb = model.LB
        self.mu = len(model.pop)
        self.pop_shape = model.pop.shape
        
        self.sigma = model.sigma
        self.k_last = model.k
        self.offspring = np.zeros((model.lam,model.pop.shape[1]))
        self.fitness = np.zeros(model.lam)
        self.k = 0
        self.last_better = 0
        self.better_count = 0
        self.best = 0
    
    def gaussian_mutation(self, avg=0):
        return np.random.normal(loc=avg, scale=self.sigma, size=self.offspring.shape[1])

    def mutate(self, model):
        for i in range(model.lam):
            self.offspring[i] += self.gaussian_mutation(model.mutation_avg)
            np.where(self.offspring[i] < self.Lb, self.population, self.Lb)
            np.where(self.offspring[i] > self.Lb, self.population, self.Ub)
        self.k += 1

        if(np.amax(self.fitness) > self.last_better):
            self.last_better = np.amax(self.fitness)
            self.better_count += 1


        if(self.k == self.k_last):
            ps = self.better_count/self.k
            if(ps > 0.2):
                self.sigma = self.sigma/np.random.uniform(low=0.8)
            elif(ps < 0.2):
                self.sigma = self.sigma*np.random.uniform(low=0.8)
            else:
                pass
            self.k = 0
            self.better_count = 0

    
    def select_parents(self, model):
        if(model.parent_selection == "random"):
            for i in range(model.lam):
                rand_idx = np.random.randint(low=0, high=model.mu)
                self.offspring[i] = self.population[rand_idx]
        else:
            raise AssertionError("Selection model not implemented or doesn't exist!")
    

    def mu_comma_lambda(self, model):
        indices = np.argpartition(self.fitness, -self.mu)[-self.mu:]
        self.population = self.offspring[indices]

    def select_offspring(self, model):
        if(model.offspring_selection == "mu+lambda"):
            pass
        elif(model.offspring_selection == "mu,lambda"):
            self.mu_comma_lambda(model)
        else:
            raise AssertionError("Selection model not implemented or doesn't exist!")

    def evaluate(self, model, is_offspring=False):
        if(is_offspring):
            evaluations = np.zeros(model.lam)
            for i in range(model.lam):
                evaluations[i] = model.objective_function(self.offspring[i])
        else:
            evaluations = np.zeros(model.mu)
            for i in range(model.mu):
                evaluations[i] = model.objective_function(self.population[i])
            self.best = model.objective_function(self.offspring[np.argmax(self.fitness)])
            
        
        if(model.max_or_min):
            if((evaluations<0).sum() > 0):
                evaluations = evaluations - np.amin(evaluations)
        else:
            evaluations = np.abs(evaluations - np.amax(evaluations))
            
        self.fitness = evaluations

    def return_best(self, model):
        print("The best gene is ", self.offspring[np.argmax(self.fitness)])
        print("Alternative gene ", self.offspring[1])
        print("Fitness of best ", model.objective_function(self.offspring[np.argmax(self.fitness)]))
        print("Fitness of alternative ", model.objective_function(self.offspring[1]))
        
    def run(self, model):
        condition = True
        while(condition):
            self.evaluate(model)
            self.select_parents(model)
            self.evaluate(model, True)
            self.mutate(model)
            self.select_offspring(model)
            condition = model.termination_condition(self.best)
        #self.return_best(model)


class Model():
    def __init__(self, pop, objective_function,  mutation, offspring_selection, parent_selection, sigma, lam, mu, k, n_of_generations, UB, LB, max_or_min, mutation_avg = 0) -> None:
        self.max_or_min = max_or_min
        self.pop = pop
        self.objective_function = objective_function
        self.mutation = mutation
        self.offspring_selection = offspring_selection
        self.parent_selection = parent_selection
        self.sigma = sigma
        self.lam = lam
        self.mu = mu
        self.k = k
        self.counter = 0
        self.mutation_avg = mutation_avg
        self.fitness = []
        self.UB = UB
        self.LB = LB
        self.n_of_generations = n_of_generations
        self.best_solution = 0

    def termination_condition(self, fitness):
        self.counter += 1
        #print("Generation: ", self.counter)
        #print("Best fitness: ",fitness)
        
        
        self.fitness.append(fitness)
        if(self.counter == self.n_of_generations):
            self.best_solution = fitness
            self.counter = 0
            return False
        else:
            return True
    def reset(self):
        self.fitness = []
        self.counter = 0


#gene = np.array([np.random.uniform(low=-32, high=32, size=5),np.random.uniform(low=-32, high=32, size=5)])
#print(gene)
#print(np.zeros((1,gene.shape[1])))



#fitness = np.array([200,3,4,2,1,7,8,4,5,123,1235,123])
#print(fitness)
#print(mu_comma_lambda(fitness, 5))


#gene = np.array([np.random.uniform(low=-32, high=32, size=5),np.random.uniform(low=-32, high=32, size=5)])
#a = np.array(r.choices(gene, weights=[1,2], k=2))
#print(a)


#fitness = np.array([200,3,4,2,1,7,8,4,5,123,1235,123])
#print(fitness[np.argpartition(fitness, -7)[:-7]])


#1. μ > 1 so that different strategies are present
#2. generation of an offspring surplus: λ>μ
#3. a not too strong selective pressure (heuristic: λ/μ = 7, e.g., (15,100))
#4. (μ, λ)-selection (to guarantee extinction of misadapted individuals)
#5. recombination, usually intermediate, of strategy parameters

#Evolution Strategies are also usually implemented with uniform random selection of parents into the mating pool,
#i.e., for each 1 ≤ i ≤ μ we have Puniform(i)=1/μ.
