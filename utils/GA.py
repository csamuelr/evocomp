import numpy as np
import random as r


class GA():
    def __init__(self, model) -> None:
        
        if type(model.pop) is np.ndarray:
            self.population = model.pop
        else:
            raise AssertionError("Not a valid population!")
        self.Ub = model.UB
        self.Lb = model.LB
        self.pop_size = len(model.pop)
        self.parents = np.zeros(model.pop.shape)
        self.pop_shape = model.pop.shape
        self.new_population = np.zeros(model.pop.shape)
        self.fitness = np.zeros(model.pop.size)
        self.best_fitness = 0

    def uniform_mutation(self, gene, mutation_rate):
        for i in range(gene.size):
            if(np.random.uniform()<mutation_rate):
                gene[i] = np.random.uniform(low=self.Lb, high=self.Ub)
        return gene
    
    def bit_flipping(self, gene, mutation_rate):
        for i in range(gene.size):
            if(np.random.uniform()<mutation_rate):
                if(gene[i] == 1):
                    gene[i] = 0
                else:
                    gene[i] = 1
        return gene
    
    def gaussian_mutation(self, gene, avg, std_dev, mutation_rate):
        for i in range(gene.size):
            if(np.random.uniform()<mutation_rate):
                gene[i] = np.random.normal(loc=avg, scale=std_dev)
                if(gene[i] > self.Ub):
                    gene[i] = self.Ub
                if(gene[i] < self.Lb):
                    gene[i] = self.Lb
            else:
                pass
        return gene

    def sweep_parent2(self, allele1, parent1, parent2, min, max, child, i):
        for j in range(parent2.shape[0]):
            if((j < min or j > max) and allele1 == parent2[j] and child[j] == -1):
                child[j] = parent2[i]
            elif(allele1 == parent2[j]):
                if(parent2[i] != allele1):
                    self.sweep_parent2(parent1[j], parent1, parent2, min, max, child, i)
        return child

    def pmx(self, parent1, parent2):
        child = -np.ones(parent1.shape)

        random_idxs = np.random.choice(np.arange(parent1.size), size=2)

        idx_aux = 0
        if(random_idxs[1] < random_idxs[0]):
            idx_aux = random_idxs[1]
            random_idxs[1] = random_idxs[0]
            random_idxs[0] = idx_aux
        elif(random_idxs[1] == random_idxs[0]):
            random_idxs[0] -= 1
            if(random_idxs[0] < 0):
                idx_aux = random_idxs[1]
                random_idxs[1] = random_idxs[0]
                random_idxs[0] = idx_aux

        if(random_idxs[1] < 0):
            random_idxs[1] %= parent1.shape[0]
        child[random_idxs[0]:random_idxs[1] + 1] = parent1[random_idxs[0]:random_idxs[1] + 1]
        idx_aux = (random_idxs[1] + 1)%parent1.shape[0]
        for i in range(random_idxs[0], random_idxs[1] + 1):
            its_ok = 1
            for allele in parent1[random_idxs[0]:random_idxs[1] + 1]:
                if(allele == parent2[i]):
                    its_ok = 0
            if(its_ok):
                child = self.sweep_parent2(parent1[i], parent1, parent2, random_idxs[0], random_idxs[1], child, i)

        child = np.where(child == -1, parent2, child)
        return child
        
    def whole_arithmetic_recombination(self, parent1, parent2, alpha= 0.5):
        return (alpha*parent1 + (1-alpha)*parent2, (1-alpha)*parent1 + alpha*parent2)
    
    def whole_arithmetic_recombination(self, parent1, parent2, alpha= 0.5):
        return (alpha*parent1 + (1-alpha)*parent2, (1-alpha)*parent1 + alpha*parent2)

    def fitness_proportional_selection(self, population, weights):
        return r.choices(population, weights)[0]

    def one_point_crossover(self, parent1, parent2):
        thresh = np.random.randint(low=0, high=len(parent1))
        child1 = np.concatenate((parent1[0:thresh], parent2[thresh:parent1.shape[0]]), axis=0)
        child2 = np.concatenate((parent2[0:thresh], parent1[thresh:parent1.shape[0]]), axis=0)
        return child1, child2

    def evaluate(self, model, is_offspring=False):
        evaluations = np.zeros(self.pop_size)
        if(is_offspring):
            for i in range(self.pop_size):
                evaluations[i] = model.objective_function(self.population[i])
            
        else:
            for i in range(self.pop_size):
                evaluations[i] = model.objective_function(self.population[i])
            self.best_fitness = model.objective_function(self.population[np.argmax(self.fitness)])
        
        

        if(model.max_or_min):
            if((evaluations<0).sum() > 0):
                evaluations = evaluations - np.amin(evaluations)
                
        else:
            evaluations = np.abs(evaluations - np.amax(evaluations))
        
        self.fitness = evaluations

            


    def select_parents(self, model):
        if(model.parent_selection == "fitness proportional selection"):
            for i in range(self.pop_size):
                self.parents[i] = self.fitness_proportional_selection(self.population, self.fitness)
        elif(model.parent_selection == "fitness proportional selection kbest"):
            temp = np.argpartition(-self.fitness, 30)
            result_args = temp[:30]
            for i in range(self.pop_size):
                self.parents[i] = self.fitness_proportional_selection(self.population[result_args], self.fitness[result_args])
            
        else:
            raise AssertionError("Selection model not implemented or doesn't exist!")



    def recombine(self, model):
        if(model.recombination == "whole arithmetic recombination"):
            for i in range(self.pop_size):
                (self.population[i], self.population[-i]) = self.whole_arithmetic_recombination(self.parents[-i], self.parents[i], model.alpha)
        elif(model.recombination == "whole arithmetic recombination best"):
            self.population = self.parents
            self.evaluate(model)
            for i in range(self.pop_size):
                if(self.fitness[i] > self.fitness[-i]):
                    (self.population[i], self.population[-i]) = self.whole_arithmetic_recombination(self.parents[i], self.parents[-i], model.alpha)
                else:
                    (self.population[i], self.population[-i]) = self.whole_arithmetic_recombination(self.parents[-i], self.parents[i], model.alpha)
            self.evaluate(model)
        elif(model.recombination == "pmx"):
            for i in range(self.pop_size):
                self.population[i] = self.pmx(self.parents[-i], self.parents[i])
                self.population[-i] = self.pmx(self.parents[i], self.parents[-i])
        elif(model.recombination == "one-point-crossover"):
            for i in range(self.pop_size):
                (self.population[i], self.population[-i]) = self.one_point_crossover(self.parents[-i], self.parents[i])
        else:
            raise AssertionError("Recombination model not implemented or doesn't exist!")

    def mutate(self, model):
        if(model.mutation == "gaussian mutation"):
            for i in range(self.pop_size):
                self.population[i] = self.gaussian_mutation(self.population[i], model.avg, model.std_dev, model.mutation_rate)
        elif(model.mutation == "uniform mutation"):
            for i in range(self.pop_size):
                self.population[i] = self.uniform_mutation(self.population[i], model.mutation_rate)
        elif(model.mutation == "bit-flipping"):
            for i in range(self.pop_size):
                self.population[i] = self.bit_flipping(self.population[i], model.mutation_rate)
        else:
            raise AssertionError("Mutation model not implemented or doesn't exist!")

    def select_offspring(self, model):
        if(model.offspring_selection == "fitness proportional selection"):
            for i in range(self.pop_size):
                self.population[i] = self.fitness_proportional_selection(self.population, self.fitness)
        elif(model.parent_selection == "fitness proportional selection kbest"):
            temp = np.argpartition(-self.fitness, 30)
            result_args = temp[:30]
            for i in range(self.pop_size):
                self.parents[i] = self.fitness_proportional_selection(self.population[result_args], self.fitness[result_args])
        else:
            raise AssertionError("Selection model not implemented or doesn't exist!")
    
    def return_best(self, model):
        print("The best gene is ", self.population[np.argmax(self.fitness)])
        print("Alternative gene ", self.population[1])
        print("Fitness of best ", self.best_fitness)
        print("Fitness of alternative ", model.objective_function(self.population[2]))
    
    def run(self, model):
        condition = True
        while(condition):
            self.evaluate(model)
            self.select_parents(model)
            #print(self.fitness)
            self.recombine(model)
            self.mutate(model)
            self.evaluate(model, True)
            condition = model.termination_condition(self.best_fitness)
            #print(self.fitness)
            #self.select_offspring(model)
        #self.return_best(model)


class Model():
    def __init__(self, pop, objective_function, recombination, mutation, offspring_selection, parent_selection, mutation_rate, n_of_generations, UB, LB, max_or_min, alpha = 0.5, avg = 0, std_dev = 0.5) -> None:
        self.max_or_min = max_or_min
        self.pop = pop
        self.UB = UB
        self.LB = LB
        self.objective_function = objective_function
        self.recombination = recombination
        self.mutation = mutation
        self.offspring_selection = offspring_selection
        self.parent_selection = parent_selection
        self.alpha = alpha
        self.avg = avg
        self.std_dev = std_dev
        self.mutation_rate = mutation_rate
        self.counter = 0
        self.n_of_generations = n_of_generations
        self.fitness = []
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


#As linhas comentadas abaixo foram utilizadas como script de teste

#Representação real

#gene = np.random.uniform(low=-32, high=32, size=5)
#print(gene[0].size)
#
#print("Um gene exemplo é dado por: ", gene)




#dist = []
#nbins = 100
#for _ in range(100000):
#    dist.append(np.random.normal()) #ATENÇÃO: essa funçao que eu usei foi alterada nos testes
#
#
#print(dist)
#plt.hist(dist, bins = 50)
#plt.show()
#gaussian_mutation(gene, 0.7)

#print("Um gene exemplo é dado por: ", gene)

##



#Operadores de adjacência

#PMX
#pai2 = np.array([4, 3, 6, 2 ,8, 5, 9 ,0, 7, 1])
#pai1 = np.array([1, 2, 5, 3 , 6, 7 ,9,8, 0, 4])
#
#print("pai 1 é ",pai1)
#print("pai 2 é ",pai2)
#
#def varredura(allele1, parent1, parent2, min, max, child, i):
#    for j in range(parent2.shape[0]):
#        if((j < min or j > max) and allele1 == parent2[j] and child[j] == -1):
#            child[j] = parent2[i]
#        elif(allele1 == parent2[j]):
#            if(parent2[i] != allele1):
#                varredura(parent1[j], parent1, parent2, min, max, child, i)
#    return child
#
#def pmx(parent1, parent2):
#    child = -np.ones(parent1.shape)
#
#    random_idxs = np.random.choice(np.arange(parent1.size), size=2)
#
#    idx_aux = 0
#    if(random_idxs[1] < random_idxs[0]):
#        idx_aux = random_idxs[1]
#        random_idxs[1] = random_idxs[0]
#        random_idxs[0] = idx_aux
#    elif(random_idxs[1] == random_idxs[0]):
#        random_idxs[0] -= 1
#        if(random_idxs[0] < 0):
#            idx_aux = random_idxs[1]
#            random_idxs[1] = random_idxs[0]
#            random_idxs[0] = idx_aux
#    
#    if(random_idxs[1] < 0):
#        random_idxs[1] %= parent1.shape[0]
#    child[random_idxs[0]:random_idxs[1] + 1] = parent1[random_idxs[0]:random_idxs[1] + 1]
#    idx_aux = (random_idxs[1] + 1)%parent1.shape[0]
#    for i in range(random_idxs[0], random_idxs[1] + 1):
#        its_ok = 1
#        for allele in parent1[random_idxs[0]:random_idxs[1] + 1]:
#            if(allele == parent2[i]):
#                its_ok = 0
#        if(its_ok):
#            child = varredura(parent1[i], parent1, parent2, random_idxs[0], random_idxs[1], child, i)
#
#    child = np.where(child == -1, parent2, child)
#    return child
#
#print(pmx(pai1,pai2))
#print(pmx(pai2,pai1))

#Recombinação

#one-point-crossover
#pai2 = np.array([1,0,1,0,1,1,0,0,0])
#pai1 = np.array([0,0,1,0,0,1,0,1,1])
#print("pai 1 é ",pai1)
#print("pai 2 é ",pai2)
#def one_point_crossover(parent1, parent2):
#    thresh = np.random.randint(low=0, high=len(parent1))
#    print(thresh)
#    child1 = np.concatenate((parent1[0:thresh], parent2[thresh:parent1.shape[0]]), axis=0)
#    child2 = np.concatenate((parent2[0:thresh], parent1[thresh:parent1.shape[0]]), axis=0)
#    return child1, child2
#print(one_point_crossover(pai1, pai2))
#Whole Arithmetic Recombination
#pai1 = np.random.uniform(low=-32, high=32, size=5)
#pai2 = np.random.uniform(low=-32, high=32, size=5)
#
#print("O pai 1 é: ", pai1, "\nO pai 2 é: ", pai2)
#
#def whole_arithmetic_recombination(parent1, parent2, alpha= 0.5):
#    return (alpha*parent1 + (1-alpha)*parent2, (1-alpha)*parent1 + alpha*parent2)
#
#filho1, filho2 = whole_arithmetic_recombination(pai1, pai2, alpha = 0.6)
#
#print("O filho 1 é: ", filho1, "\nO filho 2 é: ", filho2)

#Seleção

#Fitness Proportional Selection

#def fitness_proportional_selection(population, weights):
#    return r.choices(population, weights)[0]
#
#gene = np.array([np.random.uniform(low=-32, high=32, size=5),np.random.uniform(low=-32, high=32, size=5)])
#print(gene)
#
#choice = fitness_proportional_selection(gene, weights = [0.3,0.7])
#
#print(choice)

#MAPEAMENTO PARA FITNESS
#gene = np.array([np.random.uniform(low=-32, high=32, size=5),np.random.uniform(low=-32, high=32, size=5)])
#
#print(np.zeros(gene.shape))
#print(gene)
#
#print(gene - np.amin(gene))
#
#print(np.abs(gene - np.amax(gene)))
#
#print(r.choices(gene[0], weights=(gene - np.amin(gene))[0]))


#CLASSES SÃO PASSADAS POR REFERÊNCIA?
#class A():
#    def __init__(self) -> None:
#        self.um = 1
#        self.dois = 2
#
#def rand_func(A):
#    A.um = 2
#    A.dois = 1
#
#a = A()
#
#print("este é 1:", a.um, " este é 2: ", a.dois)
#
#rand_func(a)
#
#print("este é 1:", a.um, " este é 2: ", a.dois)


#gene = np.array([np.random.uniform(low=-32, high=32, size=5),np.random.uniform(low=-32, high=32, size=5)])
#print(gene)
#ackley(gene[0])
#
#print(gene)


#TODOS OS ALGORÍTMOS SEGUIRÃO A LÓGICA ABAIXO:

#BEGIN
#INITIALISE population with random candidate solutions;
#EVALUATE each candidate;
#REPEAT UNTIL ( TERMINATION CONDITION is satisfied ) DO
#1 SELECT parents;
#2 RECOMBINE pairs of parents;
#3 MUTATE the resulting offspring;
#4 EVALUATE new candidates;
#5 SELECT individuals for the next generation;
#OD
#END