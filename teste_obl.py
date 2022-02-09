
from utils.opposite_de import DifferentialEvolution
from utils.statistics import Statistics
from functions        import *
import matplotlib.pyplot as plt
from numpy import mean
from numpy import array

all_best_solutions = []

print("Processando...")
for i in range(20):
	# print("Execução %d de 30" % (i+1))
	de = DifferentialEvolution(
		ng=100,
		np=5,
		cr=0.9, 
		f=0.8, 
		evfunc='Ackley', 
		algorithm='rand_1_bin',
		obl=False
	)
	de.evolve()
	all_best_solutions.append(de.get_all_best_solutions())
	
print("ok.")



# print(len(all_best_solutions[0]))


best_solutions = [[]] * 100
mean_all_best_solutions = []

for execution in all_best_solutions:
	# print(len(execution))
	for i, individual in enumerate(execution):
		if i == 0:
			print(individual.get_fit())
		
		best_solutions[i].append(individual.get_fit())
		# print("\n\n")

print((best_solutions[0]))

for pos in best_solutions:
	mean_all_best_solutions.append(mean(pos))
	

plt.plot(mean_all_best_solutions)
plt.show()

# x, fitness = [], []
# for i, s in enumerate(all_best_solutions[0]):
# 	fitness.append(s.get_fit())
# 	x.append(i)

# plt.figure(figsize=(12,8))
# plt.style.use('seaborn')
# plt.plot(x, fitness, label='Best Fitness Without Opposition')
# plt.xlabel('Generations')
# plt.ylabel('Best Fitness')
# plt.show()

# # # Executando com Opposite-based Learning

# all_opposite_best_solutions = []

# print("Processando...")

# for i in range(30):
# 	de = DifferentialEvolution(
# 		ng=100,
# 		np=300,
# 		cr=0.9, 
# 		f=0.8, 
# 		evfunc='Ackley', 
# 		algorithm='rand_1_bin',
# 		obl=True
# 	)
# 	de.evolve()
# 	all_opposite_best_solutions = de.get_all_best_solutions()

# print("ok.")


# all_opposite_best_solutions

# xop, fitnessopp = [], []
# for i, s in enumerate(all_opposite_best_solutions):
# 	fitnessopp.append(s.get_fit())
# 	xop.append(i)

# plt.figure(figsize=(12,8))
# plt.style.use('seaborn')
# plt.plot(xop, fitnessopp)
# plt.xlabel('Generations')
# plt.ylabel('Best Fitness')
# plt.show()

# plt.figure(figsize=(12,8))
# plt.style.use('seaborn')
# plt.plot(x, fitness, label='quadratic')
# plt.plot(xop, fitnessopp, label='Best Fitness With Opposition')
# plt.xlabel('Generations')
# plt.ylabel('Best Fitness')
# plt.show()


