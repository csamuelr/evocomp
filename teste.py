from utils.opposite_de import DifferentialEvolution
from matplotlib import pyplot as plt
import numpy as np

all_solutions = []
for i in range(100):
    all_solutions.append([])

for i in range(10):
    
    all_best = []
    de = DifferentialEvolution(
        ng=100,
        np=50,
        cr=0.9, 
        f=0.8, 
        evfunc='Ackley', 
        algorithm='rand_1_bin',
        obl=False
    )

    de.evolve()
    all_best = de.get_all_best_solutions()

    for i, indv in enumerate(all_best):
        # if i == 0:
        #     print(indv.get_fit())
        all_solutions[i].append(indv.get_fit())
        # if i == 5:
        #     break
    # break

all_mean = []
for cfit in all_solutions:
    m = np.mean(cfit)
    all_mean.append(m)

print((all_mean))
plt.plot(all_mean)
plt.show()