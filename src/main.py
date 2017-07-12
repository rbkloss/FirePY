from firefly import Firefly
import numpy as np


def main():
    def utility(pop):
        '''
        X**2 = 2
        :param pop: 
        :return: 
        '''
        util = -abs((pop ** 2 - 2))
        return util

    def dist(v1, v2):
        return np.linalg.norm(v1 - v2, 2)

    firefly = Firefly(population_size=10, n_dims=1, n_iterations=50, utility=utility, distance=dist)
    firefly.fit(verbose=True)
    pop = firefly.population
    util = firefly.utilities_
    ordering = np.argsort(util)
    pop = pop[ordering][::-1]
    util = util[ordering][::-1]
    print(pop.tolist())
    print(util.tolist())


if __name__ == '__main__':
    main()
