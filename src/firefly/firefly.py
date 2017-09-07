import numpy as np
import logging

class Firefly():
    def __init__(self, population_size=20, n_dims=1, n_iterations=200, step=0.9, annealling=0.97, absorption=1.5,
                 initial_pop=None,
                 utility: callable = None,
                 distance: callable = None) -> None:
        super(Firefly, self).__init__()

        self.n_individuals = population_size
        self.n_dims = n_dims
        self.n_iterations = n_iterations
        self.iteration_counter = 0
        self.step = step
        self.annealling = annealling
        self.absorption = absorption
        self.optional_pop = initial_pop

        self.utilities_ = None
        self.computeUtility = utility
        self.computeDistance = distance

    def fit(self, initial_pop=None, verbose=False):
        if self.computeUtility is None:
            raise ValueError(
                "computeUtility must not be none it must be a callable of the following form\n fn(np.array individual)->float ")
        if self.computeDistance is None:
            raise ValueError(
                "computeDistance must not be none it must be a callable of the following form\n fn(np.array v1, np.array v2)->float ")
        self.population = np.array(np.random.randn(self.n_individuals, self.n_dims), dtype=np.float32)
        if self.optional_pop:
            self.population = np.vstack((self.population, self.optional_pop))
        if initial_pop:
            self.population = np.vstack((self.population, initial_pop))
        self.setup()

        while not self._iterate(self.population, self.absorption, self.annealling):
            if verbose:
                print('best element is:\n {} \nwith utility:{}'.format(self.population[0].tolist(), self.utilities_[0]))

    def setup(self):
        self.n_iterations = 0
        self.population = self.population
        self.utilities_ = np.zeros((self.population.shape[0]), dtype=np.float32)

        for i in range(self.population.shape[0]):
            self.utilities_[i] = self.computeUtility(self.population[i])

        ordering = np.argsort(self.utilities_, axis=0)
        self.population = self.population[ordering]
        self.utilities_ = self.utilities_[ordering]

    def _iterate(self, population: np.array, absorption, annealling):
        pop_rows = population.shape[0]
        pop_cols = population.shape[1]
        for i in range(pop_rows):
            for j in range(pop_cols):
                if self.utilities_[j] > self.utilities_[i]:
                    xj = population[j]
                    xi = population[i]
                    dist = self.computeDistance(xi, xj)
                    expX = (absorption * dist * dist)
                    attractiveness = self.utilities_[j] / (1 + expX + expX * expX / 2)
                    population[i] = xi * (1 - attractiveness) + attractiveness * xj + self.step * np.random.randn(
                        1, pop_cols)

        for i in range(population.shape[0]):
            self.utilities_[i] = self.computeUtility(population[i])

        self.step *= annealling

        ordering = np.argsort(self.utilities_)[::-1]
        npop = population[ordering]
        nuti = self.utilities_[ordering]
        self.population = npop
        self.utilities_ = nuti

        self.iteration_counter += 1
        if self.iteration_counter > self.n_iterations:
            return True
        if self.step < 0.001:
            logging.info('Finishing due to setp <  0.001')
            return True
        return False
