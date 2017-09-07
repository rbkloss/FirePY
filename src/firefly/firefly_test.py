import unittest
import logging
import numpy

from .firefly import Firefly

np = numpy


def _dist(v1, v2):
    return np.linalg.norm(v1 - v2, 2)


class FireflyTestCase(unittest.TestCase):
    def test_cubic_root(self):
        firefly = Firefly(population_size=100, n_dims=1, n_iterations=500, utility=lambda pop: -abs(pop ** 3 - 2),
                          distance=_dist)
        firefly.fit(verbose=True)
        population = firefly.population
        util = firefly.utilities_
        ordering = np.argsort(util)
        population = population[ordering][::-1]
        util = util[ordering][::-1]

        logging.info('utils:\n{}'.format(util.tolist()))
        logging.info('pop:\n{}'.format(population.tolist()))

        ans = population[0, 0]

        self.assertAlmostEqual(ans, 2 ** (1 / 3), delta=1e-1)

    def test_square_root(self):
        firefly = Firefly(population_size=100, n_dims=1, n_iterations=500, utility=lambda pop: -abs(pop ** 2 - 2),
                          distance=_dist)

        firefly.fit(verbose=True)
        population = firefly.population
        util = firefly.utilities_
        ordering = np.argsort(util)
        population = population[ordering][::-1]
        util = util[ordering][::-1]

        logging.info('utils:\n{}'.format(util.tolist()))
        logging.info('pop:\n{}'.format(population.tolist()))

        ans = abs(population[0, 0])

        self.assertAlmostEqual(ans, 2 ** 0.5, delta=1e-1)


if __name__ == '__main__':
    unittest.main()
