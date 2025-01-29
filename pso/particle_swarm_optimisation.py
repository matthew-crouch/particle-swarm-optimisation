"""Particle Swarm Optimisation From Scratch"""

import numpy as np
import random


def generate_large_config():
    """Generate a large configuration for testing purposes"""
    return {
        f"config_{i}": random.choice(
            ["A", "B", "C", "D", 1, 2, 3, 4, [123], [124], [125], ["bro"], ["cat"]]
        )
        for i in range(1000)
    }


class ParticleSwarmOptimisation:
    """Particle Swarm Optimisation Algorithm

    This class will try to find the least explored configuration for a given search space
    """

    def __init__(self, search_space: dict, n_particles: int):
        """Initialise the Particle Swarm Optimisation Class

        :param search_space: The search space to explore
        :param n_particles: The number of particles to use
        """
        self.particles = self._intialise_the_swarm(search_space, n_particles)

    def _intialise_the_swarm(self, search_space: dict, n_particles: int):
        """Initialise the swarm

        :param search_space: The search space to explore
        :param n_particles: The number of particles to use
        :return: The initial particles
        """
        particles = np.random.sample(search_space, n_particles)
        return particles

    def update_best_values(self):
        """Update the best values for each particle and the global best value"""
        raise NotImplementedError

    def fitness_function(self, x):
        """The fitness function to be minimised

        For categorical variables and search space configurations,
        we will aim to calculate the exploration count. We will want
        to move towards configurations with lower exploration count

        There are different options we can consider here depending on the
        problem we would like to solve.
            1. Exploration Count
            2. Similarity-Based Fitness Function
            3. Bayesian Optimisation
        """
        raise NotImplementedError

    def evaluate(self, x):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def optimise(self, n_iterations):
        raise NotImplementedError
