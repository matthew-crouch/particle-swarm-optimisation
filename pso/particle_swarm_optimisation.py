"""Particle Swarm Optimisation From Scratch"""

import numpy as np
import random
from pydantic import BaseModel, ConfigDict


def generate_large_config():
    """Generate a large configuration for testing purposes"""
    return {
        f"config_{i}": random.choice(
            ["A", "B", "C", "D", 1, 2, 3, 4, [123], [124], [125], ["bro"], ["cat"]]
        )
        for i in range(1000)
    }


class SwarmConfig(BaseModel):
    """Configuration defined by the user to initialise the particle swarm"""

    upper_bound: float
    std: float
    n_particles: int
    inertia_weight: float
    cognative_coeff: float
    social_coeff: float


class SwarmCoord(BaseModel):
    """DataModel to contain particle best and global best"""

    particle_best: np.array
    particle_best_objective: np.array
    global_best: np.array
    global_best_objective: float
    position: np.array
    velocity: np.array

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ParticleSwarmOptimisation:
    """Particle Swarm Optimisation Algorithm

    This class will try to find the least explored configuration for a given search space
    """

    def __init__(self, swarm_configuration: SwarmConfig):
        """Initialise the Particle Swarm Optimisation Class

        :param search_space: The search space to explore
        :param n_particles: The number of particles to use
        """
        self.config = SwarmConfig(**swarm_configuration)
        self.swarm_func = self._intialise_the_swarm()

    def _intialise_the_swarm(self) -> SwarmCoord:
        """Initialise the swarm with position and velocity"""
        pos = np.random.rand(2, self.config.n_particles) * self.config.upper_bound
        vel = np.random.randn(2, self.config.n_particles) * self.config.std
        particle_best = pos
        particle_best_objective = self.fitness_function(pos[0], pos[1])
        global_best = particle_best[:, particle_best_objective.argmin()]
        global_best_objective = particle_best_objective.min()
        return SwarmCoord(
            particle_best=particle_best,
            particle_best_objective=particle_best_objective,
            global_best=global_best,
            global_best_objective=global_best_objective,
            position=pos,
            velocity=vel,
        )

    def fitness_function(self, x, y):
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
        return (x - 3.14) ** 2 + (y - 2) ** 2 + np.sin(3 * x + 1) + np.sin(4 * y - 1.7)

    def update(self) -> None:
        """Function to update SwarmCoordinates"""
        r1, r2 = np.random.rand(2)
        velocity = (
            self.config.inertia_weight * self.swarm_func.velocity
            + self.config.cognative_coeff
            * r1
            * (self.swarm_func.particle_best - self.swarm_func.position)
            + self.config.social_coeff
            * r2
            * (self.swarm_func.global_best.reshape(-1, 1) - self.swarm_func.position)
        )

        # update new particle position
        self.swarm_func.position = self.swarm_func.position + velocity
        objective = self.fitness_function(
            self.swarm_func.position[0], self.swarm_func.position[1]
        )

        # update best particle and global
        self.swarm_func.particle_best[
            :, (self.swarm_func.particle_best_objective >= objective)
        ] = self.swarm_func.position[
            :, (self.swarm_func.particle_best_objective >= objective)
        ]

        self.swarm_func.particle_best_objective = np.array(
            [self.swarm_func.particle_best_objective, objective]
        ).min(axis=0)
        self.swarm_func.global_best = self.swarm_func.particle_best[
            :, self.swarm_func.particle_best_objective.argmin()
        ]
        self.swarm_func.global_best_objective = (
            self.swarm_func.particle_best_objective.min()
        )

    def run(self, n_interations: int) -> None:
        for i in range(n_interations):
            self.update()
