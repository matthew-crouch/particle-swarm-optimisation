"""Particle Swarm Optimisation From Scratch"""

import numpy as np
import random
from pydantic import BaseModel, ConfigDict
import logging

logging.basicConfig(level=logging.INFO)


class SwarmConfig(BaseModel):
    """Configuration defined by the user to initialise the particle swarm"""

    lower_bound: float
    upper_bound: float
    n_dimensions: int  # Number of features
    std: float
    n_particles: int
    inertia_weight: float
    cognitive_coeff: float
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

    def __init__(self, swarm_configuration: SwarmConfig, custom_fitness_function=None):
        """Initialise the Particle Swarm Optimisation Class

        :param search_space: The search space to explore
        :param n_particles: The number of particles to use
        """
        self.config = SwarmConfig(**swarm_configuration)
        if custom_fitness_function:
            self.fitness_function = custom_fitness_function
        self.swarm_func = self._intialise_the_swarm()

    def _intialise_the_swarm(self) -> SwarmCoord:
        """Initialise the swarm with position and velocity"""
        pos = np.random.uniform(
            self.config.lower_bound,
            self.config.upper_bound,
            (self.config.n_particles, self.config.n_dimensions),
        )

        vel = (
            np.random.randn(self.config.n_particles, self.config.n_dimensions)
            * self.config.std
        )
        particle_best = pos.copy()
        particle_best_objective = np.apply_along_axis(self.fitness_function, 1, pos)
        best_idx = np.argmin(particle_best_objective)
        return SwarmCoord(
            particle_best=particle_best,
            particle_best_objective=particle_best_objective,
            global_best=particle_best[best_idx].copy(),
            global_best_objective=particle_best_objective[best_idx],
            position=pos,
            velocity=vel,
        )

    def fitness_function(self, position) -> float:
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
        # For testing purposes we choose an arbitary function. Eventually this will be replaced
        # or allow users to customise
        return np.sum((position - 3.14) ** 2) + np.sum(np.sin(3 * position + 1))

    def update(self) -> None:
        """Function to update SwarmCoordinates"""
        r1, r2 = np.random.rand(), np.random.rand()

        # Update velocity
        self.swarm_func.velocity = (
            self.config.inertia_weight * self.swarm_func.velocity
            + self.config.cognitive_coeff
            * r1
            * (self.swarm_func.particle_best - self.swarm_func.position)
            + self.config.social_coeff
            * r2
            * (self.swarm_func.global_best - self.swarm_func.position)
        )

        # Update position and keep within bounds
        self.swarm_func.position += self.swarm_func.velocity
        self.swarm_func.position = np.clip(
            self.swarm_func.position, self.config.lower_bound, self.config.upper_bound
        )

        # Compute new objective values
        objective = np.apply_along_axis(
            self.fitness_function, 1, self.swarm_func.position
        )

        # Update best particle positions where the objective improved
        improved = objective < self.swarm_func.particle_best_objective
        self.swarm_func.particle_best[improved] = self.swarm_func.position[improved]
        self.swarm_func.particle_best_objective[improved] = objective[improved]

        # Update global best
        best_idx = np.argmin(self.swarm_func.particle_best_objective)
        if (
            self.swarm_func.particle_best_objective[best_idx]
            < self.swarm_func.global_best_objective
        ):
            self.swarm_func.global_best = self.swarm_func.particle_best[best_idx].copy()
            self.swarm_func.global_best_objective = (
                self.swarm_func.particle_best_objective[best_idx]
            )

    def run(
        self, max_iterations: int, tolerance: float = 1e-6
    ) -> tuple[np.array, np.array]:
        prev_best = self.swarm_func.global_best_objective
        for _ in range(max_iterations):
            self.update()
            if abs(prev_best - self.swarm_func.global_best_objective) < tolerance:
                logging.info("PSO Algorithm has converged.")
                break
            prev_best = self.swarm_func.global_best_objective
        return self.swarm_func.global_best, self.swarm_func.global_best_objective
