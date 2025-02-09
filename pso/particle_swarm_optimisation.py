"""Particle Swarm Optimisation From Scratch"""

import logging
from collections.abc import Callable

import numpy as np
from pydantic import BaseModel, ConfigDict

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

    def __init__(self, swarm_configuration: SwarmConfig, custom_fitness_function: Callable):
        """Initialise the Particle Swarm Optimisation Class

        :param search_space: The search space to explore
        :param n_particles: The number of particles to use
        """
        self.config = SwarmConfig(**swarm_configuration)
        self.fitness_function = custom_fitness_function
        self.swarm_func = self._intialise_the_swarm()

    def _intialise_the_swarm(self) -> SwarmCoord:
        """Initialise the swarm with position and velocity"""
        pos = np.random.uniform(
            self.config.lower_bound,
            self.config.upper_bound,
            (self.config.n_particles, self.config.n_dimensions),
        )

        vel = np.random.randn(self.config.n_particles, self.config.n_dimensions) * self.config.std
        particle_best = pos.copy()
        particle_best_objective = self.fitness_function(pos)

        best_idx = np.argmin(particle_best_objective)
        return SwarmCoord(
            particle_best=particle_best,
            particle_best_objective=particle_best_objective,
            global_best=particle_best[best_idx].copy(),
            global_best_objective=particle_best_objective[best_idx],
            position=pos,
            velocity=vel,
        )

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
        objective = self.fitness_function(self.swarm_func.position)

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
            self.swarm_func.global_best_objective = self.swarm_func.particle_best_objective[
                best_idx
            ]

    def run(self, max_iterations: int, tolerance: float = 1e-6) -> tuple[np.array, np.array]:
        prev_best = self.swarm_func.global_best_objective
        for _ in range(max_iterations):
            self.update()
            if abs(prev_best - self.swarm_func.global_best_objective) < tolerance:
                logging.info(" PSO Algorithm has converged.")
                break
            prev_best = self.swarm_func.global_best_objective
        return self.swarm_func.global_best, self.swarm_func.global_best_objective
