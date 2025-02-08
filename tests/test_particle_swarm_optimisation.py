"""Test Particle Swarm Optimisation"""

import numpy as np

from pso.particle_swarm_optimisation import (
    ParticleSwarmOptimisation,
)

config = {
    "n_particles": 20,
    "lower_bound": 0,
    "upper_bound": 5,
    "std": 0.1,
    "inertia_weight": 0.8,
    "cognitive_coeff": 0.1,
    "social_coeff": 0.1,
    "n_dimensions": 2,
}


def dummy_fitness_func(x):
    return (x - 3.14) ** 2 + np.sin(3 * x + 1.41)


def test_particle_swarm_optimisation():
    """Test the Particle Swarm Optimisation Algorithm"""
    pso = ParticleSwarmOptimisation(
        swarm_configuration=config, custom_fitness_function=dummy_fitness_func
    )
    global_best, global_best_objective = pso.run(max_iterations=100000)

    assert np.isclose(global_best, 3.1849, rtol=1e-1)
    assert np.isclose(global_best_objective, -0.9975, rtol=1e-1)


def test_initialise_swarm():
    """Test the swarm is initialised correctly"""
    pso = ParticleSwarmOptimisation(
        swarm_configuration=config, custom_fitness_function=dummy_fitness_func
    )
    assert pso.swarm_func is not None


def test_update_swarm():
    """Test that the swarm it updated"""
    pso = ParticleSwarmOptimisation(
        swarm_configuration=config, custom_fitness_function=dummy_fitness_func
    )
    initial_pos = pso.swarm_func.particle_best.copy()
    pso.update()
    with np.testing.assert_raises(AssertionError):
        assert np.testing.assert_array_equal(pso.swarm_func.position, initial_pos)
