"""Test Particle Swarm Optimisation."""

import numpy as np
import pytest

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
}


def dummy_fitness_func(x: np.ndarray) -> np.ndarray:
    """Return fitness function.

    :param x: Input array
    :return: Output array
    """
    return (x - 3.14) ** 2 + np.sin(3 * x + 1.41)


def dummy_fitness_func2(pos: np.ndarray) -> np.ndarray:
    """Return fitness function.

    :param x: Input array
    :return: Output array
    """
    x = pos[:, 0]
    y = pos[:, 1]
    return (x - 3.14) ** 2 + (y - 2.72) ** 2 + np.sin(3 * x + 1.41) + np.sin(4 * y - 1.73)


@pytest.mark.parametrize("n_dimensions", [1, 2])
def test_particle_swarm_optimisation(n_dimensions):
    """Test the Particle Swarm Optimisation Algorithm."""
    config["n_dimensions"] = n_dimensions

    func = dummy_fitness_func if n_dimensions == 1 else dummy_fitness_func2

    pso = ParticleSwarmOptimisation(swarm_configuration=config, custom_fitness_function=func)
    global_best, global_best_objective = pso.run(max_iterations=1000000)

    if n_dimensions == 1:
        assert np.isclose(global_best, 3.1849, rtol=1e-1)
        assert np.isclose(global_best_objective, -0.9975, rtol=1e-1)
    else:
        assert np.isclose(global_best[0], 3.1849, rtol=1e-1)
        assert np.isclose(global_best[1], 3.129, rtol=1e-1)
        assert np.isclose(global_best_objective, -1.808, rtol=1e-1)


def test_initialise_swarm():
    """Test the swarm is initialised correctly."""
    pso = ParticleSwarmOptimisation(
        swarm_configuration=config, custom_fitness_function=dummy_fitness_func
    )
    assert pso.swarm_func is not None


def test_update_swarm():
    """Test that the swarm it updated."""
    pso = ParticleSwarmOptimisation(
        swarm_configuration=config, custom_fitness_function=dummy_fitness_func
    )
    initial_pos = pso.swarm_func.particle_best.copy()
    pso.update()
    with np.testing.assert_raises(AssertionError):
        assert np.testing.assert_array_equal(pso.swarm_func.position, initial_pos)
