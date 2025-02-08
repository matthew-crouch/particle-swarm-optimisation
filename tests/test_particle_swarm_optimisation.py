"""Test Particle Swarm Optimisation"""

from pso.particle_swarm_optimisation import (
    ParticleSwarmOptimisation,
)
import numpy as np


def test_particle_swarm_optimisation():
    """Test the Particle Swarm Optimisation Algorithm"""
    pass


def test_update_best_values():
    """Test the update_best_values method"""
    pass


def test_evaluate():
    """Test the evaluate method"""
    pass


def test_optimize():
    """Test the optimise method"""
    pass


def test_initialise_swarm():
    config = {
        "n_particles": 20,
        "upper_bound": 5,
        "std": 0.1,
        "inertia_weight": 0.8,
        "cognative_coeff": 0.1,
        "social_coeff": 0.1,
    }
    pso = ParticleSwarmOptimisation(swarm_configuration=config)
    assert pso.swarm_func is not None


def test_update_swarm():
    config = {
        "n_particles": 20,
        "upper_bound": 5,
        "std": 0.1,
        "inertia_weight": 0.8,
        "cognative_coeff": 0.1,
        "social_coeff": 0.1,
    }
    pso = ParticleSwarmOptimisation(swarm_configuration=config)
    initial_pos = pso.swarm_func.particle_best.copy()
    pso.update()
    with np.testing.assert_raises(AssertionError):
        assert np.testing.assert_array_equal(pso.swarm_func.position, initial_pos)
