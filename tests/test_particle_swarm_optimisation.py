"""Test Particle Swarm Optimisation"""

from pso.particle_swarm_optimisation import (
    ParticleSwarmOptimisation,
)
import numpy as np

config = {
    "n_particles": 20,
    "lower_bound": 0,
    "upper_bound": 5,
    "std": 0.1,
    "inertia_weight": 0.8,
    "cognitive_coeff": 0.1,
    "social_coeff": 0.1,
    "n_dimensions": 20,
}


def test_particle_swarm_optimisation():
    """Test the Particle Swarm Optimisation Algorithm"""
    pso = ParticleSwarmOptimisation(swarm_configuration=config)
    pso.run(max_iterations=100)


def test_initialise_swarm():
    """Test the swarm is initialised correctly"""
    pso = ParticleSwarmOptimisation(swarm_configuration=config)
    assert pso.swarm_func is not None


def test_update_swarm():
    """Test that the swarm it updated"""
    pso = ParticleSwarmOptimisation(swarm_configuration=config)
    initial_pos = pso.swarm_func.particle_best.copy()
    pso.update()
    with np.testing.assert_raises(AssertionError):
        assert np.testing.assert_array_equal(pso.swarm_func.position, initial_pos)
