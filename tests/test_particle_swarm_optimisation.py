"""Test Particle Swarm Optimisation"""

from pso.particle_swarm_optimisation import (
    ParticleSwarmOptimisation,
    generate_large_config,
)
import numpy as np
import pandas as pd
import uuid


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


def test_intialise_swarm():
    """Generate Sample Data"""
    sample_data = [generate_large_config() for x in range(20000)]
    breakpoint()
    pso = ParticleSwarmOptimisation(search_space=sample_data, n_particles=20)
    breakpoint()
