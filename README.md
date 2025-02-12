# Particle Swarm Optimisation for Feature identification

## Motivation
The aim of this repo is to investigate how we can use generative algorithms like particle swarm optimisation to

1. Identify key features in a dataset
2. Identify features and values that dont occur that often.

# Setup and to run tests

```python -m venv psopy```
```source pso/bin/activate```
```pip install -r requirements.txt```
```export PYTHONPATH="particle-swarm-optimisation:${PYTHONPATH}"```
```pytest -v -s tests/test_particle_swarm_optimisation```

## Particle Swarm Optimisation

Particle Swarm Optimisation is an iterative method that optimizes a cost function, with the aim of trying to improve a candidate solution. The candidate solution are also referred to as particles, with these particles moving around a search-space by some mathematical function. The particles are initialised with some position and velocity, with each particles' movement influenced by its local best known position. The particle is also guided towards the best known position in the search space which are updated as better solutions are found by other particles. 

## Application to Machine Learning:
In machine learning applications PSO can be used as an efficient method of optimisation the initialisation weights in neural networks (https://link.springer.com/content/pdf/10.1007/s11831-021-09694-4.pdf) 

## Application to hardware verification

For hardware simulation verification, the data produced by a testbench can be very verbose and contain a large searchable space. These configurations are often pregenerated by some external process (or by an EDA tool itself)