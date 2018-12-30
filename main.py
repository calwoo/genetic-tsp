"""
Goal is to build a genetic algorithm to find optimal solutions to the traveling salesman problem.
To recall, the traveling salesman problem asks for optimal tours about a group of cities.
- Calvin Woo, 2018
"""

import numpy as np
import random
import matplotlib.pyplot as plt

"""
We need classes to store city nodes and the genetic algorithm.
The city nodes will be implemented as points in a 2d grid where distances are given by the L2 metric.
Alternatively, we could model it as a list of cities with another list of pairs with travelling costs.
"""
class CityNode:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.coords = np.array([self.x, self.y])

    def dist_to(self, city):
        d = np.sqrt((self.coords - city.coords) ** 2)
        return d

"""
The evolution class stores the methods used to implement the genetic algorithm. A genetic algorithm consists of
- A specification of an individual of a population
- A way to generate a population
- A way to measure the fitness of an individual
- Parent selection
- A mating method
- Random mutation method to introduce explorative variation
"""
class Evolution:
    def __init__(self, cities):
        self.cities = cities

    def create_individual(self):
        
