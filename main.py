"""
Goal is to build a genetic algorithm to find optimal solutions to the traveling salesman problem.
To recall, the traveling salesman problem asks for optimal tours about a group of cities.
- Calvin Woo, 2018
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import utils

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
        # Return a permutation of the cities list
        permutation = random.sample(cities, len(cities))
        return permutation

    def generate_population(self, size):
        population = []
        for _ in range(size):
            new_individual = self.create_individual()
            population.append(new_individual)
        return population

    def score(self, individual):
        total_dist = 0
        for i in range(len(individual)):
            if i == len(individual) - 1:
                total_dist += individual[i].dist_to(individual[0])
            else:
                total_dist += individual[i].dist_to(individual[i+1])
        return total_dist
    
    def fitness(self, individual):
        s = self.score(individual)
        return 1.0 / s

    def population_avg_fitness(self, population):
        pass

    def rank(self, population):
        fitnesses = list(map(lambda x: self.fitness(x), population))
        zipped = zip(population, fitnesses)
        return sorted(zipped, key=lambda x: x[1], reverse=True)
    
    
