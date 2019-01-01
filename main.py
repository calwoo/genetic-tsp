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
    def __init__(self, cities, population_size, elite_threshold=0.2):
        self.cities = cities
        self.elite_threshold = elite_threshold
        self.population_size = population_size

    def create_individual(self):
        # Return a permutation of the cities list
        permutation = random.sample(cities, len(cities))
        return permutation

    def generate_population(self):
        population = []
        for _ in range(self.population_size):
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
        fitnesses = list(map(lambda ind: self.fitness(ind), population))
        fitnesses = np.array(fitnesses)
        return np.mean(fitnesses)

    def rank(self, population):
        fitnesses = list(map(lambda x: self.fitness(x), population))
        zipped = zip(population, fitnesses)
        return sorted(zipped, key=lambda x: x[1], reverse=True)

    def select_parents(self, population, strategy):
        ranked = self.rank(population)
        return strategy.select(ranked)

    def mate(self, parents):
        """
        The strategy for mating for the travelling salesman problem will be that we select randomly a subset of
        the first parent and then fill in the remainder of the route with the second parent in order,
        ensuring that the resulting route is a valid one (unique cities, visited only once).
        """
        num_to_retain = int(self.population_size * self.elite_threshold)
        children = parents[:num_to_retain]
        for i in range(self.population_size - num_to_retain):
            child = self.create_child(parents[i], parents[self.population_size-i-1])
            children.append(child)
        return children

    def create_child(self, parent1, parent2):
        route_length = len(self.cities)
        thres1 = int(random.random() * route_length)
        thres2 = int(random.random() * route_length)
        gene_low, gene_high = min(thres1, thres2), max(thres1, thres2)

        child = []
        parent1_chromosome = parent1[gene_low:gene_high]
        parent2_chromosome = [node for node in parent2 if node not in parent1_chromosome]
        next_gene = 0
        for i in range(route_length):
            if i in range(gene_low, gene_high):
                child.append(parent1[i])
            else:
                child.append(parent2_chromosome[next_gene])
                next_gene += 1
        return child

    def mutate(self, child, mutation_chance=0.05):
        """
        Mutation of individuals in the population proceed by swapping a pair of cities in the route randomly.
        """
        for i in range(len(self.cities)):
            prob = random.random()
            if prob < mutation_chance:
                swap_index = random.randint(len(self.cities))
                # Do the swap
                temp = child[swap_index]
                child[swap_index] = child[i]
                child[i] = temp
        return child

    def mutate_population(self, population, mutation_chance=0.05):
        mutated_pop = []
        for i in range(self.population_size):
            individual = population[i]
            mutated_individual = self.mutate(individual, mutation_chance)
            mutated_pop.append(mutated_individual)
        return mutated_pop
        


    
    
