"""
Graphing utils and other functions.
"""
import numpy as np 
import matplotlib.pyplot as plt
import random
from functools import reduce

class Visualizer:
    def __init__(self, model):
        self.model = model
        self.record = []
        self.generation = 0

    def plot_fitness(self, model):
        pass


"""
A class for parent selection strategies. Strategies include:
- Elitism: highest performing (fitness) are parents of next generation
- Fitness proportionate selection: Uniformly choose a probability of selection, and then select based off that probability
- Tourney selection: pass
"""

class Strategy:
    def __init__(self):
        pass
    
    def select_parents(self):
        pass

class ElitismStrat(Strategy):
    def __init__(self, rankings, threshold=0.25):
        self.rankings = rankings
        self.threshold = threshold

    def select_parents(self):
        population_size = len(self.rankings)
        num_of_parents = int(self.threshold * population_size)
        return [x[0] for x in self.rankings[:num_of_parents]]

class FPSStrat(Strategy):
    def __init__(self, rankings, elite_threshold=0.2):
        self.rankings = rankings
        self.elite_threshold = elite_threshold
        self.threshold_probability = random.random()

    def select_parents(self):
        parents = []
        total_fitness = reduce(lambda x,y: x[1]+y[1], self.rankings)
        fitnesses = [(x[1] / total_fitness) for x in self.rankings]
        num_of_elites = int(self.elite_threshold * len(self.rankings))
        for i in range(num_of_elites):
            parents.append(self.rankings[i][0])
        for i in range(num_of_elites, len(self.rankings) - num_of_elites):
            if self.threshold_probability <= fitnesses[i]:
                parents.append(self.rankings[i][0])
        return parents
