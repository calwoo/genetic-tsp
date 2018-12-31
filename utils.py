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
        self.population_size = len(self.rankings)

    def select_parents(self):
        population_size = len(self.rankings)
        num_of_elites = int(self.threshold * population_size)
        elites = self.rankings[:num_of_elites]
        parents = []
        counter = 0
        while len(parents) < population_size:
            parents.append(elites[counter])
            counter += 1
            if counter == len(elites):
                counter = 0
        return parents

class FPSStrat(Strategy):
    def __init__(self, rankings, elite_threshold=0.2):
        self.rankings = rankings
        self.elite_threshold = elite_threshold

    def cumulative_fitness(self):
        total_fitness = reduce(lambda x,y: x[1]+y[1], self.rankings)
        cumulative = []
        running_total = 0
        for _, fitness in self.rankings:
            running_total += fitness
            cumulative.append(running_total)
        cumulative = np.array(cumulative)
        return cumulative / total_fitness
        
    def select_parents(self):
        parents = []
        cumulative_fitnesses = self.cumulative_fitness()
        num_of_elites = int(self.elite_threshold * len(self.rankings))
        for i in range(num_of_elites):
            parents.append(self.rankings[i][0])
        for i in range(num_of_elites, len(self.rankings)):
            threshold_probability = random.random()
            for j in range(len(self.rankings)):
                if threshold_probability <= cumulative_fitnesses[i]:
                    parents.append(self.rankings[j][0])
                    break
        return parents
