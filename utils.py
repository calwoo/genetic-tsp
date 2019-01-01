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
        self.threshold = threshold

    def select_parents(self, rankings):
        population_size = len(rankings)
        num_of_elites = int(self.threshold * population_size)
        elites = rankings[:num_of_elites]
        parents = []
        counter = 0
        while len(parents) < population_size:
            parents.append(elites[counter])
            counter += 1
            if counter == len(elites):
                counter = 0
        return parents

class FPSStrat(Strategy):
    def __init__(self, elite_threshold=0.2):
        self.elite_threshold = elite_threshold

    def cumulative_fitness(self, rankings):
        total_fitness = reduce(lambda x,y: x[1]+y[1], rankings)
        cumulative = []
        running_total = 0
        for _, fitness in rankings:
            running_total += fitness
            cumulative.append(running_total)
        cumulative = np.array(cumulative)
        return cumulative / total_fitness
        
    def select_parents(self, rankings):
        parents = []
        cumulative_fitnesses = self.cumulative_fitness(rankings)
        num_of_elites = int(self.elite_threshold * len(rankings))
        for i in range(num_of_elites):
            parents.append(rankings[i][0])
        for i in range(num_of_elites, len(rankings)):
            threshold_probability = random.random()
            for j in range(len(rankings)):
                if threshold_probability <= cumulative_fitnesses[i]:
                    parents.append(rankings[j][0])
                    break
        return parents
