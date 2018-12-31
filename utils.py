"""
Graphing utils and other functions.
"""
import numpy as np 
import matplotlib.pyplot as plt
import random

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
        num_of_parents = int(threshold * population_size)
        return [x[0] for x in self.rankings[:num_of_parents]]

class FPSStrat(Strategy):
    def __init__(self, rankings):
        self.rankings = rankings
        self.threshold_probability = random.random()

    def select_parents(self):
        parents = []
        
