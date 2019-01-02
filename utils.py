"""
Graphing utils and other functions.
"""
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from functools import reduce
from matplotlib import style

style.use('ggplot')
fig = plt.figure()

class Visualizer:
    def __init__(self):
        self.record = []
        self.generation = 0
        self.fitnesses = []
    
    def reset_visualizer(self):
        self.record = []
        self.generation = 0
        self.fitnesses = []

    def update(self, fitness):
        self.fitnesses.append((self.generation, fitness))
        self.generation += 1

    def plot_fitness(self, i):
        plt.plot(list(map(lambda x:x[1], self.fitnesses)))
        plt.ylabel("Fitness")
        plt.xlabel("Generation")
        plt.show()

    def plot_scores(self, i):
        plt.plot(list(map(lambda x: 1/x[1], self.fitnesses)))
        plt.ylabel("Average Distance")
        plt.xlabel("Generation")
        plt.show()

    def animate_plot(self, plot_type="fitness"):
        if plot_type == "fitness":
            func = lambda i: self.plot_fitness(i)
        elif plot_type == "scores":
            func = lambda i: self.plot_scores(i)
        ani = animation.FuncAnimation(fig, func, interval=1000)
        plt.show()

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
        total_fitness = reduce(lambda x,y: x+y[1], rankings, 0)
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
                if threshold_probability <= cumulative_fitnesses[j]:
                    parents.append(rankings[j][0])
                    break
        return parents
