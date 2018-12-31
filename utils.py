"""
Graphing utils and other functions.
"""
import numpy as np 
import matplotlib.pyplot as plt 

class Visualizer:
    def __init__(self, model):
        self.model = model
        self.record = []
        self.generation = 0

    def plot_fitness(self, model):