import numpy as np


class Normal_noise:
    def __init__(self, action_dimension, scale=0.1, mu=0):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
    
    def reset(self):
        pass

    def noise(self):
        return np.random.normal(self.mu, self.scale, (self.action_dimension))