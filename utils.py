import torch.nn as nn
import numpy as np


def action_rescale(action):
    """Rescale Distribution actions to exp one"""
    return np.array([0 if abs(a) < 0.5 else 10 ** (a-3) if a > 0  else -(10 ** (-a - 3)) for a in action * 3])


def ortog_weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def unif_weight_init(m):
    """Uniform weight init for Conv2D and Linear layers."""
    def __calc_f_in_and_f_out(layer):
        if isinstance(layer, nn.Linear):
            return layer.in_features, layer.out_features
        if isinstance(layer, nn.Conv2d):
            return layer.in_channels * np.prod(layer.kernel_size), layer.out_channels * np.prod(layer.kernel_size)
        return None

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        f_in, f_out = __calc_f_in_and_f_out(m)
        f = np.sqrt(1 / f_in)
        if f_out != 1 or f_out != 4:
            nn.init.uniform_(m.weight.data, -f, f)
            nn.init.uniform_(m.bias.data, -f, f)
        else:
            nn.init.uniform_(m.weight.data, -3e-4, 3e-4)
            nn.init.uniform_(m.bias.data, -3e-4, 3e-4)


class Replay_buffer:
    def __init__(self, size, state_dim, action_dim):
        if isinstance(state_dim, list):
            self.states = np.zeros((size, *state_dim))
        else:
            self.states = np.zeros((size, state_dim))
        self.actions = np.zeros((size, action_dim)) 
        self.rewards = np.zeros(size)  
        self.is_not_terminals = np.zeros(size) 
        self.importances = np.zeros(size)
        self.ind = 0
        self.max_size = size
        self.size = 0
    
    def add(self, state, action, rew, is_term):
        self.states[self.ind] = state
        self.actions[self.ind] = action
        self.rewards[self.ind] = rew
        self.is_not_terminals[self.ind] =  1 - is_term
        self.ind = (self.ind + 1) % self.max_size
        self.size = max(self.size, self.ind)
        
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.rewards[:]
        del self.is_not_terminals[:]
    
    def sample(self, batch_size):
        batch_ind = np.random.randint(0, self.size-1, batch_size)
                
        states_batch = self.states[batch_ind] 
        next_states_batch = self.states[batch_ind + 1] 
        actions_batch = self.actions[batch_ind]
        rewards_batch = self.rewards[batch_ind]
        is_not_term_batch = self.is_not_terminals[batch_ind]
            
        return batch_ind, states_batch, actions_batch, rewards_batch, next_states_batch, is_not_term_batch

