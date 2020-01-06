# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 11:48:41 2020

@author: Meer Suri
"""
import numpy as np
from scipy import stats

class State:
    
    """ Trellis states - they represent the content of the shift register 
        of the convolutional code that determines the trellis transitions
        
        Parameters -
        
        number: The state number or label - integer representation of the unsigned
        binary string represented by the shift register memory
        
        rate: Number of bits per sample. For rate r, there will be 2**r
        outgoing branches from each state at every time instant
        
        memory: The memory size of the shift register of the convolutional code
        
        path_metric: Smallest sum of branch metrics that can be achieved 
        by taking one of the many paths in the trellis that leads to this state
        
        Internals -
        
        branches: stores the states from which we have incoming branches, 
        the number of the branches between 0 and 2**r - 1,
        their corresponding reconstruction values, and the branch metrics
        
        hist: stores the state number, branch number and reconstruction value
        corresponding to the best branch at each time instant for use during
        the traceback

    """
    def __init__(self, number, rate, memory, path_metric = 0):
        self.number = number
        self.next = []
#       next states (outgoing branches) decided by LC recursion equations
        for i in range(2**rate):
            self.next.append(((2**rate)*self.number + i) % (2**memory))
        self.path_metric = path_metric
        self.next_path_metric = 0
        self.branches = []
        self.hist = []
    
    def __str__(self):
        return "state = " + str(self.state) 


class SR_LC_Int_Quantizer:
    
    """ Shift Register Linear Congruential Code Quantizer with integer number of
        bits per sample quantized.
        
        Parameters -
        
        rate: Integer number of bits per sample      
        
        memory: The memory size of the shift register of the convolutional code
        corresponding to this quantizer
        
        distortion_measure: function of true value and reconstruction value used
        to measure the closeness of the two
        
        mu: mean of Gaussian distributed input samples
        
        s: standard deviation of the Gaussian distributed input samples
        
        c_scale: constant scaling factor for the reconstruction values. Usually
        very close to 1 for iid Gausian sources
    """
        
        
    def __init__(self, memory, rate, mu, s, c_scale = 1, distortion_measure = 'mse'):
        self.rate = rate
        self.memory = memory
        self.mu = mu
        self.s = s
        self.distortion_measure = distortion_measure
        self.states = []
        self.labels = list(range(2**memory))
        self.c_sf = c_scale
#        reconstruction values according to the inverse CDF. This creates 
#        values that match the source distribution.
        q = [(2*label + 1)/(2*(2**memory)) for label in self.labels]
        self.y = self.c_sf * stats.norm.ppf(q, loc = mu, scale = s)
        self.num_to_state = {}
#       multiplier and offset coefficients for LC code recursions, for various 
#       rates and memory sizes
        self.lc_coeff_r_1 = {2: [(5, 2), (5, 0)],
                             3: [(5, 7), (5, 3)],
                             4: [(5, 8), (5, 0)]}
        
        self.lc_coeff_r_2 = {2: [(5, 3), (5, 2), (5, 1), (5, 4)],
                             3: [(5, 7), (5, 1), (5, 3), (5, 5)],
                             4: [(5, 15), (5, 4), (5, 7), (5, 12)]}
        
        self.lc_coeff_r_3 = {3: [(5, 7), (5, 1), (5, 3), (5, 5), (5, 2), (5, 6), (5, 0), (5, 4)],
                             4: [(5, 15), (5, 6), (5, 11), (5, 4), (5, 7), (5, 9), (5, 3), (5, 12)]}
        
        self.lc_coeff = {1: self.lc_coeff_r_1, 2: self.lc_coeff_r_2, 3:self.lc_coeff_r_3}
        
#       init all states except for zero state with infinite path metric as starting 
#       state of shift register is all zeros
        for i in range(2**memory):
            if i == 0: 
                self.states.append(State(i, rate, memory, 0))
            else:
                self.states.append(State(i, rate, memory, np.inf))
            self.num_to_state[i] = self.states[-1]
                
    def dist(self, x, xh):
        assert self.distortion_measure == 'mse', 'Only MSE supported currently'
        if self.distortion_measure == 'mse':
            return (x - xh)**2
        
    def encode(self, x):
        assert len(x) > 0, 'Input sequence is empty'
        self.input = x
        n = len(x)
#         forward pass through the trellis to calculate all branch metrics and 
#         save the branches corresponding to the best path(lowest overall distortion)
        for i in range(n):
            for state in self.states:
                for k in range(len(state.next)):
#                    multiplier and offsets for LC label recursion
                    g, a  = self.lc_coeff[self.rate][self.memory][k]
#                    LC recursion for label generation
                    label = (g * state.number + a) % 2**self.memory
#                    The labels are mapped one-to-one to the reconstruction values
                    rxn = self.y[label]
                    next_state = self.num_to_state[state.next[k]]
#                    branch metric
                    bm = self.dist(x[i], rxn)
                    next_state.branches.append((state.number, k, rxn, bm))
            
#            find best branch for each state at every timestep
            for state in self.states:
                min_state = None
                min_val = np.inf
                best_rxn = np.inf
                branch_num = None
                for i in range(len(state.branches)):
                    num, k, rxn, bm = state.branches[i]
                    pm = self.num_to_state[num].path_metric
                    if pm + bm < min_val:
                        min_state = num
                        min_val = pm + bm
                        best_rxn = rxn
                        branch_num = k
                state.next_path_metric = min_val
                state.hist.append((min_state, branch_num, best_rxn))
            
            for state in self.states:
                state.path_metric = state.next_path_metric
                state.branches = []
    
        
#         traceback through the trellis along the best path (lowest overall distortion)
        pms = [state.path_metric for state in self.states]
        state = self.states[np.argmin(pms)]
        self.quant_repr = []
        self.quant_val = []
        
#        follow the best branches and store quantized representation and 
#        reconstructed representation 
        for i in range(n):
            num, br, rxn = state.hist[-1 - i]
            self.quant_repr.insert(0, br)
            self.quant_val.insert(0, rxn)
            state = self.num_to_state[num]

        # calculate overall distortion as absolute value and SQNR
        self.distortion = [ self.dist(self.input[i], self.quant_val[i]) for i in range(n)]
        self.distortion = np.mean(self.distortion)
        self.sqnr = 10*np.log10((self.s**2)/self.distortion)
        
        return self.quant_repr
        
    def reset(self):
#        clear history of best brach for each state at each timestep
        for state in self.states:
            state.hist = []
        
class SR_LC_Int_Reconstructor:
    """ Reconstructs the sequence from its quantized representation (decoder)
        The quantized representation simply stores the number(between 0 and 2**r - 1)
        of the outgoing branch from the current state, for each timestep.
        Reconstruction is a forward pass through the trellis, storing the
        recontruction values corresponding to the picked branch at each timestep        
    """
    def __init__(self, memory, rate, mu, s, c_scale = 1):
        self.rate = rate
        self.memory = memory
        self.mu = mu
        self.s = s
        self.states = []
        self.labels = list(range(2**memory))
        self.c_sf = c_scale
        q = [(2*label + 1)/(2*(2**memory)) for label in self.labels]
        self.y = self.c_sf * stats.norm.ppf(q, loc = mu, scale = s)
        self.num_to_state = {}
        self.lc_coeff_r_1 = {2: [(5, 2), (5, 0)],
                             3: [(5, 7), (5, 3)],
                             4: [(5, 8), (5, 0)]}
        
        self.lc_coeff_r_2 = {2: [(5, 3), (5, 2), (5, 1), (5, 4)],
                             3: [(5, 7), (5, 1), (5, 3), (5, 5)],
                             4: [(5, 15), (5, 4), (5, 7), (5, 12)]}
        
        self.lc_coeff_r_3 = {3: [(5, 7), (5, 1), (5, 3), (5, 5), (5, 2), (5, 6), (5, 0), (5, 4)],
                             4: [(5, 15), (5, 6), (5, 11), (5, 4), (5, 7), (5, 9), (5, 3), (5, 12)]}
        
        self.lc_coeff = {1: self.lc_coeff_r_1, 2: self.lc_coeff_r_2, 3:self.lc_coeff_r_3}
        
        for i in range(2**memory):
            if i == 0: 
                self.states.append(State(i, rate, memory))
            else:
                self.states.append(State(i, rate, memory))
            self.num_to_state[i] = self.states[-1]
            
    def decode(self, quant_repr, init_state = 0):
        assert len(quant_repr) > 0, 'Input sequence is empty'
        self.input = quant_repr
        self.out = []
        n = len(quant_repr)
        state = self.num_to_state[init_state]
        for i in range(n):
            branch = self.input[i]
            g, a  = self.lc_coeff[self.rate][self.memory][branch]
            label = (g * state.number + a) % 2**self.memory
            rxn = self.y[label]
            state = self.num_to_state[state.next[branch]]
            self.out.append(rxn)
			
        return self.out