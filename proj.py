# -*- coding: utf-8 -*-
"""
author: Sam Spillard
python script to complete final project of Quantum Many-Body Physics module
Simulate experiment by Kaufman et al. 2016
"""

from scipy.special import factorial as fact
import numpy as np

def vec2bin(vec):
    state_str = ''.join(map(str, vec))
    idx = int(state_str, 2)
    return idx

class StateObj:
    def __init__(self, init_vec, idx):
        self.vector = np.asarray(init_vec)
        self.idx = idx
        
    @property
    def  binIndex(self):
        return vec2bin(self.vector)
    
    prefactor = 1
    
def getBasisStates(N, M):
    num = fact(N + M - 1)/(fact(N) * fact(M - 1))
    basis = np.stack([np.zeros(M) for i in range(int(num))])
    returnDict = {}
    
    return basis
    
if(__name__ == '__main__'):
    print('In module.')
    print(getBasisStates(2, 2))