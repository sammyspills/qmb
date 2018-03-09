# -*- coding: utf-8 -*-
"""
author: Sam Spillard
python script to complete final project of Quantum Many-Body Physics module
Simulate experiment by Kaufman et al. 2016
"""

from scipy.special import factorial as fact
import numpy as np
import itertools

def vec2bin(vec):
    state_str = ''.join(map(str, vec))
    idx = int(state_str, 2)
    return idx

class StateObj:
    def __init__(self, init_vec, idx):
        self.vector = np.asarray(init_vec)
        self.idx = idx
    
    prefactor = 1

def creationOp(state, idx, N):
    a = state.vector[idx]
    if(a == N):
        state.prefactor = 0
    else:
        state.vector[idx] += 1
    return state

def annihilationOp(state, idx):
    a = state.vector[idx]
    if(a == 0):
        state.prefactor = 0
    else:
        state.vector[idx] -+ 1
    return state

def numberOp(state, idx):
    if(state.vector[idx] == 0):
        state.prefactor = state.prefactor * -1
    return state

def getBasisStates(N, M):
    
    basis, count = [], 0    
    x = itertools.product(range(N + 1), repeat=M)
    for i in x:
        if(np.sum(i) == N):
            state = StateObj(i, count)
            basis.append(state)
            count += 1
    
    return np.asarray(basis)

def actHam(state, N, J, U):
    return_state_list = []
    # First term
    for i in range(len(state.vector)-1):
        return_state_list.append()

def getHamMatrix(basis, J, U):
    ham_matrix = np.zeros((len(basis), len(basis)))
    
    return ham_matrix
    
if(__name__ == '__main__'):
    print('In module.')
    basis = getBasisStates(2, 3)