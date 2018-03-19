# -*- coding: utf-8 -*-
"""
author: Sam Spillard
python script to complete final project of Quantum Many-Body Physics module
Simulate experiment by Kaufman et al. 2016
"""

from scipy.special import factorial as fact
import scipy as sp
import numpy as np
import itertools
from copy import deepcopy

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
    trans = deepcopy(state)
    a = trans.vector[idx]
    if(a == N):
        trans.prefactor = 0
    else:
        trans.vector[idx] += 1
    return trans

def annihilationOp(state, idx):
    trans = deepcopy(state)
    a = trans.vector[idx]
    if(a == 0):
        trans.prefactor = 0
    else:
        trans.vector[idx] -= 1
    return trans

def numberOp(state, idx):
    if(state.vector[idx] == 0):
        return(-1)
    return 1

def getBasisStates(N, M):
    
    basis, sums, states, count = [], [], [], 0    
    x = itertools.product(range(N + 1), repeat=M)
    for i in x:
        if(np.sum(i) == N):
            basis.append(np.asarray(i, dtype=int))
            val = int(''.join(np.asarray(i, dtype=str)))
            sums.append(val)
            
    for i in range(len(sums)):
        idx = np.argmin(sums)
        state = StateObj(basis[idx], count)
        states.append(state)
        count += 1
        sums[idx] = np.inf
        
    return np.asarray(states)

def actHam(state, N, J, U):
    first_term, second_term = [], []
    # First term
    for i in range(len(state.vector)-1):
        temp = creationOp(state, i+1, N)
        first_term.append(annihilationOp(temp, i))
        
        temp2 = creationOp(state, i, N)
        first_term.append(annihilationOp(temp2, i+1))
        
    for i in range(len(state.vector)):
        new_pre = (numberOp(state, i) ** 2) - numberOp(state, i)
        trans = deepcopy(state)
        trans.prefactor = trans.prefactor * new_pre
        second_term.append(trans)
    
    for x in first_term:
        x.prefactor = x.prefactor * J * -1
    for x in second_term:
        x.prefactor = x.prefactor * U/2
        
    return np.r_[first_term, second_term]

def getHamMatrix(N, M, J, U):
    basis = getBasisStates(N, M)
    ham_matrix = np.zeros((len(basis), len(basis)))
    for state in basis:
        acted = actHam(state, N, J, U)
        for x in acted:
            for b in basis:
                if(np.all(x.vector == b.vector)):
                    ham_matrix[state.idx][b.idx] = ham_matrix[state.idx][b.idx] + x.prefactor
                    
    return ham_matrix

def getInitialState(N, M):
    
    initialStateVec = [int(x) for x in input('Enter initial state configuration (e.g. "1, 1, 1, 1, 2, 0"):  ').split(', ')]
    if(len(initialStateVec) != M):
        raise ValueError('Length of initial state should match M')
    if(np.sum(initialStateVec) != N):
        raise ValueError('Total number of bosons should match N')
        
    return StateObj(initialStateVec, 0)

def expHam(hamMatrix, t):
    new_mat = -1j * hamMatrix * t
    return sp.linalg.expm(new_mat)
    
if(__name__ == '__main__'):
    print('In module.')
    N, M, J, U = [float(x) for x in input('Enter params (comma sep e.g. "2, 2, 1, 1"):  ').split(', ')]
    N, M = int(N), int(M)
    initialState = getInitialState(N, M)
    hamMatrix = getHamMatrix(N, M, J, U)
    print(expHam(hamMatrix, 1))