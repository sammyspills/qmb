# -*- coding: utf-8 -*-
"""
author: Sam Spillard
python script to complete final project of Quantum Many-Body Physics module
Simulate experiment by Kaufman et al. 2016

TODO: Get RDM
"""

from scipy.special import factorial as fact
import scipy as sp
import numpy as np
import itertools
from copy import deepcopy

class tColors:
    WARN = '\x1b[93m'
    FAIL = '\x1b[91m'
    ENDC = '\x1b[0m'

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
                    ham_matrix[state.idx][b.idx] = ham_matrix[state
                              .idx][b.idx] + x.prefactor
                    
    return ham_matrix, basis

def getInitialState(N, M, count=0):
    LENGTH = int((fact(N+M-1))/(fact(N)*fact(M-1)))
    PREAMBLE = """Enter initial state configuration (e.g. "1, 0, 0").
Note: This is the normalised vector representing the superposition of basis
states.
The basis states are ordered in ascending order of their integer representation
e.g. the state (0, 0, 2) -> "2" will be before the state (0, 1, 1) -> "11" will
be before the state (2, 0, 0) -> "200".
Should have length: C(N+M-1, N)=""" + str(LENGTH) + ': '
    initialStateVec = np.asarray([float(x) for x in input(PREAMBLE)
        .split(', ')])
    if(len(initialStateVec) != LENGTH):
        print(tColors.FAIL + '\nLength of initial state should be C(N+M-1,'
            +'N)\n' + tColors.ENDC)
        input('Press Enter to continue...')
        count += 1
        initialStateVec = getInitialState(N,M,count=count)
        
    return initialStateVec

def newState(initialStateVec, hamMatrix, t):
    expMat = -1j * hamMatrix * t
    expMat = sp.linalg.expm(expMat)
    vNewState = np.dot(expMat, initialStateVec)
    tempSum = np.sum(vNewState)
    vNewState = vNewState / tempSum
    return vNewState
    
def convertToBoson(M, vec, basis):
    bosons = np.zeros(M, dtype=complex)
    for i in range(len(vec)):
        print(vec[i] * basis[i].vector)
        val = vec[i] * basis[i].vector
        bosons += val
    return bosons

def getDensityMatrix(state, N, M):
    LENGTH = len(state)
    ASIZE = N+1
    BSIZE = LENGTH - ALENGTH
    
    c_matrix = np.zeros((ASIZE, 2**BLENGTH))
    print("'A' system - " + str(BLENGTH) + " site(s).")
    
    return

if(__name__ == '__main__'):
    print('In module.')
    N, M, J, U = [float(x) for x in input(
            'Enter params (comma sep e.g. "2, 2, 1, 1"):  ').split(', ')]
    N, M = int(N), int(M)
    initialState = getInitialState(N, M)
    hamMatrix, basis = getHamMatrix(N, M, J, U)
    tState = newState(initialState, hamMatrix, 10e-3)
    tStateBoson = convertToBoson(M, tState, basis)
    print('New State: ')
    print(tStateBoson)