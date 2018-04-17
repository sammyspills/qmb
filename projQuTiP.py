#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 12:18:38 2018

@author: sammyspills
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from scipy.special import factorial as fact
import itertools
import qutip as qt

class tColors:
    WARN = '\x1b[93m'
    FAIL = '\x1b[91m'
    ENDC = '\x1b[0m'
    
def getBasisStates(N, M):
    """
    Function to get an list of basis states, ordered by the integer
    representation of the state vector.
    """
    states = []
    x = itertools.product(range(N + 1), repeat=M)
    for i in x:
        if(np.sum(i) == N):
            # If the vector is a basis state, store the vector
            newVec = np.reshape(np.asarray(i), (N, 1))
            states.append(qt.Qobj(newVec))
        
    return np.asarray(states)

def actHam(state, N, M, J, U):
    t1, t2, c, d = [], [], qt.create(M+1), qt.destroy(M+1)
    for i in range(M-1):
        temp = c * state
    
def getHam(N, M, J, U, dim):
    """
    Act hamiltonian on a state, as a series of Creation, Annihilation and 
    Number operators. Copy states so that the original state is not effected.
    Multiply by appropriate parameters, J, U.
    Returns new states with appropriate prefectors.
    """
    ham = None
    dim = N+1
    c, d = qt.create(dim), qt.destroy(dim)
    
    for i in range(M-1):
        t1, t2 = [qt.qeye(dim) for x in range(M)], [qt.qeye(dim) for
                     x in range(M)]
    
        t1[i+1], t1[i] = c, d
        t2[i], t2[i+1] = c, d
        
        ham = ham - J*qt.tensor(t1) - J*qt.tensor(t2)
        
    for i in range(M):
        t3, t4 = [qt.qeye(dim) for x in range(M)], [qt.qeye(dim) for
                 x in range(M)]
        t3[i] = qt.num(dim)**2
        t4[i] = qt.num(dim)
        
        ham = ham + (U/2)*qt.tensor(t3) - (U/2)*qt.tensor(t4)
        
    return ham

def getInitialState(N, M, count=0):
    """
    Get initial state vector from user.
    """
    LENGTH = int((fact(N+M-1))/(fact(N)*fact(M-1)))
    PREAMBLE = """Enter initial state configuration (e.g. "1, 0, 0").
Note: This is the normalised vector representing the superposition of basis
states.
The basis states are ordered in ascending order of their integer representation
e.g. the state (0, 0, 2) -> "2" will be before the state (0, 1, 1) -> "11" will
be before the state (2, 0, 0) -> "200".
Should have length: C(N+M-1, N) = """ + str(LENGTH) + ': '
    initialStateVec = np.asarray([float(x) for x in input(PREAMBLE)
        .split(', ')])
    if(len(initialStateVec) != LENGTH):
        print(tColors.FAIL + '\nLength of initial state should be C(N+M-1,'
            +'N)\n' + tColors.ENDC)
        input('Press Enter to continue...')
        count += 1
        initialStateVec = getInitialState(N,M,count=count)
        
    if(np.sum(initialStateVec) != 1):
        print(tColors.FAIL + '\nInitial state should be normalised!' + 
              ' Sum != 1\n' + tColors.ENDC)
        input('Press Enter to continue...')
        count += 1
        initialStateVec = getInitialState(N,M,count=count)
        
    return initialStateVec

def init():
    N, M, J, U = [float(x) for x in input(
            'Enter params (comma separated: "N, M, J, U"): ').split(', ')]
    N, M = int(N), int(M)
    NUM = int((fact(N+M-1))/(fact(N)*fact(M-1)))
    ham = getHam(N, M, J, U, NUM)
    initialState = getInitialState(N, M)
    return {'N': N, 'M': M, 'J': J, 'U': U, 'initState': initialState,
            'ham': ham}