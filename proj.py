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
    """
    Class to keep track of state vector and prefactor.
    idx is a parameter that keeps basis states organised in the correct order
    according to the integer representation of the vector.
    """
    def __init__(self, init_vec, idx, _type):
        self.vector = np.asarray(init_vec)
        self.idx = idx
        self.type = _type
    prefactor = 1

def creationOp(state, idx, N):
    """
    Creation operator for acting on a StateObj
    """
    trans = deepcopy(state)
    a = trans.vector[idx]
    if(a == N):
        trans.prefactor = 0
    else:
        trans.vector[idx] += 1
    return trans

def annihilationOp(state, idx):
    """
    Annihilation operator for acting on a StateObj
    """
    trans = deepcopy(state)
    a = trans.vector[idx]
    if(a == 0):
        trans.prefactor = 0
    else:
        trans.vector[idx] -= 1
    return trans

def numberOp(state, idx):
    """
    Number operator for acting on a StateObj
    """
    if(state.vector[idx] == 0):
        return(-1)
    return 1

def getBasisStates(N, M):
    """
    Function to get an list of basis states, ordered by the integer
    representation of the state vector.
    """
    basis, sums, states, count = [], [], [], 0    
    x = itertools.product(range(N + 1), repeat=M)
    for i in x:
        if(np.sum(i) == N):
            # If the vector is a basis state, store the vector
            basis.append(np.asarray(i, dtype=int))
            val = int(''.join(np.asarray(i, dtype=str)))
            # Store the integer representation of the vector
            sums.append(val)
            
    for i in range(len(sums)):
        # Store StateObj's of basis vectors, odered from smallest integer
        # representation to largest
        idx = np.argmin(sums)
        state = StateObj(basis[idx], count, 'boson')
        states.append(state)
        count += 1
        # Set to infinity so argmin always picks next-smallest int.rep.
        sums[idx] = np.inf
        
    return np.asarray(states)

def actHam(state, N, J, U):
    """
    Act hamiltonian on a state, as a series of Creation, Annihilation and 
    Number operators. Copy states so that the original state is not effected.
    Multiply by appropriate parameters, J, U.
    Returns new states with appropriate prefectors.
    """
    first_term, second_term = [], []
    # First term
    for i in range(len(state.vector)-1):
        temp = creationOp(state, i+1, N)
        first_term.append(annihilationOp(temp, i))
        
        temp2 = creationOp(state, i, N)
        first_term.append(annihilationOp(temp2, i+1))
        
    # Second term
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
    """
    Get hamiltonian matrix by acting hamiltonian on each basis state.
    """
    basis = getBasisStates(N, M)
    ham_matrix = np.zeros((len(basis), len(basis)))
    for state in basis:
        # Act ham on each basis state
        acted = actHam(state, N, J, U)
        for x in acted:
            for b in basis:
                # Find the matrix location of the 'acted' state and enter into
                # Hamiltonian matrix.
                if(np.all(x.vector == b.vector)):
                    ham_matrix[state.idx][b.idx] = ham_matrix[state
                              .idx][b.idx] + x.prefactor
    # Return ham matrix and basis
    return ham_matrix, basis

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
        
    return StateObj(initialStateVec, None, 'state')

def timeEvolve(initialState, hamMatrix, t):
    """
    Evolve a state vector in time according to the hamiltonian matrix.
    """
    if(initialState.type == 'boson'):
        raise TypeError('State cannot be in boson format.')
    expMat = -1j * hamMatrix * t
    expMat = sp.linalg.expm(expMat)
    vNewState = np.dot(expMat, initialState.vector)
    tempSum = np.sum(vNewState)
    vNewState = vNewState / tempSum
    newState = StateObj(vNewState, None, 'state')
    return newState
    
def convertToBoson(M, state, basis):
    """
    Convert a state vector (normalised vector corresponding to superposition
    of basis states) to a boson vector (vector corresponding to boson
    occupation at each site).
    """
    if(state.type != 'state'):
        print(tColors.WARN + 'State to be converted is already bosonic'
              + tColors.ENDC)
        return state
    bosons = np.zeros(M, dtype=complex)
    for i in range(len(state.vector)):
        val = state.vector[i] * basis[i].vector
        bosons += val
    state.vector = bosons
    state.type = 'boson'
    return bosons    

def getReducedDensityMatrix(state, N, M, basis):
    """
    Get density matrix of a state.
    Input state in state vector representation e.g. [1/sqrt(2), 1/sqrt(2), 0]
    """
    b_basis = []
    x = itertools.product(range(N+1), repeat=M-1)
    for i in x:
        # Get every available state that the B subsystem can be in
        b_basis.append(np.asarray(i))
    # Initialise c_matrix as zeros
    BLEN = len(b_basis)
    c_matrix = np.zeros((N+1, BLEN), dtype=complex)
    for i in range(len(state.vector)):
        a_idx = basis[i].vector[0]
        _b = basis[i].vector[1:]
        b_idx = np.where(b_basis==_b)[0]
        c_matrix[a_idx, b_idx] = state.vector[i]
        
    rdm = np.dot(c_matrix, c_matrix.conj().T)
    
    return c_matrix, rdm

def init():
    N, M, J, U = [float(x) for x in input(
            'Enter params (comma separated: "N, M, J, U"): ').split(', ')]
    N, M = int(N), int(M)
    initialState = getInitialState(N, M)
    hamMatrix, basis = getHamMatrix(N, M, J, U)
    return {'N': N, 'M': M, 'J': J, 'U': U, 'initState': initialState,
            'ham': hamMatrix, 'basis': basis}

if(__name__ == '__main__'):
    print('In module.')
    initDict = init()
    tState = timeEvolve(initDict['initState'], initDict['ham'], 10e-3)
    cmat, rdm = getReducedDensityMatrix(tState, initDict['N'], initDict['M'],
                                  initDict['basis'])
    
    