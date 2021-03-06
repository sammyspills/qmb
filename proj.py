# -*- coding: utf-8 -*-
"""
author: Sam Spillard
python script to complete final project of Quantum Many-Body Physics module
Simulate experiment by Kaufman et al. 2016

Python version: 3.5.4
"""

from scipy.special import factorial as fact
import scipy as sp
import numpy as np
import itertools
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

class StateObj:
    """
    Class to keep track of state vector, prefactor and type
    idx is a parameter that keeps basis states in the correct order according
    to the integer representation of the vector.
    Includes creation, annihilation, number, time evolution, rdm and entropy
    operators, deepcopy'd to prevent modifying basis states.
    """
    # Initialise attributes
    def __init__(self, init_vec, idx, _type):
        if(type(init_vec) != np.ndarray):
            raise TypeError('init_vec should be a numpy.ndarray')
        self.vector = init_vec
        self.idx = idx
        self.type = _type
        self.prefactor = 1
    # Creation operator
    def create(self, index, N):
        trans = deepcopy(self)
        if(np.sum(trans.vector) > N):
            trans.prefactor = 0
        else:
            trans.vector[index] += 1
            trans.prefactor *= np.sqrt(trans.vector[index])
        return trans
    # Annihilation operator
    def destroy(self, index):
        trans = deepcopy(self)
        if(trans.vector[index] == 0):
            trans.prefactor = 0
        else:
            trans.prefactor *= np.sqrt(trans.vector[index])
            trans.vector[index] -= 1
        return trans
    # Number operator
    def num(self, index):
        return self.vector[index]
    # Time Evolution
    def tevolve(self, hamMatrix, t):
        if(self.type == 'boson'):
            raise TypeError('State should not be in boson format')
        trans = deepcopy(self)
        expMat = sp.linalg.expm(-1j * hamMatrix * t)
        trans.vector = np.dot(expMat, self.vector)
        return trans
    # Reduced Density Matrix
    def rdm(self, N, M, basis, ASIZE=1):
        # Every available a state
        a_basis_ints =[int(''.join([str(i) for i in j])) for j in np.asarray(
                [np.asarray(i) for i in itertools.product(range(N+1),
                            repeat=ASIZE)])]
        # Every available b state
        b_basis_ints =[int(''.join([str(i) for i in j])) for j in np.asarray(
                [np.asarray(i) for i in itertools.product(range(N+1),
                            repeat=(M-ASIZE))])]
        # Initialise c_matrix as zeros
        ALEN, BLEN = len(a_basis_ints), len(b_basis_ints)
        c_matrix = np.zeros((ALEN, BLEN), dtype=complex)
        for i in range(len(self.vector)):
            a_vec = basis[i].vector[:ASIZE]
            b_vec = basis[i].vector[ASIZE:]
            
            # Get matrix indices for entry
            a_idx = findIndex(a_basis_ints, a_vec)
            b_idx = findIndex(b_basis_ints, b_vec)
            
            c_matrix[a_idx, b_idx] = self.vector[i]*self.prefactor

        RDM = np.dot(c_matrix, c_matrix.conj().T)

        return RDM
    # Entropy
    def entropy(self, N, M, basis, ASIZE):
        rdm = self.rdm(N, M, basis, ASIZE)
        vals, vecs = sp.linalg.eigh(np.dot(rdm, rdm))
        return -np.log(np.sum(vals))

def findIndex(basis_ints, state_vector):
    """
    Implementation of binary search specifically for finding index of a vector
    in a Fock space
    
    Parameters:
        :basis_ints: Array of integer representations of basis vectors
        :state_vector: State vector
        
    Returns:
        :index: Index where basis state and input state match
    """
    state_int = int(''.join([str(i) for i in state_vector]))
    upper, lower, found, count = int(len(basis_ints)), 0, False, 0
    while not found:
        if(count>len(basis_ints)):
            raise RecursionError('State not found in basis')
        count += 1
        mid = int((upper + lower)/2)
        if(state_int == basis_ints[mid]):
            index = mid
            found = True
        elif(state_int > basis_ints[mid]):
            lower = mid
        else:
            upper = mid
            
    return index

def getBasisStates(N, M):
    """
    Function to get a list of basis states, ordered by the integer
    representation of the state vector.
    """
    x = itertools.product(range(N + 1), repeat=M)
    basis = [np.asarray(i, dtype=int) for i in x if np.sum(i)==N]
    states = np.asarray([StateObj(basis[i], i, 'boson') for i in
                        range(len(basis))])

    return states

def actHam(state, N, J, U):
    """
    Act hamiltonian on a state, as a series of Creation, Annihilation and
    Number operators.
    Multiply by appropriate parameters, J, U.
    Returns an array of new states with appropriate prefactors.
    """
    t1, t2 = [], []
    # First term
    for i in range(len(state.vector)-1):
        t1.append(state.create(i+1, N).destroy(i))
        t1.append(state.create(i, N).destroy(i+1))
    # Second term
    for i in range(len(state.vector)):
        PREFACTOR = state.num(i) * (state.num(i) - 1)
        temp3 = deepcopy(state)
        temp3.prefactor *= (PREFACTOR*U/2)
        t2.append(temp3)

    for state in t1:
        state.prefactor *= (-1 * J)

    return np.r_[t1, t2]

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
                    ham_matrix[state.idx][b.idx] += x.prefactor
    # Return ham matrix and basis
    return ham_matrix, basis

def getInitialState(N, M):
    """
    Get initial state vector from user.
    """
    LENGTH = int((fact(N+M-1))/(fact(N)*fact(M-1)))
    temp_basis = np.asarray([x.vector for x in getBasisStates(N, M)])
    for i in range(len(temp_basis)):
        if(np.prod(temp_basis[i])==1):
            idx = i
            break
    initialStateVec = np.zeros(LENGTH)
    initialStateVec[idx] = 1
    state = StateObj(initialStateVec, None, 'state')

    return state

def getPlot(initDict, tArr):
    """
    Given dictionary of initial parameters and an array of times to evaluate
    at, calculate the Renyi entropy and plot.
    """
    entropy_arr = []
    for t in tArr:
        print('Time: ' + str(t), flush=True)
        tEntropy = initDict['initState'].tevolve(initDict['ham'], t).entropy(
            initDict['N'], initDict['M'], initDict['basis'], initDict['ASIZE'])
        entropy_arr.append(tEntropy)
    plt.plot(tArr, entropy_arr)
    plt.xlabel('Time (s)')
    plt.ylabel('Renyi entropy: $S_{A}$')
    plt.title('Renyi entropy for Bose-Hubbard model, equal bipartition')
    plt.ylim((-0.1, 4))
    #plt.savefig('plot.png', format='png', dpi=200)
    plt.show()
    return

def init():
    """
    Initialisation. Gets parameters from user and construct Hamiltonian matrix.
    Returns dict of params.
    """
    N, M, J, U = [float(x) for x in input(
            'Enter params (comma separated: "N, M, J, U"): ').split(', ')]
    N, M = int(N), int(M)
    initialState = getInitialState(N, M)
    hamMatrix, basis = getHamMatrix(N, M, J, U)
    if(bool(int(input('Print 4 lowest energies? (yes:1, no:0): ')))):
        vals, vecs = sp.linalg.eigh(hamMatrix)
        print(vals[:4])
    ASIZE = int(input('Enter size of A subsystem: '))
    return {'N': N, 'M': M, 'J': J, 'U': U, 'initState': initialState,
            'ham': hamMatrix, 'basis': basis, 'ASIZE': ASIZE}

if(__name__ == '__main__'):
    print('In module.')
    initDict = init()
    getPlot(initDict, np.linspace(0, 14, 401))
