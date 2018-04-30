import numpy as np
import math as m
import matplotlib.pyplot as plt
from cmath import exp
import itertools as it
from scipy.special import factorial as fact
import scipy as sp
from copy import copy

t = np.linspace(0,1.6,500)
from scipy.linalg import expm


def Fock_space(M,N):
    
    x= np.linspace(0,N,N+1, dtype=int)      
    vectors = np.asarray([np.asarray(i) for i in it.product(x, repeat= M)])
    fock_basis = []    
    for i in vectors:
        if np.sum(i) == N:
            fock_basis.append(i)
        else:
            continue
    fock_basis = np.array(fock_basis) 
        
    return fock_basis

def creation(vector, site):
        vector[site] = vector[site] + 1    
        return vector

def annahilation (vector, site): 
    if vector[site] != 0:
        vector[site] = vector[site] - 1
    else:
        vector = np.zeros(len(vector))     
    return vector
         
def number_operator(vector):
    N = []
    for i in range(len(vector)):
        n = (vector[i]*(vector[i]-1))
        N.append(n)    
    n = np.sum(N)
    return n

def Hamiltonian(vector,J, fock_basis):
    
    produced_vector = np.zeros(len(fock_basis))
    elements =np.zeros(len(fock_basis))

    for site in range(len(vector)-1):
        
        x, y = copy(vector), copy(vector)
        
        a_1 = annahilation(y,site)                 
        if np.sum(a_1) != 0:
            c = np.sqrt(a_1[site]+1)
            b_1 = creation(a_1, site + 1)
            b_1 = np.array(b_1)
            d = np.sqrt(b_1[site+1])
            norm_1 = c*d
            for i in range(len(fock_basis)):
                if np.all(fock_basis[i]==b_1):
                    elements[i]=-J*norm_1
                    
        a_2 = annahilation(x, site + 1)
        if np.sum(a_2) != 0 :
        
            e = np.sqrt(a_2[site + 1]+1)            
            b_2 = creation(a_2, site)
            b_2 = np.array(a_2)
            f = np.sqrt(b_2[site]) 
            norm_2 = e*f
            
            for i in range(len(fock_basis)):
                if np.all(fock_basis[i]==b_2):
                    elements[i] = -J*norm_2
    
    return produced_vector, elements

        
def Hamiltonian_matrix(M,N,U,J):

    '-------- calculating the multiplicity ---'
    numerator = fact(N + (M - 1))
    n_fact = fact(N)
    m_fact = fact(M - 1)
    demoninator = n_fact*m_fact
    omega = numerator/demoninator         
    fock_basis = Fock_space(M,N)    
    H = []
    b=[]
    indx = 0

    '-----aquring off diagonal hamiltonian elements----' 
    
    for i in range(len(fock_basis)):
        vector = fock_basis[i]
        
        produced_vector, elements = Hamiltonian(vector,J,fock_basis)
        if np.prod(vector)==1:
            indx = indx + i             
        
        '----- appending off digonal elements ---'
        
        H.append(elements)
        
        '-----appending diagonal elements----'        
        a = np.zeros(len(fock_basis))       
        n = number_operator(vector)
        a[i] = n*U*1/2
        b.append(a)
   
    b = np.array(b)
    H = H + b
    H = np.array(H)
 
    return H, indx, fock_basis, omega
    
def initialise_state(H,indx, fock_basis,t):
    initial_state = np.zeros((len(fock_basis)))
    initial_state[indx] = 1
   
    #expHam = sp.linalg.expm(-1j*H*t)
    #wave_function = np.dot(expHam,initial_state)
    return initial_state 

def density_matrix(wave_function):
    bra = np.conjugate(wave_function)
    ket = wave_function
    density_matrix = np.outer(ket,bra)
    #print 'density matrix'
    #print density_matrix
    return density_matrix

def reduced_density_matrix(density_matrix,H,initial_state,t, omega):

    hamilt =H
    configurations = omega
    
    print np.shape(t)
    unit_matrix = [expm(-1j*hamilt*i) for i in t]
    
    print 'unit_matrix' 
    print np.shape(unit_matrix)
    
    
    psi_t = [np.dot(i,initial_state) for i in unit_matrix] #calculates the dot product of the matrix
                                               #with the original state
    dens_matrices = [np.outer(i,np.conj(i)) for i in psi_t] #produces the density matrix P_AB
    
    P_AB = [1.0*i/np.trace(i) for i in dens_matrices]
    P_AB_tensor = [np.array(i.reshape([configurations,1,configurations,1]),dtype=complex) for i in P_AB]
    P_A_separate = []
    for i in P_AB_tensor:
        P = []
        for j in i:
            P.append(np.outer(j, np.conj(j)))
        P_A_separate.append(P)
    P_A = [sum(j for j in i) for i in P_A_separate ]
    P_A2 = [i*i for i in P_A]
    
    return P_A2

def renyi():
    """
    Renyi entropy is calculated via the equation S = -ln(tr(P_A^2))
    """
    P_A2 = reduced_density_matrix(density_matrix,H,initial_state,t, omega)
    trace = [np.trace(np.real(i)) for i in P_A2]
    S = -np.log(trace)
    
    return S
    

    

H, indx, fock_basis, omega = Hamiltonian_matrix(5,5,1,1)
initial_state = initialise_state(H,indx, fock_basis,t)
P_A2 = reduced_density_matrix(density_matrix,H,initial_state, t, omega)
S = renyi()

plt.plot(t,S)




#def Reduced_density_matrix(n,wave_function, fock_basis):
#    
#    x= np.linspace(0,n,n+1, dtype=int)
#    m_1 = 6    
#    m_2 = 6
#    vector_a = np.asarray([np.asarray(i) for i in it.product(x, repeat= m_1)])          
#    vector_b = np.asarray([np.asarray(i) for i in it.product(x, repeat= m_2)])
#    
#    
#    state_matrix = np.zeros((len(vector_a),len(vector_b)), dtype=np.complex128)
#    #print wave_function 
#    for i in range(len(wave_function)):
#        
#        sub_a = fock_basis[i][:m_1]
#        sub_b = fock_basis[i][m_1:]
#        
#        for j in range(len(vector_a)):
#            if np.all(sub_a == vector_a[j]):
#                a_indx = j
#                break
#        for k in range(len(vector_b)):
#            if np.all(sub_b == vector_b[k]):
#                b_indx = k
#                break
#        
#        #print wave_function[i]
#            state_view = state_matrix.view(dtype=float)
#            state_view[a_indx,b_indx] = np.real(wave_function[i])
#            state_view[a_indx,b_indx+1] = np.imag(wave_function[i])
#            state_matrix = state_view.view(dtype = complex)
#        
#    #s = sp.linalg.svd(state_matrix, compute_uv = False)
#    sm_conjugate = np.conjugate(state_matrix).T
#    reduced_density_matrix =  np.dot(state_matrix,sm_conjugate)
#    
#    print np.shape(reduced_density_matrix)
#    
#    
##    if np.all(reduced_density_matrix==reduced_density_matrix.T):
##        print 'yes'
##        print reduced_density_matrix
##        
#    #print reduced_density_matrix
#    reduced_density_matrix_sq = np.dot(reduced_density_matrix,reduced_density_matrix)  
#    #vel = np.trace(reduced_density_matrix_sq)    
#    vel, vec =np.linalg.eigh(reduced_density_matrix_sq) 
#        
#        
#    entropy = -np.log(np.sum(vel))
#    return entropy


