#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 10:13:59 2021

@author: sam
"""

import samHelper as sh
import numpy as np
from math import pi

import csv

import qutip as qt
import qutip.qip.operations as op

'''
For running a variety of different quantum networks using the QuTiP package.
Allows for extraction of useful quantities such as the Fisher matrix and
effective dimension
'''

#%%
###############################################################################
####                 Useful functions                                     ####
###############################################################################

def featureMap_hard(feature, N_qubit, N_layer):
    '''
    This is a classically hard-to-simulate feature map,
    as used in the 'Power of Quantum Neural Networks' paper
    '''
    psi = qt.ket('0'*N_qubit)
    H = op.hadamard_transform(N_qubit)
    
    rz = [ op.rz( float(feature[ind]), N=N_qubit, target=ind ) for ind in range(N_qubit)   ]

    RZ = qt.qeye([2]*N_qubit)
    for u in rz:
        RZ = u*RZ

    rzz = []
    for ind1 in range(1,N_qubit):
        for ind2 in range(ind1):
            rzz.append( op.cnot(N=N_qubit, target=ind1, control=ind2) * 
              op.rz( (pi-float(feature[ind1]))*(pi-float(feature[ind2])), N=N_qubit, target=ind1  ) *
              op.cnot(N=N_qubit, target=ind1, control=ind2) )

    RZZ = qt.qeye([2]*N_qubit)
    for u in rzz:
        RZZ = u*RZZ
    for l in range(N_layer):
        psi = RZZ*RZ*H*psi
    return psi

def featureMap_easy(feature, N_qubit):
    '''
    This is a classically easy-to-simulate feature map,
    which we don't expect to give much advantage over a purely classical model
    '''
    
    psi_init = qt.ket('0'*N_qubit)
    H = op.hadamard_transform(N_qubit)
    psi_1 = H*psi_init
    
    rz = [ op.rz( float(feature[ind]), N=N_qubit, target=ind ) for ind in range(N_qubit)   ]

    RZ = qt.qeye([2]*N_qubit)
    for u in rz:
        RZ = u*RZ
        
    return RZ*psi_1

def V_x(x):
    # Generates the 2x2 matrices which constitute the diagonal blocks for our AR perceptron
    return [[f_x(x), f_x(-x)], [f_x(-x), -f_x(x)]] # Note that f_x already includes the sqrt

def f_x(x):
    # Note that I have defined this to include the sqrt
    return np.sqrt(0.5*(1 + x/np.sqrt(1+x**2)))

def AR_perceptron(weights,N_qubits,controls,target,bias=True):
    '''
    Construct the unitary gate for an AR perceptron.
    weights should be a list. 
    The legnth of weights is equal to the number of inputs
    By default, we include a bias.
    '''
    if bias:
        N = len(weights)-1
    else:
        N = len(weights)
    B = sh.allBin(N) # We will loop over all possible input basis states

    Theta = []
    for b in B:
        activation = sum([weights[x]*b[x] for x in range(len(b))])
        if bias: # Add a bias if we have one
            activation += weights[-1]
        Theta.append(activation)
    blocks = np.array([V_x(t) for t in Theta])
    
    U = qt.Qobj( block_diagonal(blocks, 0), dims=[(N+1)*[2], (N+1)*[2]]  )
    
    # Note that, because I use my lazy 'expand' function, controls and targets must be in order
    return sam_expand(U, N_qubits, controls=controls, target=target)

def sam_expand(U,N,controls,target):
    '''
    This is my own version of qutips gate_expand_MtoN() functions, generalized for any M.
    However, M larger than 3 makes things rather complicated, so I'm going to take some shortcuts.
    I will straight-up assume that target>control for all controls.
    I will also assume that control[x]>control[y] whenever x>y.
    So order your shit before you send it into this function
    '''
    
    M = len(controls)
    p = list(range(N))
    
    for m in range(M):
        p[m], p[controls[m]] = p[controls[m]], p[m]
    p[M], p[target] = p[target],p[M]
    
    return qt.tensor( [U] + [qt.identity(2)]*(N-M-1) ).permute(p)
    

def block_diagonal(x, k):
    ''' x should be a tensor-3 (#num matrices, n,n)
        k : int
        Diagonal in question. it is 0 in case of main diagonal. 
        Use k>0 for diagonals above the main diagonal, and k<0 for diagonals below the main diagonal.
    '''

    shape = x.shape
    n = shape[-1]

    absk = abs(k)

    indx = np.repeat(np.arange(n),n)
    indy = np.tile(np.arange(n),n)

    indx = np.concatenate([indx + a * n for a in range(shape[0])])
    indy = np.concatenate([indy + a * n for a in range(shape[0])])

    if k<0: 
        indx += n*absk
    else:
        indy += n*absk

    block = np.zeros(((shape[0]+absk)*n,(shape[0]+absk)*n))
    block[(indx,indy)] = x.flatten()

    return block


def CAN_gate(weights, N_qubits, control, target):
    # gx = -1j*weights[0]*pi*qt.tensor(qt.sigmax(),qt.sigmax())
    # gy = -1j*weights[1]*pi*qt.tensor(qt.sigmay(),qt.sigmay())
    # gz = -1j*weights[2]*pi*qt.tensor(qt.sigmaz(),qt.sigmaz())

    # small_gate = (gx + gy + gz).expm()
    
    x = weights[0]
    y = weights[1]
    z = weights[2]
    
    # Putting in the matrix by hand is a bit quicker
    # This is important, since this function gets called very often when
    # training the DDQNN. Small gains are worth it.
    small_gate = qt.Qobj( [[ np.exp(-1j*z)*(np.cos(x)*np.cos(y) + np.sin(x)*np.sin(y)), 0, 0, -1j*np.exp(-1j*z)*(np.sin(x)*np.cos(y) - np.cos(x)*np.sin(y))],
                             [0, np.exp(1j*z) *(np.cos(x)*np.cos(y) - np.sin(x)*np.sin(y)), -1j*np.exp(1j*z)*(np.sin(x)*np.cos(y) + np.cos(x)*np.sin(y)), 0],
                             [0, -1j*np.exp(1j*z)*(np.sin(x)*np.cos(y) + np.cos(x)*np.sin(y)), np.exp(1j*z) *(np.cos(x)*np.cos(y) - np.sin(x)*np.sin(y)), 0],
                             [1j*np.exp(-1j*z)*(np.cos(x)*np.sin(y) - np.sin(x)*np.cos(y)), 0, 0, np.exp(-1j*z) *(np.cos(x)*np.cos(y) + np.sin(x)*np.sin(y))]
                            ], dims=[[2,2],[2,2]] )
    
    return op.gate_expand_2toN(small_gate, N_qubits, target=target, control=control)
    
def singleQubitGate(weights, N_qubits, target):
    u = qt.Qobj( [ [np.cos(weights[0]/2), -np.exp(1j*weights[2]) * np.sin(weights[0]/2)],
                   [np.exp(1j*weights[1]) * np.sin(weights[0]/2), 
                    np.exp(1j*(weights[1] + weights[2])) * np.cos(weights[0]/2)  ] ]
               )
    return op.gate_expand_1toN(u, N_qubits, target)



def randomIsing(N):
    '''
    Generate Hamiltonian of a random Ising chain on N qubits
    Includes only nearest-neighbour interactions
    Couplings are drawn randomly from a Gaussian distribution 
    '''
    J = np.random.normal(size=N)
    dims = [ [2]*N, [2]*N ]
    Ham = qt.Qobj(np.zeros([2**N,2**N]))
    Ham.dims=dims
    for ind in range(N):
        Ham += J[ind]*sh.ZZ(ind,(ind+1)%N, N)
    return Ham


def trickyFish(network, samples, grad_a):
    fish_list = []
    for s in samples:
        psi = network.load_features(s)
        
        fish_a = []
        for grad in grad_a:
            grad_vec = [ qt.expect(g,psi) for g in grad  ]
            fish_a.append( np.outer( grad_vec, grad_vec ) )
            
        fish_list.append( sum(fish_a) )
    return sum(fish_list)/len(samples)

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def get_data(datatype,inputs=4,outputs=3):
    if datatype == 'random':
        data = np.array([ np.r_[ np.random.randint(outputs), np.random.random(inputs) ] for _ in range(100) ])
        
    elif datatype == 'iris':
        filename = 'bezdekIris.data'
        raw_data = open(filename, 'rt')
        reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
        x = list(reader)[:-1]
        
        for y in x:
            if  y[-1] == 'Iris-setosa':
                y[-1] = '0'
            if y[-1] == 'Iris-versicolor':
                y[-1] = '1'
            if y[-1] == 'Iris-virginica':
                y[-1] = '2'
                
        data = np.array(x).astype('float')
        a = [-1] + list(range(len(data[0])-1))
        data = data[:,a]
        
        # Map data onto interval [0,2*pi]
        for c in range(1,5):
            upper = max(data[:,c])
            lower = min(data[:,c])
            
            data[:,c] = 2*pi*(data[:,c]-lower)/(upper-lower)
        
    elif datatype == 'wine':
        datafilename = 'wine.data'
        raw_data = open(datafilename, 'rt')
        reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
        x = list(reader)
        wine_data = np.array(x).astype('float')
        
        data = wine_data
                
    return data

#%%
###############################################################################
####                 The QuantumNetwork Class                              ####
###############################################################################

class QuantumNetwork:
    '''
    This class will be used to define the networks that I train and compare.
    Inputs: S -- the network structure. This is an array for neural-network type
                    circuits, and a number (of qubits) for more straightforward
                    circuits.
            network_type -- specifies which kind of network I am looking at.
            fmap -- the feature map used. This is how classicla input data
                    is encoded in the network.
    
    network_type will determine the number of parameters (for a 
    given network structure S), as well as the way the output of
    the network is calculated. Most functions should be defined
    to be network independent.
    
    network_type options will be:
    AR -- adiabatic ramp perceptron network (see: Torrontegui and Garcia-Ripoll). This has a 
            neural network-like layered structure, in which previous layers are discarded.
    RUS -- repeat-until-sueccess network (see: Cao, Guerreschi and Aspuru-Guzik). Much like AR.
    DQNN -- dissipative quantum neural network (see Beer et al.). Another neural network-like 
            structure in which previous layers are discarded.
    QAOA -- quantum approximate optimization algorithm (see Fahri, Goldstone and Gutmann).
            A straightforward circuit (input qubits are also output qubits, none are discarded).
    HEA -- Hardware Efficient Ansatz. The variational circuit used by Abbas et al. in the original Power of Quantum Neural Networks
            paper. Has straightforward circuit much like QAOA.
            
    These circuit types fall into two braod categories: dissipative and non-dissipative. 
    
    For the dissipative networks, each layer of the network involves different qubits. 
    The structure S = [l0,l1,..,lN] is an array, where lj is the number of qubits in the
    jth layer, and N is the total number of layers (so that the total number of qubits is)
    the sum over S.
    
    For non-dissipative networks, every layers involves the same qubits, and thus the number
    of qubits in each layers is fixed. Thus, the structure need only be a two- or tHEAe-element array,
    S = [NQ,NL], where NQ is the number of qubits and NL is the number of layers. If the number of outputs
    is different from the number of inputs, a third element NO can be appended to this list.
            
    The other main distinguishing feature will be fmap. The options are:
    easy -- see Abbas et al.
    hard -- see Abbas et al.
    none -- no feature map, instead we directly specify a quantum state
    If using the 'hard' feature map, we must specify a number of feature map layers, flayers
    '''
    
    def __init__(self,S,network_type='AR',fmap='easy',flayers=2):
        self.S = S
        self.network_type = network_type
        self.fmap = fmap
        if self.fmap == 'hard':
            self.flayers = flayers
        
        if self.network_type in ['AR', 'AR with single', 'RUS', 'DQNN']: # dissipative circuits
            self.N_qubit = np.sum(S)
            
            self.N_layer = len(self.S)
            
            # Generate layer structure, so that we can always keep track of which weights correspond to which edges
            # Two-qubit weights always come first, single qubit weights after.
            self.L = []
            ind = 0
            for s in S:
                begin = ind
                ind = begin + s
                l = [x for x in range(begin,ind)]
                self.L.append(l)

            self.A = np.zeros([self.N_qubit,self.N_qubit])
            for k in range(self.N_layer-1):
                for j in self.L[k+1][:self.S[k+1]]:
                    for i in self.L[k][:self.S[k]]:
                        self.A[i][j] = 1

            rows, cols = np.nonzero(self.A)
            
            if self.network_type in ['DQNN']:
                self.edges = [ [rows[i],cols[i]] for i in range(len(rows)) ] 
                self.N_params = 3*(self.N_qubit + len(self.edges))
            else:
                self.edges = [ [rows[i],cols[i]] for i in range(len(rows)) ] + [i for i in range(self.S[0],self.N_qubit)]
                self.N_params = np.sum( [self.S[ind]*self.S[ind+1]
                                 for ind in range(len(self.S)-1) ] )\
                                + sum(self.S[1:])
                if self.network_type == 'AR with single':
                    self.N_params += 3*self.N_qubit # extra params for single qubit rotations
            

        elif self.network_type in ['QAOA', 'HEA']: # non-dissipative circtuis
            self.N_qubit = S[0]
            if self.network_type == 'QAOA':
                self.N_params = 2*S[1]
                self.A_gate = randomIsing(self.N_qubit) # generate random Ising model -- but this can be overwritten with other gate types
                self.B_gate = qt.tensor( [qt.sigmax() for _ in range(self.N_qubit)] )
            elif self.network_type == 'HEA':
                self.N_params = S[0]*(S[1]+1)
            self.N_layer = S[1]
        
        self.weights = 2*(np.random.rand(self.N_params) - 0.5) # evenly distributed random initial parameters
        
    
    def runNetwork(self,initial=None, weights=None,feature=None):
        '''
        Pretty self-explanatory: we run the network.
        Inputs: self
                weights -- the parameters of the network
                feature -- the data fed into the network
        Outputs: A quantum state
        '''
        
        if feature is None:
            feature = [0]*self.S[0]
        
        # Including feature maps in runNetwork may prove inefficient if I want to run the network multiple
        # times on a single feature -- which I may do for getting gradients...
        if initial==None:
            if self.fmap == 'easy':
                initial = featureMap_easy(feature,self.S[0]) # Use S[0] rather than N_qubit to work for both dissipative and non-dissipative networks
            elif self.fmap == 'hard':
                initial = featureMap_hard(feature,self.S[0],self.flayers)
            elif self.fmap == 'none':
                # For now, we are assuming a computational basis state
                initial = qt.ket(str(feature)) 
            
        if self.network_type in ['AR', 'AR with single', 'DQNN']:
            initial = qt.tensor(initial, qt.ket('0'*self.S[1]) )
            if self.network_type == 'AR':
                return self.run_ARnetwork(weights=weights,initial=initial)
            elif self.network_type in ['DQNN', 'AR with single']:
                return self.run_dissNetwork(weights=weights,initial=initial)
        elif self.network_type == 'HEA':
            return self.run_HEAnetwork(initial,weights=weights)
        elif self.network_type == 'QAOA':
            return self.run_QAOAnetwork(initial,weights=weights)
        
    
    def load_features(self,feature):
        if self.fmap == 'easy':
            return featureMap_easy(feature,self.S[0]) # Use S[0] rather than N_qubit to work for both dissipative and non-dissipative networks
        elif self.fmap == 'hard':
            return featureMap_hard(feature,self.S[0],self.flayers)
        elif self.fmap == 'none':
            # For now, we are assuming a computational basis state
            return qt.ket(str(feature)) 
            
            
    def fisherMatrix(self,data,eps,weights=None):
        '''
        Gerenates the Fisher information matrix at a particular point in parameter space
        by averaging over input-output pairs listed in 'pairs'
        '''
        fish_list = []
       
        if weights is None:
            weights = self.weights

        for d in data:
            for ind1 in range(self.S[-1]): # loop over output qubits
                # Prob for output qubit to read '1'
                y_op = op.gate_expand_1toN(qt.num(2), self.S[-1], ind1)
                grad = []
                for ind2 in range(len(self.weights)):
                    diff = np.zeros(len(self.weights))
                    diff[ind2] = eps
#                     finite_diff = ( np.log(qt.expect( y_op, self.runNetwork(weights=weights+diff, feature=d) ))
#                                    - np.log(qt.expect( y_op, self.runNetwork(weights=weights-diff, feature=d)) )
#                                       )/(2*eps)
                    finite_diff = ( qt.expect( y_op, self.runNetwork(weights=weights+diff, feature=d) )
                                   - qt.expect( y_op, self.runNetwork(weights=weights-diff, feature=d))
                                      )/(2*eps)
                    grad.append(finite_diff)
#                     set_trace()
                fish_list.append( np.outer(grad,grad) )
        
        return np.sum(fish_list,axis=0)/len(data)
    
    def fisherSpectrum(self,data,instances,eps):
        '''
        Returns the mean spectrum of the Fisher information matrix, averaged over a number 
        of points in parameter space (given by 'instances')
        '''
        
        spec_array = []
        for inst in range(instances):
            weights = 2*(np.random.rand(self.N_params) - 0.5)
#             set_trace()
            Fmat = self.fisherMatrix(data,eps,weights=weights)
            spec = np.linalg.eig(Fmat)[0]
            spec_array.append(spec)
        s = spec_array[0]
        for ind in range(1,instances):
            s = s + spec_array[ind]
        return s/instances
    
    def trickyFish(self, samples, grad_a):
        '''
        A slightly better way of computing the Fisher matrix, if you have your gradient
        operators beforehand
        '''
        fish_list = []
        for s in samples:
            psi = self.load_features(s)
            
            fish_a = []
            for grad in grad_a:
                grad_vec = [ qt.expect(g,psi) for g in grad  ]
                fish_a.append( np.outer( grad_vec, grad_vec ) )

            fish_list.append( sum(fish_a) )
        return sum(fish_list)/len(samples)
    
    
    def effectiveDimension(self,F_list,n):
        '''
        Calculates the effective dimension of the model.
        F_list is a list of Fisher matrices evaluated at various points in parameter space
        n is the number of training data
        '''
        
        C = n/(2*pi)
        F_hat = ( (self.N_params * len(F_list))/( np.sum( [ np.trace(F) for F in F_list ])  ) ) * F_list
        inside = [ np.sqrt( np.linalg.det( np.identity(self.N_params) + C*F )  ) for F in F_hat ]
    
        return 2*np.log( sum(inside)/len(inside) )/np.log(C)
    
    def networkCore(self,weights=None,out=0):
        '''
        This computes just the parameterized part of the network -- i.e. not the feautre map
        '''
        if self.network_type == 'AR':
            return self.AR_core(weights=weights,out=out)
        if self.network_type in ['DQNN', 'AR with single']:
            return self.diss_core(weights=weights,out=out)
        elif self.network_type == 'HEA':
            return self.HEA_core(weights=weights,out=out)
        elif self.network_type == 'QAOA':
            return self.QAOA_core(weights=weights,out=out)
    
    def gradCore(self,weight_set,eps=1e-3,batch=None):
        '''
        Gradient of the 'core' part of the network -- independently of the input state/feature map.
        Makes computing things like the Fisher matrix a little easier.
        
        Will evaluate this gradient operator at various different points in parameter space,
        with each point being an element of the weight_set
        '''
        
        grad_O = []

        for weights in weight_set:
            grad_a = []
            for ind2 in range(self.S[-1]):
                grad_vec = []
                
                if batch is None:
                    for ind1 in range(self.N_params):
                        
                        diff = np.zeros(self.N_params)
                        diff[ind1] = eps
                        forward = self.networkCore(weights=weights+diff, out=ind2)
                        backward = self.networkCore(weights=weights-diff, out=ind2)
                        finite_diff = (forward - backward)/(2*eps)
    
                        grad_vec.append(finite_diff)
                else: # We only comput a random subset of the gradients each time
                    target_thetas = np.random.permutation(np.arange(0,len(weights)))[:batch]
                    zero =  qt.Qobj( np.zeros( [2**self.S[0], 2**self.S[0]] ), dims = [[2]*self.S[0],[2]*self.S[0]] )
                    for ind in range(len(weights)):
                        if ind in target_thetas:
                            diff = np.zeros(self.N_params)
                            diff[ind] = eps
                            forward = self.networkCore(weights=weights+diff, out=ind2)
                            backward = self.networkCore(weights=weights-diff, out=ind2)
                            finite_diff = (forward - backward)/(2*eps)
                        else:
                            finite_diff = zero
            
                        grad_vec.append(finite_diff)

                grad_a.append( grad_vec )

            grad_O.append(grad_a)
            
        return grad_O
    
    
######################################################################################
#######                   Network-specific stuff goes here                     #######
######################################################################################
    
    def run_dissNetwork(self,weights=None,initial=None):
        if weights is None:
            weights = self.weights
        # If we also have single-qubit rotations, N_E is the index at which we start counting them
        if self.network_type in ['DQNN']:
            N_E = 3*len(self.edges)
        elif self.network_type == 'AR with single':
            N_E = len(self.edges)
        for step in range(self.N_layer-1):
            N_sub = self.S[step] + self.S[step+1]  # number of qubits in the subsystem we consider for now
            control_layer = self.L[step]
            target_layer = self.L[step+1]

            U_list = []
            sub_target = len(control_layer)

            if step==0:
                if initial==None:
                    state = qt.ket('0'*N_sub) * qt.bra('0'*N_sub)
                else:
                    state = initial
                if state.dims[1][0]==1:
                    state = state * state.dag()
                if len( (state).dims[0] ) < N_sub:
                    padding = qt.ket('0'*(N_sub - len(state.dims[0]) ))
                    state = qt.tensor(state, padding * padding.dag() )
                if self.network_type in ['DQNN', 'AR with single']:
                    # Initial layer of single-qubit gates
                    for target in control_layer:
                        single_list = [ N_E+3*target, N_E+3*target+1, N_E+3*target+2 ]

                        U_list.append( singleQubitGate( weights[single_list], N_sub, sub_target ) )

            else:
                state = qt.tensor(state, qt.ket('0'*len(target_layer))*qt.bra('0'*len(target_layer))   )
            for target in target_layer:
                '''
                I need separate 'target' and 'sub_target'.
                'target' indicates which weight I need to use (i.e. where the target fits in the bigger
                picture of the whole network), while 'sub_target' tells me where the target fits
                in the smaller subsystem we are considering for now (i.e. what I feed into 'perceptron')
                '''
                connections = [ [control, target ] for control in control_layer ]
                if self.network_type in ['AR', 'AR with single']:
                    connections = connections + [target]
                ind_list = [ self.edges.index(c) for c in connections ]
                if self.network_type in ['AR', 'AR with single']:
                    current_weights = weights[ind_list]
                    sub_controls = [x for x in range(len(control_layer))]
                    U_list.append( AR_perceptron( current_weights, N_sub, sub_controls, sub_target) )
                elif self.network_type in ['DQNN']:
                    '''
                    DQNN requires a bit of mucking about to deal with the fact that there are 3 weights per edge
                    '''
                    ind_list = [3*i for i in ind_list]
                    weight_list = []
                    for i in ind_list:
                        weight_list.append(i)
                        weight_list.append(i+1)
                        weight_list.append(i+2)
                    current_weights = weights[weight_list]
                    sub_controls = [x for x in range(len(control_layer))]
                    for control in sub_controls:
                        control_list = [3*control, 3*control+1, 3*control+2]
                        control_weights = current_weights[control_list]
                        U_list.append( CAN_gate(control_weights, N_sub, control, sub_target) )

                # Add single-qubit gates
                if self.network_type in ['DQNN', 'AR with single']:
                    single_list = [ N_E+3*target, N_E+3*target+1, N_E+3*target+2 ]
                    U_list.append( singleQubitGate( weights[single_list], N_sub, sub_target ) )
                sub_target += 1

            gates = qt.identity(N_sub*[2])
            for U in U_list:
                gates = U*gates

            state = gates*state*gates.dag()


            state = state.ptrace([*range(len(control_layer),N_sub)])
            
        return state


    def run_ARnetwork(self,weights=None,initial=None):
        if weights is None:
            weights = self.weights
        for step in range(self.N_layer-1):
            N_sub = self.S[step] + self.S[step+1]  # number of qubits in the subsystem we consider for now
            control_layer = self.L[step]
            target_layer = self.L[step+1]
            if step==0:
                if initial==None:
                    state = qt.ket('0'*N_sub) * qt.bra('0'*N_sub)
                else:
                    state = initial
                    if state.dims[1][0]==1:
                        state = state * state.dag()
            else:
                state = qt.tensor(state, qt.ket('0'*len(target_layer))*qt.bra('0'*len(target_layer))   )
            U_list = []
            sub_target = len(control_layer)
            for target in target_layer:
                '''
                I need separate 'target' and 'sub_target'.
                'target' indicates which weight I need to use (i.e. where the target fits in the bigger
                picture of the whole network), while 'sub_target' tells me where the target fits
                in the smaller subsystem we are considering for now (i.e. what I feed into 'perceptron')
                '''
                connections = [ [control, target ] for control in control_layer ] + [target]
                ind_list = [ self.edges.index(c) for c in connections ]
                current_weights = weights[ind_list]
                sub_controls = [x for x in range(len(control_layer))]
                U_list.append( AR_perceptron( current_weights, N_sub, sub_controls, sub_target) )
                sub_target += 1
            gates = qt.identity(N_sub*[2])
            for U in U_list:
                gates = U*gates
            
            state = gates*state*gates.dag()

            state = state.ptrace([*range(len(control_layer),N_sub)])
            
        return state
    
    
    def diss_core(self, out=0, weights=None):
        '''
        The expected value of and output qubit alpha of a network with unitary
        gates U(theta) and quantum feature map F(x) can be written 
        <0|F^\dag(x) U^\dag(theta) p_alpha U(theta) F(x)|0>
        If we can calculate the core section, O = U^\dag(theta) p_alpha U(theta) beforehand,
        then we can easily average this over many different inputs x without running the
        entire circuit each time.
        '''
        
        if weights is None:
            weights = self.weights
        if self.network_type == 'DQNN':
            N_E = 3*len(self.edges)
        elif self.network_type == 'AR with single':
            N_E = len(self.edges)
        for step in reversed(range(self.N_layer-1)):
            N_sub = self.S[step] + self.S[step+1]  # number of qubits in the subsystem we consider for now
            control_layer = self.L[step]
            target_layer = self.L[step+1]

            if step == self.N_layer-2:
                # This is the expected value of the target output qubit 'out'
                # The final two layers should have size S[-1]+S[-2], and in the
                # standard ordering, 'out' should be qubit number S[-1]+out
                n_out = op.gate_expand_1toN(qt.num(2),self.S[-1]+self.S[-2],self.S[-2]+out)
            else:
                # add new layer
                n_out = qt.tensor( qt.qeye([2]*len(control_layer)), n_out )
            U_list = []
            sub_target = N_sub-1
            for target in reversed(target_layer):
                '''
                I need separate 'target' and 'sub_target'.
                'target' indicates which weight I need to use (i.e. where the target fits in the bigger
                picture of the whole network), while 'sub_target' tells me where the target fits
                in the smaller subsystem we are considering for now (i.e. what I feed into 'perceptron')
                '''
                connections = [ [control, target ] for control in control_layer ]
                if self.network_type in ['AR', 'AR with single']:
                    connections = connections +  [target]
                ind_list = [ self.edges.index(c) for c in connections ]
                if self.network_type in ['AR', 'AR with single']:
                    if self.network_type == 'AR with single':
                        single_list = [ N_E+3*target, N_E+3*target+1, N_E+3*target+2 ]
                        U_list.append( singleQubitGate( weights[single_list], N_sub, sub_target ) )
                    current_weights = weights[ind_list]
                    sub_controls = [x for x in range(len(control_layer))]
                    U_list.append( AR_perceptron( current_weights, N_sub, sub_controls, sub_target) )
                elif self.network_type == 'DQNN':
                    # Add single-qubit gates
                    single_list = [ N_E+3*target, N_E+3*target+1, N_E+3*target+2 ]
                    U_list.append( singleQubitGate( weights[single_list], N_sub, sub_target ) )

                    '''
                    DQNN requires a bit of mucking about to deal with the fact that there are 3 weights per edge
                    '''
                    ind_list = [3*i for i in ind_list]
                    weight_list = []
                    for i in ind_list:
                        weight_list.append(i)
                        weight_list.append(i+1)
                        weight_list.append(i+2)
                    current_weights = weights[weight_list]
                    sub_controls = [x for x in range(len(control_layer))]
                    for control in reversed(sub_controls):
                        control_list = [3*control, 3*control+1, 3*control+2]
                        control_weights = current_weights[control_list]
                        U_list.append( CAN_gate(control_weights, N_sub, control, sub_target) )
                sub_target -= 1

            if step==0 and self.network_type in ['DQNN', 'AR with single']:
                for target in reversed(control_layer):
                    single_list = [ N_E+3*target, N_E+3*target+1, N_E+3*target+2 ]
                    U_list.append( singleQubitGate( weights[single_list], N_sub, target ) )

            gates = qt.identity(N_sub*[2])
            for U in reversed(U_list):
                gates = U*gates

            n_out = gates.dag() * n_out * gates

            P = qt.tensor( qt.qeye([2]*len(control_layer)),
                           qt.ket('0'*len(target_layer)) )
            n_out = P.dag() * n_out * P

        return n_out
    
    
    def AR_core(self, out=0, weights=None):
        '''
        The expected value of and output qubit alpha of a network with unitary
        gates U(theta) and quantum feature map F(x) can be written 
        <0|F^\dag(x) U^\dag(theta) p_alpha U(theta) F(x)|0>
        If we can calculate the core section, O = U^\dag(theta) p_alpha U(theta) beforehand,
        then we can easily average this over many different inputs x without running the
        entire circuit each time.
        '''
        
        if weights is None:
            weights = self.weights
        
        for step in reversed(range(self.N_layer-1)):
#             print('doing step ' + str(step))

            N_sub = self.S[step] + self.S[step+1]  # number of qubits in the subsystem we consider for now
            control_layer = self.L[step]
            target_layer = self.L[step+1]

            if step == self.N_layer-2:
                # This is the expected value of the target output qubit 'out'
                # The final two layers should have size S[-1]+S[-2], and in the
                # standard ordering, 'out' should be qubit number S[-1]+out
                n_out = op.gate_expand_1toN(qt.num(2),self.S[-1]+self.S[-2],self.S[-2]+out)
            else:
                # add new layer
                n_out = qt.tensor( qt.qeye([2]*len(control_layer)), n_out )

                
            U_list = []
            sub_target = len(control_layer)
            for target in target_layer:
                '''
                I need separate 'target' and 'sub_target'.
                'target' indicates which weight I need to use (i.e. where the target fits in the bigger
                picture of the whole network), while 'sub_target' tells me where the target fits
                in the smaller subsystem we are considering for now (i.e. what I feed into 'perceptron')
                '''
                connections = [ [control, target ] for control in control_layer ] + [target]
                ind_list = [ self.edges.index(c) for c in connections ]
                current_weights = weights[ind_list]
                sub_controls = [x for x in range(len(control_layer))]
                U_list.append( AR_perceptron( current_weights, N_sub, sub_controls, sub_target) )
                sub_target += 1
            gates = qt.identity(N_sub*[2])
            for U in U_list:
                gates = U*gates
            
            n_out = gates.dag() * n_out * gates

            
            P = qt.tensor( qt.qeye([2]*len(control_layer)),
                           qt.ket('0'*len(target_layer)) )
            n_out = P.dag() * n_out * P

        return n_out
    
    
    def run_HEAnetwork(self, psi, weights=None):
        '''
        The variational ansazt used in the 'Power of Quantum Neural Networks' paper
        '''
        
        if weights is None:
            weights = self.weights
        
        cnots = []
        for ind1 in range(1,self.N_qubit):
            for ind2 in range(ind1):
                cnots.append( op.cnot(N=self.N_qubit, target=ind1, control=ind2) )
        CNOTS = qt.qeye([2]*self.N_qubit)
        for u in cnots:
            CNOTS = u*CNOTS

        for l in range(self.N_layer+1):   
            ry = [ op.ry( weights[l*self.N_qubit + ind], N=self.N_qubit, target=ind ) 
                  for ind in range(self.N_qubit)   ]
            RY = qt.qeye([2]*self.N_qubit)
            for u in ry:
                RY = u*RY
            if l < self.N_layer:
                psi = CNOTS*RY*psi
            else:
                psi = RY*psi

#         psi = op.hadamard_transform(self.N_qubit)*psi

        if len(self.S)==2 or (len(self.S)==3 and self.S[0]==self.S[2]):
            return psi
        else:
            return psi.ptrace(range(self.S[2]))
        
    def HEA_core(self, out=0, weights=None):
    
        if weights is None:
            weights = self.weights
        
        n_out = op.gate_expand_1toN(qt.num(2),self.N_qubit,out)
        
        cnots = []
        for ind1 in range(1,self.N_qubit):
            for ind2 in range(ind1):
                cnots.append( op.cnot(N=self.N_qubit, target=ind1, control=ind2) )
        CNOTS = qt.qeye([2]*self.N_qubit)
        for u in cnots:
            CNOTS = u*CNOTS
        
        for l in reversed(range(self.N_layer+1)):
            ry = [ op.ry( weights[l*self.N_qubit + ind], N=self.N_qubit, target=ind ) 
                  for ind in range(self.N_qubit)   ]
            RY = qt.qeye([2]*self.N_qubit)
            for u in ry:
                RY = u*RY
            
            if l == 0:
                n_out = RY.dag() * n_out * RY
            else:
                n_out = CNOTS.dag() * RY.dag() * n_out * RY * CNOTS
                
        return n_out
        
        
    def run_QAOAnetwork(self, psi, weights=None):
        if weights is None:
            weights=self.weights
            
        for l in range(self.N_layer):
            psi = (-1j*weights[self.N_layer + l]*self.B_gate).expm() * (-1j*weights[l]*self.A_gate).expm() * psi
            
        if self.S[0] == self.S[-1]:
            return psi
        else:
            return psi.ptrace(range(self.S[2]))
        
    def QAOA_core(self, out=0, weights=None):
        if weights is None:
            weights = self.weights
        
        n_out = op.gate_expand_1toN(qt.num(2), self.N_qubit, out)
        
        for l in reversed( range(self.N_layer) ):
            U = (-1j*weights[self.N_layer + l]*self.B_gate).expm() * (-1j*weights[l]*self.A_gate).expm()
            n_out = U.dag() * n_out * U
        return n_out