#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 11:19:21 2021

@author: sam
"""

import scipy as sp
import numpy as np
from math import pi
import time

from QuantumNetwork import QuantumNetwork

import qutip as qt



#%%
# Defining functions

def tensoredId(N):
    #Make Identity matrix
    res = qt.qeye(2**N)
    #Make dims list
    dims = [2 for i in range(N)]
    dims = [dims.copy(), dims.copy()]
    res.dims = dims
    #Return
    return res


def randomQubitUnitary(numQubits):
    # Generates random N-qubit unitary
    # Hilariously, this is *significantly* quicker than the built-in qutip function
    dim = 2**numQubits
    #Make unitary matrix
    res = sp.random.normal(size=(dim,dim)) + 1j * sp.random.normal(size=(dim,dim))
    res = sp.linalg.orth(res)
    res = qt.Qobj(res)
    #Make dims list
    dims = [2 for i in range(numQubits)]
    dims = [dims.copy(), dims.copy()]
    res.dims = dims
    #Return
    return res


def randomQubitState(numQubits):
    dim = 2**numQubits
    #Make normalized state
    res = sp.random.normal(size=(dim,1)) + 1j * sp.random.normal(size=(dim,1))
    res = (1/sp.linalg.norm(res)) * res
    res = qt.Qobj(res)
    #Make dims list
    dims1 = [2 for i in range(numQubits)]
    dims2 = [1 for i in range(numQubits)]
    dims = [dims1, dims2]
    res.dims = dims
    #Return
    return res


def randomTrainingData(unitary, N):
    numQubits = len(unitary.dims[0])
    trainingData=[]
    #Create training data pairs
    for i in range(N):
        t = randomQubitState(numQubits)
        ut = unitary*t
        trainingData.append([t,ut])
    #Return
    return trainingData


def cost(network, data, weights=None):
    # Cost function for learning a unitary given labelled input-output pairs
    if weights is None:
        weights = network.weights
        print('YOU FORGOT TO GIVE ME WEIGHTS')
        
    cost_list = []
    for t in data:
        in_state = t[0]
        label_state = t[1]
        out_state = network.runNetwork(initial=in_state,weights=weights)
        if network.network_type in ['HEA', 'QAOA']:
            cost_list.append( (out_state.dag() * label_state)[0][0][0] )
        else:
            cost_list.append( qt.expect(out_state, label_state) )


    return np.sum(cost_list)/len(data)


def trainUnitary(thetas, network, trainingData, validationData, steps, eta, eps, batch, noisy=False):
    
    cost_list = []
    validation_list = []
    parameter_list = []
    if noisy:
        start = time.time()

    for s in range(steps):
        cost_list.append(cost(network, trainingData, thetas))
        validation_list.append( cost(network, validationData, thetas)  )

        grad_vec = []
        target_thetas = list(np.random.permutation(np.arange(0,len(thetas)))[:batch])
        for ind in range(len(thetas)):
            if ind in target_thetas:
                diff = np.zeros(network.N_params)
                diff[ind] = eps

                forward = cost(network, trainingData, weights=thetas+diff)
                backward = cost(network, trainingData, weights=thetas-diff)
                finite_diff = (forward - backward)/(2*eps)
            else:
                finite_diff = 0

            grad_vec.append(finite_diff)
        
        thetas = thetas - eta*np.array(grad_vec)

        
        if noisy:
            current = time.time()
            print('Finished step ' + str(s))
            print(str((current-start)/60) + ' minutes elasped so far')

        cost_list.append(cost(network, trainingData, thetas))
        validation_list.append( cost(QNN, validationData, thetas)  )
        parameter_list.append(thetas)
    
    return cost_list, validation_list, parameter_list


#%%

note = '''
Include description of simulation for self-reference
'''

saving = True
filename = 'UnitaryLearning/DQNN'

start = time.time()

print(note)


#%%
# Define networks

S = [2,2,2] # Structure of network

N_S = 50 # Number of training samples
N_V = 10 # Number of validation samples
time_steps = 50
eps = 1e-3
eta = 1
batch = 15

QNN = QuantumNetwork(S,network_type='DQNN',fmap='none')

#%%
# Define data
targetU = randomQubitUnitary(S[-1])
trainingData = randomTrainingData(targetU, N_S)
validationData = randomTrainingData(targetU, N_V)

#%%
start = time.time()

costs = []
validations = []
parameter_trajectories = []

# Initial parameters.
thetas = pi*( np.random.rand(QNN.N_params) - 0.5 )
if QNN.network_type in ['AR', 'AR with single']:
    thetas = 10*thetas

# Train network, get training cost, validation cost and new parameters
costs, validations, parameters = trainUnitary(thetas, QNN, trainingData, 
                                                      validationData, time_steps, eta, eps, batch, noisy=False)

finish = time.time()
duration = (finish-start)/3600  # time in hours
print('Finished all training')
print(str(duration) + ' hours elapsed')



#%%
# Package all of the data to save
if saving:
    import pickle
    inputs = {'QNN':QNN, 'N_S':N_S, 'N_V':N_V,
              'eta':eta, 'eps':eps,
              'time_steps':time_steps, 'targetU':targetU,
              'trainingData':trainingData, 'validationData':validationData,
              'note':note}
    
    results = { 
                'costs':costs,
                'validations':validations,
                'parameters':parameters,
                'duration':duration
               }
    
    pickling_on = open(filename+"_inputs.pickle","wb")
    pickle.dump(inputs, pickling_on)
    pickling_on.close()
    
    pickling_on = open(filename+"_results.pickle","wb")
    pickle.dump(results, pickling_on)
    pickling_on.close()

print('finished saving')
print('finished everything')