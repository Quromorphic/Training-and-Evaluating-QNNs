#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 10:22:43 2021

@author: sam
"""

import scipy as sp
import samHelper as sh
import numpy as np
from math import pi
import time
import multiprocessing as mp

from QuantumNetwork import QuantumNetwork

import qutip as qt
import qutip.qip.operations as op
import qutip.qip.circuit as qc

import csv

#%%

def cost(network,core,inputs,labels):
    cost_list = []
    for x in range(len(labels)):
        psi_x = inputs[x]
        y_x = labels[x]
        
        out_vec = [ qt.expect( core[out], psi_x ) for out in range(network.S[-1]) ]
        cost_list.append( np.linalg.norm(out_vec - y_x)**2 )
        
    return sum(cost_list)/len(cost_list)

def train(network, data, samples, steps, theta=None, eta=1, batch=None, alpha=0.01, beta1=0.9, beta2=0.99,
          eps=1e-3, noisy=False, lam=0):
    
    if noisy:
        start_train = time.time()
        print('Training ' + network.network_type + ' with ' 
                  + network.fmap + ' feature map')
    
    # Partition data into training and testing sets
    np.random.shuffle(data)
    data_train = data[:samples]
    data_test = data[samples:]
    
    
    # set up feature vectors and training labels
    Psi = []
    testPsi = []
    training_label = []
    testing_label = []
    for d in data_train:
        Psi.append( network.load_features(d[1:]) )

        y_ind = int(d[0])-1
        y_x = np.zeros(network.S[-1])
        y_x[y_ind] = 1
        training_label.append(y_x)
        
    for d in data_test:
        testPsi.append( network.load_features(d[1:]) )

        y_ind = int(d[0])-1
        y_x = np.zeros(network.S[-1])
        y_x[y_ind] = 1
        testing_label.append(y_x)
        

    # initialize first and second moments
    m = np.array([0.0 for _ in range(network.N_params)])
    v = np.array([0.0 for _ in range(network.N_params)])
    
    
    # set up parameter vector
    if theta is None:
        theta = network.weights
    
    cost_list = []
    generalisation_cost = []
    
    # train via Adam optimiser
    for t in range(steps):
        # Calculate gradient of cost function
        grad_core = network.gradCore([theta],eps=eps,batch=batch)
        if t==0:
            cost_core = [ network.networkCore(theta,out=out) for out in range(network.S[-1]) ]
        else:
            new_core = [ network.networkCore(theta,out=out) for out in range(network.S[-1]) ]

            
            cost_core = new_core

        grad_C = []

        for x in range(len(data_train)):
            # Define label
            y_x = training_label[x]
            psi_x = Psi[x]


            grad_C.append( sum([ (y_x[ind] - qt.expect(cost_core[ind],psi_x))
                            * qt.expect(grad_core[0][ind],psi_x) for ind in range(network.S[-1]) ]) )

        grad_C = np.array( sum(grad_C)/len(data_train) ) + 2*lam*theta

#         Update parameters
        for ind in range(len(theta)):
            m[ind] = beta1 * m[ind] + (1.0 - beta1) * grad_C[ind]
            v[ind] = beta2 * v[ind] + (1.0 - beta2) * grad_C[ind]**2

        mhat = m / (1.0 - beta1**(t+1))
        vhat = v / (1.0 - beta2**(t+1))
            
#         set_trace()
        
        theta = theta + alpha * mhat / (np.sqrt(vhat) + eps)
        
#         theta = np.random.rand(network.N_params)
    
        # Calculate new cost (and add to list)
        current_cost = cost(network, cost_core, Psi, training_label)\
                        + lam*np.linalg.norm(theta)**2
        cost_list.append(current_cost)
        
#         set_trace()
        test_cost = cost(network, cost_core, testPsi, testing_label)\
                        + lam*np.linalg.norm(theta)**2
        generalisation_cost.append(test_cost)


        if noisy:
            if not t%1000:
                print('step ' + str(t) + ' completed')
                current = time.time()
                print(str((current-start)/60) + ' minutes elasped so far')

    
    if noisy:
        finish_train = time.time()
    
        print('Finished training ' + network.network_type + ' with ' 
                  + network.fmap + ' feature map')
        print('that took ' + str((finish_train-start_train)/60) + ' minutes')
        
    
    
    return cost_list, generalisation_cost, theta
    

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

note = '''
Increasing size of AR+ network so it has same number of parameters as the others.
First, running a smaller, simpler version to see if it is likely to finish on time.
'''

saving = True
dirname = 'NetworkTraining'
filename = 'HEA_iris_[4,4,4,3]'

if saving:
    print('we are saving')
    import os, shutil

    if not os.access(dirname, os.F_OK):
        os.mkdir(dirname, 0o700)
        
    # Save copy of current script.
    # os.path.basename(__file__)[:-3] gives name of script without .py
    # I append the current filename after ___
    shutil.copy(__file__, dirname + '/' + os.path.basename(__file__)[:-3]
                + '___' + filename + '.py') 

print(note)

#%%
# Generate networks

# number of qubits in input and output layers
inputs = 4
outputs = 3

# structure of model
S_list = [
           [inputs, 4, 4, outputs]
         ] * 4


N_S = 30
eps = 1e-3
time_steps = int(5e2)
eta = 10

network_type = 'AR with single'

## 'easy' and 'hard' refer to choice of feature map. Depending on time costraints, one can include either or both
networks_easy = [ QuantumNetwork(S, network_type=network_type, fmap='easy') for S in S_list  ]
# networks_hard = [ QuantumNetwork(S, network_type=network_type, fmap='hard') for S in S_list  ]
networks = networks_easy # + networks_hard

N_p = networks[0].N_params
batch = [None for _ in range(len(networks))] 

#%%
# Get data
datatype = 'iris'

data = get_data(datatype,inputs=inputs,outputs=outputs)

#%%
# Train the networks, get the costs and parameters
start = time.time()

costs = []
validations = []
parameters = []

pool = mp.Pool()
pool = mp.Pool(processes=4)

mask = 9*( np.array( [ 1*(QNN.network_type in ['AR', 'AR with single']) for QNN in networks] ) + 1/9)
thetaList = [ mask[ind]*pi*(np.random.rand(networks[ind].N_params) - 0.5) for ind in range(len(networks)) ]

params = [(networks[ind], data, N_S, time_steps, thetaList[ind], eta, batch[ind])
          for ind in range(len(networks))]

training_outputs = pool.starmap(train, params)
for it in training_outputs:
    costs.append(it[0])
    validations.append(it[1])
    parameters.append(it[2])

finish = time.time()
duration = (finish-start)/3600  # (finish-start)/60
print('Finished all training')
print(str(duration) + ' hours elapsed')

#%%
# Package all of the data to save

if saving:
    print('we are really saving')
    import pickle
    inputs = {'networks':networks, 'N_S':N_S, 'eta':eta, 'batch':batch,
              'time_steps':time_steps, 'data':datatype,         
              'note':note}
    
    results = { 
                # 'training_results':training_results,
                'costs':costs,
                'validations':validations,
                'parameters':parameters,
                'duration':duration
               }
    
    pickling_on = open(dirname + '/' + filename+"_inputs.pickle","wb")
    pickle.dump(inputs, pickling_on)
    pickling_on.close()
    
    pickling_on = open(dirname + '/' + filename+"_results.pickle","wb")
    pickle.dump(results, pickling_on)
    pickling_on.close()

    print('finished saving')
print('finished everything')