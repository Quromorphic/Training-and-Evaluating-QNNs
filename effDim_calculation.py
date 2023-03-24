#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 10:20:25 2021

@author: sam
"""

import numpy as np
from math import pi
import time


from QuantumNetwork import QuantumNetwork



#%%

note = '''
Effective dimension as a function of circuit depth (no. of parameters)
for an AR network with single qubit rotations
'''

saving = True
filename = 'effDim/depth_DQNN'

start = time.time()

print(note)

#%%

inputs = 4
outputs = 3

N_S = 10
N_theta = 5
eps = 1e-3
sample_list = np.array( range(int(1e2),int(1e6),int(1e1)) )

network_type = 'DQNN'


data = [np.random.rand(inputs) for _ in range(N_S)]

#%%

# List of structures to get effective dimension of
S_list = [ [ inputs, 3, outputs ],
            [ inputs, 4, outputs ],
            [ inputs, 3, 3, outputs ],
            [ inputs, 4, 3, outputs ],
            [ inputs, 4, 4, outputs ],
          ]  


samples = 1e6 # Number of samples in eff. dim. calculation
effD_easy = []
effD_hard = []

N_params = []

#%%

for x in range(len(S_list)):
    S = S_list[x]
    
    # 'easy' and 'hard' refer to choice of feature map
    
    QNN_easy = QuantumNetwork(S,network_type=network_type,fmap='easy')
    # QNN_hard = QuantumNetwork(S,network_type=network_type,fmap='hard')

    N_params.append(QNN_easy.N_params)
    thetas = pi*(np.random.rand(N_theta,QNN_easy.N_params) - 0.5)
    
    print('calculating for easy network')
    grad_easy = QNN_easy.gradCore(thetas[x%len(S)])
    F_easy = np.array([ QNN_easy.trickyFish(data, g) for g in grad_easy ])
    effD_easy.append( QNN_easy.effectiveDimension(F_easy, samples) )
    
    # print('calculating for hard network')
    # grad_hard = QNN_hard.gradCore(thetas[x%len(S)])
    # F_hard = np.array([ QNN_hard.trickyFish(data, g) for g in grad_hard ])
    # effD_hard.append( QNN_hard.effectiveDimension(F_hard, samples) )
    
    print('Finished for ' + str(QNN_easy.N_params) + ' parameters')
    current = time.time()
    print( str((current-start)/60) + ' minutes elapsed so far')


finish = time.time()
duration = (start-finish)/3600
print('All done!')
print('That took ' + str(duration) + ' hours')

#%%

if saving:
    import pickle
    inputs = {'S':S, 'network_type':network_type, 'N_S':N_S, 'N_theta':N_theta,
              'eps':eps, 'sample_list':sample_list, 'data':data, 'N_params':N_params,
              'note':note}
    
    results = { 
                'effD_easy':effD_easy,
                'effD_hard':effD_hard,
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