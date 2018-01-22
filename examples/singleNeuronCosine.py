#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 20:18:56 2017

@author: conorquinn
"""

import numpy as np
import matplotlib.pyplot as plt
import decimal as dc
import nengo
import time
from nengo.dists import Uniform


t0 = time.time()
model = nengo.Network(label='A Single Neuron')
with model:
    neuron = nengo.Ensemble(1, dimensions=1, # Represent a scalar
                            intercepts=Uniform(-.5, -.5),  # Set intercept to 0.5
                            max_rates=Uniform(100, 100),  # Set the maximum firing rate of the neuron to 100hz
                            encoders=[[1]], seed=10)  # Sets the neurons firing rate to increase for positive input

with model:
    cos = nengo.Node(lambda t: np.cos(8 * t))
    
with model:
    # Connect the input signal to the neuron
    nengo.Connection(cos, neuron)
with model:
    cos_probe = nengo.Probe(cos)  # The original input
    spikes = nengo.Probe(neuron.neurons)  # The raw spikes from the neuron
    voltage = nengo.Probe(neuron.neurons, 'voltage')  # Subthreshold soma voltage of the neuron
    filtered = nengo.Probe(neuron, synapse=0.01) # Spikes filtered by a 10ms post-synaptic filter
#print(filtered)
sim = nengo.Simulator(model) # Create the simulator
sim.run(1) # Run it for 1 seconds

# Plot the decoded output of the ensemblev
plt.plot(sim.trange(), sim.data[filtered])
plt.plot(sim.trange(), sim.data[cos_probe])

plt.ylim(np.min(sim.data[cos_probe]), np.max(sim.data[cos_probe]))

# Plot the spiking output of the ensemble
from nengo.utils.matplotlib import rasterplot
plt.figure(figsize=(10, 8))
plt.subplot(221)
rasterplot(sim.trange(), sim.data[spikes])
plt.ylabel("Neuron")
#plt.xlim(0, 1)

# Plot the soma voltages of the neurons
plt.subplot(222)
plt.plot(sim.trange(), sim.data[voltage][:,0], 'r')
#plt.xlim#(0, 1);
t1= time.time()
total = t1-t0

print('Time: ' + str(total))
fileName = str(nengo.rc.get('precision', 'dtype'))+'Cosine.txt'
cosValues = sim.data[filtered]
np.savetxt(fileName,cosValues, delimiter = ', ')
np.savetxt('rangeCosine.txt',sim.trange(),delimiter = ', ')
