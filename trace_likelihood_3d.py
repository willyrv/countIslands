# -*- coding: utf-8 -*-
"""
Created on Mon May  2 14:48:39 2016

@author: willy
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import re
from nisland_model import log_likelihood_hist

# The file with the data (.ndiff)
INPUT_FILENAME = './data_test.ndiff'

# Range for the parameters (start, end, step)
n_values = np.arange(2, 100, 1)
M_values = np.arange(0.01, 5, 0.01)

# Mutation rate, set to False for take the value of the file, otherwise
# specify a value
theta = False

if __name__ == '__main__':
    # Reading the file
    with open(INPUT_FILENAME, 'r') as f:
        text = f.read()
    (header, obs_text) = text.split('# Value|count\n')
    if not theta:
        theta = float(header.split('\n')[3])
    # Get the observed values    
    obs = []
    pattern = '[0-9]*\|[0-9]*\n'
    p = re.compile(pattern)
    for line in p.findall(obs_text):
        obs.append(int(line.split('|')[1]))
    
    # Evaluate the likelihood
    (X, Y) = np.meshgrid(n_values, M_values)
    zs = np.array([log_likelihood_hist(obs, x, y, theta)
                    for (x,y) in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    # Do the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
    ax.plot_surface(X, Y, Z, rstride=10, cstride=10, color='b')
    ax.set_xlabel('n values')
    ax.set_ylabel('M values')
    ax.set_zlabel('log-likelihood')
    plt.show()
