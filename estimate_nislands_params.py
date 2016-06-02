# -*- coding: utf-8 -*-
"""
Created on Mon May  2 14:48:39 2016

@author: willy
"""

import re
import argparse

from nisland_model import max_llk_estim_hist_theta_variable

parser = argparse.ArgumentParser(description='Estimates the parameters of an n-island model.')


# The file with the data (.ndiff)
parser.add_argument('input_file', type=str,
                      help='Full path to the input data.')

parameters = parser.parse_args()
INPUT_FILENAME = parameters.input_file

# Reading the file
with open(INPUT_FILENAME, 'r') as f:
    text = f.read()
(header, obs_text) = text.split('# Value|count\n')

# Get the observed values    
obs = []
pattern = '[0-9]*\|[0-9]*\n'
p = re.compile(pattern)
for line in p.findall(obs_text):
    obs.append(int(line.split('|')[1]))
    
# Do the maximum likelihood estimation 
[n, M, theta, llk] = max_llk_estim_hist_theta_variable(obs)

# Print the results

print("Number of islands: {}. Migration rate: {} ".format(n, M))
print("theta: {}".format(theta))
print("Value of the log-likelihood function: {}".format(llk))
