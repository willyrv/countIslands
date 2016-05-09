# -*- coding: utf-8 -*-
import sys
import math
import numpy as np
import decimal
from decimal import Decimal
from scipy.optimize import fmin
from scipy.optimize import fminbound

def compute_constants(n, M):
    A = 1+(float(n)/(n-1))*M
    B = np.sqrt(A**2-float(4*M)/(n-1))
    a = 0.5 + (1.0 + (float(n-2)*M)/(n-1))/(2*B)
    alpha = 0.5*(A+B)
    beta = 0.5*(A-B)
    E = (1 + (float(n-2)*M)/(n-1)) / (2*B)
    AplusB = A+B
    AminusB = A-B
    return [a, alpha, beta]
    
def prob_distrib(k, theta, a, alpha, beta):
    factor1 = (1.0/(1+alpha/(2*theta)))**k
    factor2 = (1.0/(1+beta/(2*theta)))**k
    if factor1+factor2 > 0:
        result = factor1*float(a)/(alpha+2*theta) + factor2*(1.0-a)/(beta+2*theta)
        if result > 0:
            return result
        else:
            return Decimal(factor1)*Decimal(float(a)/(alpha+2*theta)) + Decimal(factor2)*Decimal((1.0-a)/(beta+2*theta))
    else:
        factor1 = Decimal(1.0/(1+alpha/(2*theta)))**k
        factor2 = Decimal(1.0/(1+beta/(2*theta)))**k
        return factor1*Decimal(float(a)/(alpha+2*theta)) + factor2*Decimal((1.0-a)/(beta+2*theta))
    
def log_likelihood(obs, n, M, theta):
    """
    This function computes the likelihood of the parameters n, M and theta
    with respect to the observations.
    """
    if (n<2) or (M<0):
        return -sys.maxint
    else:
        [a, alpha, beta] = compute_constants(n, M)
        temp = 0
        computed_values = {} # We store the computed values.
        for k in obs:
            if computed_values.has_key(k):
                temp+=computed_values[k]
            else:
                prob_k = prob_distrib(k, theta, a, alpha, beta)
                # For some values, prob_k is a Decimal instance and we need
                # to use the Decimal method for the log                
                if type(prob_k) == decimal.Decimal:
                    log_prob_k = prob_k.ln()
                    computed_values[k] = float(log_prob_k)
                else:
                    computed_values[k] = math.log(prob_k)
                temp+=computed_values[k]
        return temp
        
def log_likelihood_hist(obs, n, M, theta):
    """
    This function computes the likelihood of the parameters n, M and theta
    with respect to the observations. Assumes the observations are sorted
    from 0 to max_ndiff in a histogram
    """
    if (n<2) or (M<0):
        return -sys.maxint
    else:
        [a, alpha, beta] = compute_constants(n, M)
        temp = 0
        for k in range(len(obs)):
            if obs[k] != 0:
                prob_k = prob_distrib(k, theta, a, alpha, beta)
                # For some values, prob_k is a Decimal instance and we need
                # to use the Decimal method for the log
                prob_k = prob_distrib(k, theta, a, alpha, beta)
                if type(prob_k) == decimal.Decimal:
                    log_prob_k = prob_k.ln()
                else:
                    log_prob_k = math.log(prob_k)
                temp+= obs[k] * log_prob_k
        return temp
        
def max_llk_est_full_n(obs, theta):
    """
    Estimates the parameters n and M that maximize the log-likelihood of 
    the observations.
    We take the best value of M for every n positive integer such that 
    min_n <= n <= max_n
    """
    min_n=2
    max_n=1000
    min_M = 0
    max_M=1000
    nvalues = np.arange(min_n+1, max_n+1)
    llM = lambda x: -log_likelihood(obs, min_n, x, theta)
    temp_n = min_n
    # the method minimize_scalar only works from version 0.11.0
    #temp_result = minimize_scalar(llM, bounds=(min_M, max_M), method='bounded')

    #here we use version 0.9
    temp_result = fminbound(llM, min_M, max_M, full_output=True, disp=False)
    for n in nvalues:
        llM = lambda x: -log_likelihood(obs, n, x, theta)
        #res = minimize_scalar(llM, bounds=(min_M, max_M), method='bounded')
        res = fminbound(llM, min_M, max_M, full_output=True, disp=False)
        if res[1] < temp_result[1]:
            temp_result = res
            temp_n = n
    return [temp_n, temp_result[0], -temp_result[1]]
    
def max_llk_est(obs, theta):
    """
    Estimates the parameters n and M that maximize the log-likelihood of 
    the observations.
    The maximum likelihood is approximated by doing a maximization on 
    RxR. Then the we do a maximization on R for each n in the intervall
    [estim_n-5, estim_n+5]. Returns the best of the estimated values.
    """
    # Find an approximation for n real
    llk = lambda x: -log_likelihood(obs, x[0], x[1], theta)
    (n_real, M_real) = fmin(llk, [10, 1], disp=False)
    
    # Take an intervall arround n_real    
    n_integer = int(n_real)
    min_M = 0
    max_M = 1000
    n_list = np.arange(max(2, n_integer-5), n_integer+5, 1)
    min_n = n_list[0]
    llM = lambda x: -log_likelihood(obs, min_n, x, theta)
    temp_n = min_n
    # the method minimize_scalar only works from version 0.11.0
    #temp_result = minimize_scalar(llM, bounds=(min_M, max_M), method='bounded')

    #here we use version 0.9
    temp_result = fminbound(llM, min_M, max_M, full_output=True, disp=False)
    for n in n_list[1:]:
        llM = lambda x: -log_likelihood(obs, n, x, theta)
        #res = minimize_scalar(llM, bounds=(min_M, max_M), method='bounded')
        res = fminbound(llM, min_M, max_M, full_output=True, disp=False)
        if res[1] < temp_result[1]:
            temp_result = res
            temp_n = n
    return [temp_n, temp_result[0], -temp_result[1]]
    
def max_llk_estim_hist(obs, theta):
    """
    Estimates the parameters n and M that maximize the log-likelihood of 
    the observations.
    The maximum likelihood is approximated by doing a maximization on 
    RxR. Then the we do a maximization on R for each n in the intervall
    [estim_n-5, estim_n+5]. Returns the best of the estimated values.
    The input is assumed to be a histogram.
    """
    # Find an approximation for n real
    llk = lambda x: -log_likelihood_hist(obs, x[0], x[1], theta)
    (n_real, M_real) = fmin(llk, [10, 1], disp=False)
    
    # Take an intervall arround n_real   
    n_integer = int(n_real)
    min_M = 0
    max_M = 1000
    n_list = np.arange(max(2, n_integer-5), n_integer+5, 1)
    min_n = n_list[0]
    llM = lambda x: -log_likelihood_hist(obs, min_n, x, theta)
    temp_n = min_n
    # the method minimize_scalar only works from version 0.11.0
    #temp_result = minimize_scalar(llM, bounds=(min_M, max_M), method='bounded')

    #here we use version 0.9
    temp_result = fminbound(llM, min_M, max_M, full_output=True, disp=False)
    for n in n_list[1:]:
        llM = lambda x: -log_likelihood_hist(obs, n, x, theta)
        #res = minimize_scalar(llM, bounds=(min_M, max_M), method='bounded')
        res = fminbound(llM, min_M, max_M, full_output=True, disp=False)
        if res[1] < temp_result[1]:
            temp_result = res
            temp_n = n
    return [temp_n, temp_result[0], -temp_result[1]]    
