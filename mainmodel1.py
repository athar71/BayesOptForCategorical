#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 11:31:10 2019
@author: athar

This code is modling bayseian optimization for the cases with categorical variables
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from bayesian_optimization_util import plot_approximation, plot_acquisition
import BayesOpt_utils as BO
import data_generator as datagen
import matplotlib.pyplot as plt

"""
defining inputs:
    The continous input variables could be in R^n , with an arbitrary n
    "m" is the number of categorical input variables 
    The out put variable, assuming one out put
    Number of measurments (budjet)
    The surrogate model(type)
    Acquisition function(type)
    
"""
    

"""
Merging categorical inputs
input_dim = X.shape[1]

"""
m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)

# Initialize samples
noise = 0.2
bounds = np.array([[-1.0, 2.0]])

X_init = np.array([[-0.9], [1.1]])
Y1_init = f1(X_init)
Y2_init = f2(X_init)
Y3_init = f3(X_init)

X1_sample = X_init
Y1_sample = Y1_init
X2_sample = X_init
Y2_sample = Y2_init
X3_sample = X_init
Y3_sample = Y3_init

grid1 = np.arange(bounds[:, 0], bounds[:, 1], 0.01).reshape(-1, 1)
grid2 = np.arange(bounds[:, 0], bounds[:, 1], 0.01).reshape(-1, 1)
grid3 = np.arange(bounds[:, 0], bounds[:, 1], 0.01).reshape(-1, 1)

# Number of iterations
n_iter = 10

plt.figure(figsize=(12, n_iter * 3))
plt.subplots_adjust(hspace=0.4)

for i in range(n_iter):
    # Update Gaussian process with existing samples
    
    
    score1, m1, s1 = OB.GPR_model(kernelf= m52, noise_var=1e-10, optimizerf='fmin_l_bfgs_b',\
                               optimize_restarts=5, X1, Y1, cross_validate= False, grid1)
    
    score2, m2, s2 = OB.GPR_model(kernelf= m52, noise_var=1e-10, optimizerf='fmin_l_bfgs_b',\
                               optimize_restarts=5, X2, Y2, cross_validate= False, grid2)
    
    score3, m3, s3 = OB.GPR_model(kernelf= m52, noise_var=1e-10, optimizerf='fmin_l_bfgs_b',\
                               optimize_restarts=5, X3, Y3, cross_validate= False, grid3)
    
    fmin =BO.get_fmin(np.append(Y1_sample,Y2_sample,Y3_sample))
    # Obtain next sampling point from the acquisition function (expected_improvement)
    
    acq1 = BO._compute_acq(grid1, m1, s1, fmin, epsilon)
    acq2 = BO._compute_acq(grid2, m2, s2, fmin, epsilon)
    acq3 = BO._compute_acq(grid3, m3, s3, fmin, epsilon)
    
    min_aqu_vec = [np.min(acq1),np.min(acq2),np.min(acq3)]
    nxt_smpl_vec = [grid1[np.argmin(acq1)],grid2[np.argmin(acq2)],grid3[np.argmin(acq3)]]
    
    
    X_next, cat_min = BO.pick_next_sample(min_aqu_vec, nxt_smpl_vec)
   
    
    # Obtain next noisy sample from the objective function
    if cat_min==1:
         Y_next = f1(X_next, noise)
         X1_sample = np.vstack((X1_sample, X_next))
         Y1_sample = np.vstack((Y1_sample, Y_next))
        
    elif cat_min==2:
         Y_next = f2(X_next, noise)
         X2_sample = np.vstack((X2_sample, X_next))
         Y2_sample = np.vstack((Y2_sample, Y_next))
         
    else :
         Y_next = f3(X_next, noise)
         X3_sample = np.vstack((X3_sample, X_next))
         Y3_sample = np.vstack((Y3_sample, Y_next))
    
    # Plot samples, surrogate function, noise-free objective and next sampling location
    plt.subplot(n_iter, 2, 2 * i + 1)
    plot_approximation(gpr, X, Y, X_sample, Y_sample, X_next, show_legend=i==0)
    plt.title(f'Iteration {i+1}')

    plt.subplot(n_iter, 2, 2 * i + 2)
    plot_acquisition(X, expected_improvement(X, X_sample, Y_sample, gpr), X_next, show_legend=i==0)
    
    # Add sample to previous samples
    X_sample = np.vstack((X_sample, X_next))
    Y_sample = np.vstack((Y_sample, Y_next))
