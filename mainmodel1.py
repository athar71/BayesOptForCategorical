#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 11:31:10 2019
@author: athar
This code is modling bayseian optimization for the cases with categorical variables
"""
"""
Assumption  : Minimization
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
#from bayesian_optimization_util import plot_approximation, plot_acquisition
import BayesOpt_utils as BO
import data_generator as dgen
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
kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)

# Initialize samples 

noise = 0
bounds = np.array([[-1.0, 2.0]])
epsilon = 0 #exploration coefficient

X_init = np.array([[-0.8], [1.3]])
Y1_init = dgen.f1(X_init)
Y2_init = dgen.f2(X_init)
Y3_init = dgen.f3(X_init)

X1_sample = X_init
Y1_sample = Y1_init
X2_sample = X_init
Y2_sample = Y2_init
X3_sample = X_init
Y3_sample = Y3_init

grid1 = np.arange(bounds[:, 0], bounds[:, 1], 0.01).reshape(-1, 1)
grid2 = np.arange(bounds[:, 0], bounds[:, 1], 0.01).reshape(-1, 1)
grid3 = np.arange(bounds[:, 0], bounds[:, 1], 0.01).reshape(-1, 1)

Y1 = dgen.f1(grid1)
Y2 = dgen.f2(grid2)
Y3 = dgen.f3(grid3)


# Number of iterations

n_iter = 4
plt.figure(num =1, figsize=(12, n_iter * 3))
plt.subplots_adjust(hspace=0.5)
#plt.subplots_adjust(left=0.9, bottom=0.9, right=0.9, top=0.9, wspace=0.5, hspace=0.5)


plt.figure(num = 2, figsize=(12, n_iter * 3))
plt.subplots_adjust(hspace=0.5)
color1 = 'r-'
color2 = 'g-'
color3 = 'm-'
 

for i in range(n_iter):
    # Update Gaussian process with existing samples
    
    
    score1, m1, s1, gpr1 = BO.GPR_model(kernel, X1_sample, Y1_sample, grid1, noise_var=1e-10, optimizerf='fmin_l_bfgs_b',\
              optimize_restarts=5)
    
    score2, m2, s2, gpr2 = BO.GPR_model(kernel, X2_sample, Y2_sample, grid2, noise_var=1e-10, optimizerf='fmin_l_bfgs_b',\
              optimize_restarts=5)
    
    score3, m3, s3, gpr3 = BO.GPR_model(kernel, X3_sample, Y3_sample, grid3, noise_var=1e-10, optimizerf='fmin_l_bfgs_b',\
              optimize_restarts=5)
    
    #fmin =BO.get_fmin([Y1_sample,Y2_sample,Y3_sample])
    fmin = np.min([np.amin(Y1_sample), np.amin(Y2_sample), np.amin(Y3_sample)])
    # Obtain next sampling point from the acquisition function (expected_improvement)
    
    acq1 = BO._compute_acq( m1, s1, fmin, epsilon)
    acq2 = BO._compute_acq( m2, s2, fmin, epsilon)
    acq3 = BO._compute_acq( m3, s3, fmin, epsilon)
    
    max_aqu_vec = np.array([np.max(acq1),np.max(acq2),np.max(acq3)])
    nxt_smpl_vec = np.array([grid1[np.argmax(acq1)],grid2[np.argmax(acq2)],grid3[np.argmax(acq3)]])
    
    
    X_next, cat_max = BO.pick_next_sample(max_aqu_vec, nxt_smpl_vec)
    
    
    # Obtain next noisy sample from the objective function
    if cat_max==1:
         Y_next = dgen.f1(X_next, noise)
         X1_sample = np.vstack((X1_sample, X_next))
         Y1_sample = np.vstack((Y1_sample, Y_next))
         
         plt.figure(1)
         plt.subplot(n_iter, 2, 2 * i + 1)
         BO.plot_approximation(gpr1, grid1, Y1, X1_sample, Y1_sample,cat_max,color1, X_next, show_legend= i%1==0)
         #plt.title(f 'Iteration {i+1}', fontsize=6)
         plt.title('Iteration %d' %(i + 1), fontsize=6)
         plt.subplot(n_iter, 2, 2 * i + 2)
         BO.plot_acquisition(grid1, acq1, X_next, show_legend=i==0)
    
        
         
         plt.figure(2)
         plt.subplot(n_iter, 2, 2 * i + 1)
         BO.plot_all(gpr1, grid1, Y1, color1, X1_sample, Y1_sample,cat_max, grid2, Y2, color2, grid3, Y3, color3, X_next, show_legend= i%1==0)
         #plt.title(f'Iteration {i+1}', fontsize=6)
         plt.title('Iteration %d' % (i + 1), fontsize=6)
         plt.subplot(n_iter, 2, 2 * i + 2)
         BO.plot_acquisition_all(grid1, acq1,color1, X_next,grid2, acq2,color2, grid3, acq3, color3, show_legend=i==0)
    
         # Add sample to previous samples
         X1_sample = np.vstack((X1_sample, X_next))
         Y1_sample = np.vstack((Y1_sample, Y_next))
         
         
    elif cat_max==2:
         Y_next = dgen.f2(X_next, noise)
         X2_sample = np.vstack((X2_sample, X_next))
         Y2_sample = np.vstack((Y2_sample, Y_next))
         
         plt.figure(1)
         plt.subplot(n_iter, 2, 2 * i + 1)
         BO.plot_approximation(gpr2, grid2, Y2, X2_sample, Y2_sample, cat_max, color2, X_next, show_legend=i%1==0)
         #plt.title(f'Iteration {i+1}', fontsize=6)
         plt.title('Iteration %d' %(i + 1), fontsize=6)
         plt.subplot(n_iter, 2, 2 * i + 2)
         BO.plot_acquisition(grid2,acq2, X_next, show_legend=i==0)
         
         
         
         plt.figure(2)
         plt.subplot(n_iter, 2, 2 * i + 1)
         BO.plot_all(gpr2, grid2, Y2, color2, X2_sample, Y2_sample,cat_max, grid1, Y1, color1, grid3, Y3, color3, X_next, show_legend= i%1==0)
         #plt.title(f'Iteration {i+1}', fontsize=6)
         plt.title('Iteration %d' %(i + 1), fontsize=6)
         plt.subplot(n_iter, 2, 2 * i + 2)
         BO.plot_acquisition_all(grid2, acq2,color2, X_next,grid1, acq1,color1, grid3, acq3,color3,  show_legend=i==0)
    
         # Add sample to previous samples
         X2_sample = np.vstack((X2_sample, X_next))
         Y2_sample = np.vstack((Y2_sample, Y_next))
    
    else :
         Y_next = dgen.f3(X_next, noise)
         X3_sample = np.vstack((X3_sample, X_next))
         Y3_sample = np.vstack((Y3_sample, Y_next))
         
         plt.figure(1)
         plt.subplot(n_iter, 2, 2 * i + 1)
         BO.plot_approximation(gpr3, grid3, Y3, X3_sample, Y3_sample, cat_max, color3, X_next, show_legend=i%1==0)
         #plt.title(f'Iteration {i+1}', fontsize=6)
         plt.title('Iteration %d' %(i + 1), fontsize=6)
         plt.subplot(n_iter, 2, 2 * i + 2)
         BO.plot_acquisition(grid3,acq3, X_next, show_legend=i==0)
         
         plt.figure(2)
         plt.subplot(n_iter, 2, 2 * i + 1)
         BO.plot_all(gpr3, grid3, Y3, color3, X3_sample, Y3_sample,cat_max, grid1, Y1, color1, grid2, Y2, color2, X_next, show_legend= i%1==0)
         #plt.title(f'Iteration {i+1}', fontsize=6)
         plt.title('Iteration %d' %(i + 1), fontsize=6)

         plt.subplot(n_iter, 2, 2 * i + 2)
         BO.plot_acquisition_all(grid3, acq3,color3, X_next,grid1, acq1,color1, grid2, acq2, color2, show_legend=i==0)
    
         
         # Add sample to previous samples
         X3_sample = np.vstack((X3_sample, X_next))
         Y3_sample = np.vstack((Y3_sample, Y_next))