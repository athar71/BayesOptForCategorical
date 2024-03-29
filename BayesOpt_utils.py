#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:42:47 2019

@author: athar

Refs : 
#https://github.com/SheffieldML/GPyOpt/blob/master/GPyOpt/acquisitions/EI.py
#https://github.com/SheffieldML/GPyOpt/blob/master/GPyOpt/models/gpmodel.py
#http://krasserm.github.io/2018/03/21/bayesian-optimization/
"""

#For all the functions the default variables should go at the end
import numpy as np
import pandas as pd
from scipy import special
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

        


def Data_organizer(Data):
    
    """
    This function takes The whole data, 
    1- mixes all categorical variables to one categorical varaiable with all
    the possible combinations of the initial categorical variables.
    2- for each levels of the new categorical variable it finds the data points
    in that category and report a (x,y) continuous data array.
    """
    pass



def GPR_model(kernelf, X, Y, grid, noise_var=1e-10, optimizerf='fmin_l_bfgs_b',\
              optimize_restarts=5):
    
    """
    model_type : GPR is the default, rfr (random forest regression or other methods could be added)
    params : parameters of the surrogate model
    data : shoule be a class of continous data with data.x as variables and data.y as features
    cross_validate : If we want to do cross validation for the training.
    grid : The unknown points for the prediction
    
    normalize_y :This parameter should be set to True if the target values’ mean 
    is expected to differ considerable from zero. When enabled, the normalization effectively modifies 
    the GP’s prior based on the data,
    which contradicts the likelihood principle; normalization is thus disabled per default.
    
    fitted_model : The output
    """
            
            
    gpr = GaussianProcessRegressor(kernel= kernelf, alpha=noise_var, optimizer=optimizerf,\
                                   n_restarts_optimizer= optimize_restarts, normalize_y=False).fit(X, Y)
    score = gpr.score(X, Y)
    m,s = gpr.predict(grid, return_std=True, return_cov=False)
    s = np.array(s)
    m = np.array(m)  
    return score, m.reshape(-1,1), s.reshape(-1,1),gpr

    

def _compute_acq( m, s, fmin, epsilon = 0.01):
        
    """
        Computes the Expected Improvement per unit of cost.
        fmin : is the global minimum.
        grid : The grid that we want to fit the function and do the optimization on.
        epsilon : positive value to make the acquisition more explorative. Check the Simon paper. 
        
    """  
    phi, Phi, u = get_quantiles(epsilon, fmin, m, s)
    f_acqu = s * (u * Phi + phi)
    return f_acqu


def get_quantiles(acquisition_par, fmin, m, s):
    '''
    Quantiles of the Gaussian distribution useful to determine the acquisition function values
    :param acquisition_par: parameter of the acquisition function
    :param fmin: current minimum.
    :param m: vector of means.
    :param s: vector of standard deviations.
    '''
    if isinstance(s, np.ndarray):
        s[s<1e-5] = 1e-5
    elif s< 1e-5:
        s = 1e-5
    u = np.array((fmin - m - acquisition_par))/s
    phi = np.exp(-0.5 * u**2) / np.sqrt(2*np.pi)
    Phi = 0.5 * special.erfc(-u / np.sqrt(2))
    return (phi, Phi, u)


def pick_next_sample(max_aqu_vec, nxt_smpl_vec):
    
    max_pos = max_aqu_vec.argmax()
    aqu_max_total = max_aqu_vec.max()
    next_sample_total = nxt_smpl_vec[max_pos]
    
    return next_sample_total, max_pos+1

def pick_next_sample2(aqu_array):
    
    min_pos = aqu_array.argmin()
    minpoint = min_pos[0]
    cat_min = min_pos[1]
    return minpoint, cat_min+1

def get_fmin(Y) :
   
        """
        Returns the location where the posterior mean is takes its minimal value.
        min_pos = min_aqu_vec.argmin()
    aqu_min_total = min_aqu_vec.min()
    next_sample_total = nxt_smpl_vec[min_pos]
    
    return fitted_model.predict(fitted_model.X)[0].min()
        """
        return np.amin(Y)



"""
def _compute_acq_withGradients(self, x):
        
       # Computes the Expected Improvement and its derivative (has a very easy derivative!)
        
        fmin = self.model.get_fmin()
        m, s, dmdx, dsdx = self.model.predict_withGradients(x)
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        f_acqu = s * (u * Phi + phi)
        df_acqu = dsdx * phi - Phi * dmdx
        return f_acqu, df_acqu
        
"""

def plot_approximation(gpr, X, Y, X_sample, Y_sample, num_cat,color, X_next=None, show_legend=False):
    mu, std = gpr.predict(X, return_std=True)
    plt.fill_between(X.ravel(), 
                     mu.ravel() + 1.96 * std, 
                     mu.ravel() - 1.96 * std, 
                     alpha=0.1) 
    
    plt.plot(X, Y, color, lw=1, label='Noise-free actual objective'+" "+str( num_cat))
    plt.plot(X, mu, 'b-', lw=1)
    #plt.plot(X_sample, Y_sample, 'kx', mew=3, label='Noisy samples')
    plt.plot(X_sample, Y_sample, 'kx', mew=3)
    """
    plt.plot(X, Y, 'y--', lw=1, label='Noise-free actual objective'+" "+str( num_cat))
    plt.plot(X, mu, 'b-', lw=1, label='Surrogate function')
    #plt.plot(X_sample, Y_sample, 'kx', mew=3, label='Noisy samples')
    plt.plot(X_sample, Y_sample, 'kx', mew=3, label='Samples')
    """
    if X_next:
        plt.axvline(x=X_next, ls='--', c='k', lw=1)
    if show_legend:
        plt.legend(prop={'size':6})
        
def plot_acquisition(X, Y, X_next, show_legend=False):
    plt.plot(X, Y, 'r-', lw=1, label='Acquisition function')
    plt.axvline(x=X_next, ls='--', c='k', lw=1, label='Next sampling location')
    if show_legend:
        plt.legend()            
        
def plot_all(gpr, X, Y,color, X_sample, Y_sample, num_cat,X2, Y2,color2, X3, Y3,color3,  X_next=None, show_legend=False):
         
    plt.plot(X, Y, color, X2,Y2, color2, X3, Y3, color3, lw=1, label='Noise-free actual objective'+" "+str( num_cat))
    plt.plot(X_sample, Y_sample, 'kx', mew=3)
    
    if X_next:
        plt.axvline(x=X_next, ls='--', c='k', lw=1)
    if show_legend:
        plt.legend(prop={'size':6})
        
def plot_acquisition_all(X, Y,color, X_next,X2, Y2, color2,X3, Y3,color3,   show_legend=False):
    plt.plot(X, Y, color,X2,Y2, color2, X3, Y3, color3)
    plt.axvline(x=X_next, ls='--', c='k', lw=1, label='Next sampling location')
    
    if show_legend:
        plt.legend()   

def plot_functions(X1,Y1, X2,Y2, X3,Y3, color1, color2, color3)             :
    plt.plot(X1, Y1, color1,X2,Y2, color2, X3, Y3, color3)
    plt.legend( ['f1', 'f2', 'f3'])
    