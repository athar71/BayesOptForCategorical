#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:42:47 2019

@author: athar

Refs : 
#https://github.com/SheffieldML/GPyOpt/blob/master/GPyOpt/acquisitions/EI.py
#https://github.com/SheffieldML/GPyOpt/blob/master/GPyOpt/models/gpmodel.py
"""
import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

def Data_generator():
    """
    num_cat 
    
    """
    pass


def Data_organizer(Data):
    
    
    #return x_cnts, x_cat, y, num_cat, num_cnts_var
    return Data_organazied, num_cat, num_cnts_var



def Surrogate_model(model_type = "gpr", params ,data, cross_validate= False, grid):
    
    """
    model_type : GPR is the default, rfr (random forest regression or other methods could be added)
    params : parameters of the surrogate model
    data : shoule be a class of continous data with data.x as variables and data.y as features
    cross_validate : If we want to do cross validation for the training.
    grid : The unknown points for the prediction
    
    fitted_model : The output
    """
    
    if model_type == "gpr":
       
        gpr = GaussianProcessRegressor(kernel=params.kernel ,random_state=0).fit(data.x, data.y)
        fitted_model.score = gpr.score(data.x, data.y)
        fitted_model.predict =  predict(grid, return_std=False, return_cov=False)
        fitted_model. params = gpr.get_params
   
    return fitted_model 


    

def _compute_acq(fitted_model,grid, fmin, epsilon):
        """
        Computes the Expected Improvement per unit of cost.
        fmin : is the global minimum.
        grid : The grid that we want to fit the function and do the optimization on.
        epsilon : positive value to make the acquisition more explorative. Check the Simon paper. 
        
        """
        m, s = fitted_model.predict(grid)
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
        s[s<1e-10] = 1e-10
    elif s< 1e-10:
        s = 1e-10
    u = (fmin - m - acquisition_par)/s
    phi = np.exp(-0.5 * u**2) / np.sqrt(2*np.pi)
    Phi = 0.5 * erfc(-u / np.sqrt(2))
    return (phi, Phi, u)


def pick_next_sample(min_aqu_vec, nxt_smpl_vec):
    
    min_pos = min_aqu_vec.argmin()
    aqu_min_total = min_aqu_vec.min()
    next_sample_total = nxt_smpl_vec[min_pos]
    
    return next_sample_total, aqu_min_total

def get_fmin(fitted_model, X) :
   
        """
        Returns the location where the posterior mean is takes its minimal value.
        """
    min_pos = min_aqu_vec.argmin()
    aqu_min_total = min_aqu_vec.min()
    next_sample_total = nxt_smpl_vec[min_pos]
    
    return fitted_model.predict(fitted_model.X)[0].min()



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