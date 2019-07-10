#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:42:47 2019

@author: athar
"""
import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

def Data_generator():
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


def aqu (fun_type, glb_min, fitted_model, grid ):
    
    
    
    return aqu_val_min, nxt_smpl


#https://github.com/SheffieldML/GPyOpt/blob/master/GPyOpt/acquisitions/EI.py
#https://github.com/SheffieldML/GPyOpt/blob/master/GPyOpt/models/gpmodel.py
    
def _predict(self, X, full_cov, include_likelihood):
        if X.ndim == 1:
            X = X[None,:]
        m, v = self.model.predict(X, full_cov=full_cov, include_likelihood=include_likelihood)
        v = np.clip(v, 1e-10, np.inf)
        return m, v

def predict(self, X, with_noise=True):
        """
        Predictions with the model. Returns posterior means and standard deviations at X. Note that this is different in GPy where the variances are given.
        Parameters:
            X (np.ndarray) - points to run the prediction for.
            with_noise (bool) - whether to add noise to the prediction. Default is True.
        """
        m, v = self._predict(X, False, with_noise)
        # We can take the square root because v is just a diagonal matrix of variances
        return m, np.sqrt(v)

def _compute_acq(fitted_model,grid):
        """
        Computes the Expected Improvement per unit of cost
        """
        m, s = fitted_model.predict(grid)
        fmin = fitted_model.get_fmin()
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        f_acqu = s * (u * Phi + phi)
        return f_acqu

def _compute_acq_withGradients(self, x):
        """
        Computes the Expected Improvement and its derivative (has a very easy derivative!)
        """
        fmin = self.model.get_fmin()
        m, s, dmdx, dsdx = self.model.predict_withGradients(x)
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        f_acqu = s * (u * Phi + phi)
        df_acqu = dsdx * phi - Phi * dmdx
        return f_acqu, df_acqu

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