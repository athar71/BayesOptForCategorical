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


def pick_next_sample(min_aqu_vec, nxt_smpl_vec):
    
    min_pos = min_aqu_vec.argmin()
    aqu_min_total = min_aqu_vec.min()
    next_sample_total = nxt_smpl_vec[min_pos]
    
    return next_sample_total, aqu_min_total