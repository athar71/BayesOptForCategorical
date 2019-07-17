#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 16:42:28 2019

@author: athar
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

def Checking_data_gen( X1,X2,X3 , noise=0.02):
    

    """
    bounds = np.array([[-1.0, 2.0]])
    noise = 0.2
    num_cats = 3
    """
    f1 = -np.sin(3*X1) - X1**2 + 0.7*X1 + noise * np.random.randn(*X1.shape)
    f2 = -np.sin(5*X2) - X2**2 + X2 + noise * np.random.randn(*X.shape) 
    f3 = -np.sin(3*X3) - X3**2 + 2*X3 + noise * np.random.randn(*X.shape)
    
    return f1,f2,f3

def f1(X1,  noise=0.02):
    
    return -np.sin(3*X1) - X1**2 + 0.7*X1 + noise * np.random.randn(*X1.shape)

def f2(X2,  noise=0.02):
    
    return -np.sin(5*X2) - X2**2 + X2 + noise * np.random.randn(*X2.shape) 

def f3(X3,  noise=0.02):
    
    return -np.sin(3*X3) - X3**2 + 2*X3 + noise * np.random.randn(*X3.shape)
def Data_generator():
    """
    num_cat 
    
    """
    pass

