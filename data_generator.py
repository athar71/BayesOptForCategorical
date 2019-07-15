#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 16:42:28 2019

@author: athar
"""

import numpy as np
import pandas as pd



from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

def Checking_data_gen(num_cats, bounds ):
    