#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 07:00:12 2020

@author: isaac.kleisle-murphy
"""

import pickle
import random
import numpy as np

def save_pickle(file, filename):
    
    with open(filename, 'wb') as handle:
        pickle.dump(file, handle)
        
def load_pickle(filename):
    
    with open(filename, 'rb') as handle:
        result = pickle.load(handle)
        
    return result



def shuffle_folds(list_in, n, s = 2020):
    random.seed(s)
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]


def cross_entropy(ypred, ytrue):
    '''
    Calculates categorical cross-entropy
    '''
    return -np.sum(ytrue*np.log(ypred))/(ypred.shape[0])


def widen_array(x):
    '''
    Takes categorical array to one hot. Array must be 0, 1, 2, ..., i.e. not strings
    '''
    mx = max(x)
    return np.array([[1 if j == x[i] else 0 for j in range(mx + 1)] for i in range(len(x))])