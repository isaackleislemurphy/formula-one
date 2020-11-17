#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 11:51:54 2020

@author: isaac.kleisle-murphy
"""



import numpy as np
import random 
import copy
from utils import shuffle_folds, cross_entropy, widen_array
from itertools import product
from itertools import chain

from sklearn.linear_model import LogisticRegression




class MNCV:
    
    def __init__(self, nfolds=10, seed=2020):
        
        self.nfolds=nfolds
        self.history = []
        self.seed=2020
        
        
    def fit(self, X, y, model, **kwargs):
        
        
        fold_indices = shuffle_folds([i for i in range(y.shape[0])], self.nfolds)
        self.folds = fold_indices
        
        for ii in range(self.nfolds):
            
            print(f"    fold: {ii+1}/{self.nfolds}")
            model_ = copy.deepcopy(model)
            
            #assign folds
            val_idx = fold_indices[ii]
            fit_idx = []
            for jj in range(len(fold_indices)):
                 if jj != ii:
                    fit_idx += fold_indices[jj]
                  
            x_tr, x_val = np.vstack(X[fit_idx,]), np.vstack(X[val_idx,])
            y_tr, y_val = np.vstack(y[fit_idx,]), np.vstack(y[val_idx,])
            y_tr_sparse = y_tr.dot(range(0, y.shape[2]))
            
            model_.fit(x_tr, y_tr_sparse)
            
            y_hat = model_.predict_proba(x_val)
            val_loss = cross_entropy(y_hat, y_val)
            self.history.append(val_loss)
        
        
        
        


class MNGridSearchCV:
    
    def __init__(self, param_grd, model, verbose=True, **kwargs):
        
        self.param_grd = param_grd
        self.model = copy.deepcopy(model)
        self.verbose = verbose
        
    def fit(self, X, y, k=10, s=2020, **kwargs):
        
        self.tune_params = []
        self.tune_results = []
        
        param_list = [dict(zip(self.param_grd, v)) for v in product(*self.param_grd.values())]
        
        ctr = 1
        n_tunes = len(param_list)
        for params in param_list:
            if self.verbose:
                print(f'...Fitting tune {ctr}/{n_tunes}...\n')
            mncv = MNCV(nfolds=k, seed=s)
            mncv.fit(X, y, self.model, **params)
            
            self.tune_params.append(params)
            self.tune_results.append(mncv)
            
            ctr += 1
                
                
    def get_best_tune_results(self, smaller_is_better=True):
    
        arg_fun = np.min if smaller_is_better else np.max
        scoring = [np.mean(item.history) for item in self.tune_results]
        return arg_fun(scoring)


    def get_best_tune_params(self, smaller_is_better=True):
        
        arg_fun = np.argmin if smaller_is_better else np.argmax
        scoring = [np.mean(item.history) for item in self.tune_results]
        return self.tune_params[arg_fun(scoring)]
    
    def get_all_results(self):
        
        return list(zip(self.tune_params, [np.mean(item.history) for item in self.tune_results]))