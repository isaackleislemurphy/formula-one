#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 17:05:12 2020

@author: isaac.kleisle-murphy
"""

import numpy as np


from itertools import product

import matplotlib.pyplot as plt

import keras
import pydotplus
import pydot
from keras.utils.vis_utils import model_to_dot
keras.utils.vis_utils.pydot = pydot

from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv3D
from keras.layers.convolutional import MaxPooling2D, MaxPooling3D
from keras.layers.core import Activation, Dropout, Lambda, Dense
from keras.layers import Flatten, Concatenate
from keras.layers import Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from utils import shuffle_folds





def build_ffnn(X,
               y, 
               hidden_layers = [100, 100],
               d = .1,
               activation='sigmoid',
               lr = .001,
                **kwargs):

    
    input_lyr = Input(shape=(X.shape[-1], ), name='input')
    
    for lyr in range(len(hidden_layers)):
        if lyr==0:
            h0 = Dense(hidden_layers[lyr], activation=activation, name=f'h{lyr}')(input_lyr)
        else:
            h0 = Dense(hidden_layers[lyr], activation=activation, name=f'h{lyr}')(h0)
        h0 = BatchNormalization(axis=-1)(h0)
        if d:
            h0 = Dropout(d)(h0)
    
    output_lyr = Dense(y.shape[-1], activation='softmax', name='output')(h0)
    

    model = Model(inputs=[input_lyr], 
                  outputs=[output_lyr])
    # summarize layers
    # print(model.summary())
    
    
    
    model.compile(optimizer=Adam(learning_rate=lr), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy', 'categorical_crossentropy'])
    
    return model


class NNCV:
    
    def __init__(self, nfolds=10, seed=2020):
        
        self.nfolds=nfolds
        self.history = []
        self.seed=2020
        
        
    def fit(self, X, y, model, **kwargs):
        
        
        fold_indices = shuffle_folds([i for i in range(y.shape[0])], self.nfolds)
        self.folds = fold_indices
        init_weights=model.get_weights()
        
        for ii in range(self.nfolds):
            
            print(f"    fold: {ii+1}/{self.nfolds}")
            
            #assign folds
            val_idx = fold_indices[ii]
            fit_idx = []
            for jj in range(len(fold_indices)):
                 if jj != ii:
                    fit_idx += fold_indices[jj]
                  
            x_tr, x_val = np.vstack(X[fit_idx,]), np.vstack(X[val_idx,])
            y_tr, y_val = np.vstack(y[fit_idx,]), np.vstack(y[val_idx,])
            
            model.set_weights(init_weights)
            history = model.fit(x_tr, y_tr,
                                validation_data = (x_val, y_val),
                                **kwargs)
            self.history.append(history.history)
            
            
    def get_scores(self, metric='categorical_crossentropy'):
        
        train_scores = np.vstack([item[f'{metric}'] for item in self.history]).mean(axis=0)
        val_scores = np.vstack([item[f'val_{metric}'] for item in self.history]).mean(axis=0)
        return val_scores, train_scores
    
    def get_best_val_score(self, metric='categorical_crossentropy'):
        
        val_scores = np.vstack([item[f'val_{metric}'] for item in self.history]).mean(axis=0)
        return np.min(val_scores)
    
    
    


class NNGridSearchCV:
    
    def __init__(self, model_param_grd, fit_param_grd, constructor, verbose=True,  **kwargs):
        
        self.model_param_grd = model_param_grd
        self.fit_param_grd = fit_param_grd
        self.constructor = constructor
        self.verbose = verbose
        
    def fit(self, X, y, k=5, s=2020, **kwargs):
        
        self.tune_params = []
        self.tune_results = []
        
        fit_param_list = [dict(zip(self.fit_param_grd, v)) 
                          for v in product(*self.fit_param_grd.values())]
        mod_param_list = [dict(zip(self.model_param_grd, v)) 
                          for v in product(*self.model_param_grd.values())]
        
        ctr = 1
        n_tunes = len(fit_param_list)*len(mod_param_list)
        for fit_params in fit_param_list:
            for mod_params in mod_param_list:
                
                if self.verbose:
                    print(f'...Fitting tune {ctr}/{n_tunes}...\n')
                
                model = self.constructor(X, y, **mod_params)
                multi_cv = NNCV(nfolds=k)
                multi_cv.fit(X, y, model, **fit_params)
                
                self.tune_params.append((fit_params, mod_params))
                self.tune_results.append(multi_cv)
                
                ctr += 1
                
                
    def get_tune_results(self, smaller_is_better=True):
        
        get_fun = np.min if smaller_is_better else np.max
        
        return [get_fun(item.get_scores()[0]) for item in self.tune_results]
                
                
    def get_best_tune_results(self, smaller_is_better=True):
    
        get_fun = np.min if smaller_is_better else np.max
        arg_fun = np.argmin if smaller_is_better else np.argmax
        
        scoring = [get_fun(item.get_scores()[0]) for item in self.tune_results]
        return self.tune_results[arg_fun(scoring)]
    
    
    def get_opt_epochs(self, smaller_is_better=True, idx = None):
        
        if not idx:
            get_fun = np.min if smaller_is_better else np.max
            arg_fun = np.argmin if smaller_is_better else np.argmax
            scoring = [get_fun(item.get_scores()[0]) for item in self.tune_results]
            idx = arg_fun(scoring)
        opt_epochs = arg_fun(self.tune_results[idx].get_scores()[0]) + 1
        return opt_epochs
        


    def get_best_tune_params(self, smaller_is_better=True):
    
        get_fun = np.min if smaller_is_better else np.max
        arg_fun = np.argmin if smaller_is_better else np.argmax
        
        scoring = [get_fun(item.get_scores()[0]) for item in self.tune_results]
        return self.tune_params[arg_fun(scoring)]

                
                
def plot_validation_ffnn(grid_search, filename='ffnn_gridsearch.png'):

    best_tune = np.argmin(grid_search.get_tune_results())
    
    for ii in range(len(grid_search.tune_results[best_tune].history)):
        tune = grid_search.tune_results[0].history[ii]
        plt.plot(tune['val_categorical_crossentropy'], color = 'red', alpha = .25)
        plt.plot(tune['categorical_crossentropy'], color = 'blue', alpha = .25)
    
    
    
    plt.plot(grid_search.get_best_tune_results().get_scores()[0], color = 'red', label = 'Validation')
    plt.plot(grid_search.get_best_tune_results().get_scores()[1], color = 'blue', label = 'Training')
    plt.vlines([grid_search.get_opt_epochs()], 
               ymin = 1,
               ymax = 2,
               color = 'black',
               alpha=.55,
               linestyle = 'dashed',
               label = f'Optimal Epochs: {int(grid_search.get_opt_epochs())}')
    plt.xlabel('Epoch')
    plt.ylabel('Categorical Crossentropy (Log-Loss)')
    plt.ylim(1, )
    plt.legend()
    plt.savefig(filename)
    plt.show()                
        
        
        



