#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 17:05:12 2020

@author: isaac.kleisle-murphy
"""

import numpy as np
import random 


from itertools import product
from itertools import chain

from sklearn.preprocessing import MinMaxScaler

import keras
import pydotplus
import pydot


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
import matplotlib.pyplot as plt





def build_cnn(image_x, 
                features_x, 
                y, 
                conv_neurons = [32, 16],
                pool_sizes = [(1, 1), (2,2)],
                kernel_sizes = [4, 4],
                flatten_neurons = 25,
                feature_neurons = 25,
                merged_neurons = [500, 500],
                output_neurons = [100],
                conv_dropout = .1,
                merged_dropout = .1,
                output_dropout = .1,
                lr = .001,
                **kwargs):


    deltas = Input(shape=(image_x.shape[1], image_x.shape[2], image_x.shape[3]), name = 'conv_features')
    
    conv1 = Conv2D(conv_neurons[0], kernel_size=kernel_sizes[0], activation='relu', name = 'conv1')(deltas)
    conv1 = BatchNormalization(axis=-1)(conv1)
    if conv_dropout > 0:
        conv1 = Dropout(conv_dropout)(conv1)
    pool1 = MaxPooling2D(pool_size=pool_sizes[0], name='pool1')(conv1)
    
    conv2 = Conv2D(conv_neurons[1], kernel_size=kernel_sizes[1], activation='relu', name = 'conv2')(pool1)
    conv2 = BatchNormalization(axis=-1)(conv2)
    if conv_dropout > 0:
        conv2 = Dropout(conv_dropout)(conv2)
    pool2 = MaxPooling2D(pool_size=pool_sizes[1], name = 'pool2')(conv2)
    
    
    # flatten convolutional layers
    fconv = Flatten(name = 'fconv')(pool2)
    fconv_out = Dense(flatten_neurons, activation='relu', name = 'fconv_out')(fconv)
    
    
    additional_features = Input(shape = (features_x.shape[1], ), name = 'additional_features')
    hidden_f1 = Dense(feature_neurons, activation = 'relu')(additional_features)
    
    
    combo_layer = Concatenate(name='concatenation')([hidden_f1, fconv_out])
    
    combo_layer_2 = Dense(merged_neurons[0], activation='relu', name = 'combined_layer_2')(combo_layer)
    if merged_dropout > 0:
        combo_layer_2 = Dropout(merged_dropout)(combo_layer_2)
        
    for i in range(1, len(merged_neurons)):
        combo_layer_2 = Dense(merged_neurons[i], activation='relu', name = f'combined_layer_{2+i}')(combo_layer_2)
        if merged_dropout > 0:
            combo_layer_2 = Dropout(merged_dropout)(combo_layer_2)
        
    outputs=[]
    for pos in range(y.shape[1]):
        hidden_pos = Dense(output_neurons[0], activation='relu', name = f'hidden_p{pos+1}')(combo_layer_2)
        output_pos = Dense(y.shape[2], activation='softmax', name = f'output_p{pos+1}')(hidden_pos)
        outputs.append(output_pos)
    

    model = Model(inputs=[deltas, additional_features], 
                  outputs=outputs)
    # summarize layers
    # print(model.summary())
    
    
    
    model.compile(optimizer=Adam(learning_rate=lr), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy', 'categorical_crossentropy'])
    
    return model


class MultiOutputCV:
    
    def __init__(self, nfolds=5, seed=2020):
        
        self.nfolds=nfolds
        self.history = []
        self.seed=2020
        
    def fit(self, X, y, model, **kwargs):
        
        fold_indices = shuffle_folds([i for i in range(y.shape[0])], self.nfolds)
        init_weights=model.get_weights()
        self.folds = fold_indices
        
        for ii in range(self.nfolds):
            
            print(f"    fold: {ii+1}/{self.nfolds}")
            
            #assign folds
            val_idx = fold_indices[ii]
            fit_idx = []
            for jj in range(len(fold_indices)):
                 if jj != ii:
                    fit_idx += fold_indices[jj]
            
            
            x_train, x_val = [item[fit_idx, ] for item in X], [item[val_idx, ] for item in X]
            y_train, y_val = y[fit_idx, ], y[val_idx]
            
            y_train_ = [y_train[:, j, ] for j in range(y_train.shape[1])]
            y_val_ = [y_val[:, j, ] for j in range(y_val.shape[1])]
            #y_train, y_val = [item[fit_idx, ] for item in y], [item[val_idx, ] for item in y]
            
            model.set_weights(init_weights)
            history = model.fit(x_train, y_train_,
                                validation_data = (x_val, y_val_),
                                **kwargs)
            self.history.append(history.history)
            
            
    def weigh_scores(self, 
                     outputs=list(map(lambda x: 'p'+ str(x), range(1, 25))), 
                     weights = np.ones(24), 
                     metric='categorical_crossentropy'):
    
        parallel_train_scores = np.vstack([
            np.vstack([item[f'output_{outputs[i]}_{metric}'] 
                       for item in self.history]).mean(axis=0) * weights[i]/np.sum(weights)
            for i in range(len(weights))
            ]).sum(axis=0)
        
        
        parallel_val_scores = np.vstack([
            np.vstack([item[f'val_output_{outputs[i]}_{metric}'] 
                       for item in self.history]).mean(axis=0) * weights[i]/np.sum(weights)
            for i in range(len(weights))
            ]).sum(axis=0)
        
        return parallel_val_scores, parallel_train_scores


class MultiOutputGridSearchCV:
    
    def __init__(self, model_param_grd, fit_param_grd, constructor, verbose=True, **kwargs):
        
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
                
                model = self.constructor(X[0], X[1], y, **mod_params)
                multi_cv = MultiOutputCV(nfolds=k, seed=s)
                multi_cv.fit(X, y, model, **fit_params)
                
                self.tune_params.append((fit_params, mod_params))
                self.tune_results.append(multi_cv)
                
                ctr += 1
                
                
    def get_all_tune_results(self, smaller_is_better=True):
        
        get_fun = np.min if smaller_is_better else np.max
        scoring = [get_fun(item.weigh_scores()[0]) for item in self.tune_results]
        print('\n' + "-"*10 + ' All Scoring ' +  "-"*10)
        for ii in range(len(scoring)):
            print('\t Tune setting: ' + str(self.tune_params[ii]))
            print('\t Score: ' + str(scoring[ii]))
        
                
                
    def get_best_tune_results(self, smaller_is_better=True):
    
        get_fun = np.min if smaller_is_better else np.max
        arg_fun = np.argmin if smaller_is_better else np.argmax
        
        scoring = [get_fun(item.weigh_scores()[0]) for item in self.tune_results]
        return self.tune_results[arg_fun(scoring)]


    def get_best_tune_params(self, smaller_is_better=True):
    
        get_fun = np.min if smaller_is_better else np.max
        arg_fun = np.argmin if smaller_is_better else np.argmax
        
        scoring = [get_fun(item.weigh_scores()[0]) for item in self.tune_results]
        return self.tune_params[arg_fun(scoring)]

                
                
                
                
def plot_validation_cnn(grid_search, filename='cnn_gridsearch.png'):
    
    best_tune = np.argmin([np.min(item.weigh_scores()[0]) for item in grid_search.tune_results])

    opt_ep=np.argmin(grid_search.get_best_tune_results().weigh_scores()[0]) + 1
    
    for ii in range(len(grid_search.tune_results[best_tune].history)):
        tune = grid_search.tune_results[0].history[ii]
        
        val_keys = [f'val_output_p{j}_categorical_crossentropy' for j in range(1, 25)]
        tr_keys = [f'output_p{j}_categorical_crossentropy' for j in range(1, 25)]
        
        val_losses = np.vstack(tune[k] for k in val_keys).mean(axis=0)
        tr_losses = np.vstack(tune[k] for k in tr_keys).mean(axis=0)
        
        plt.plot(val_losses, color = 'red', alpha = .255)
        plt.plot(tr_losses, color = 'blue', alpha = .255)
    
    
    
    plt.plot(grid_search.get_best_tune_results().weigh_scores()[0], color = 'red', label = 'Validation')
    plt.plot(grid_search.get_best_tune_results().weigh_scores()[1], color = 'blue', label = 'Training')
    
    plt.vlines([opt_ep], 
               ymin = 1,
               ymax = 2,
               color = 'black',
               alpha=.55,
               linestyle = 'dashed',
               label = f'Optimal Epochs: {int(opt_ep)}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Categorical Crossentropy (Log-Loss)')
    plt.ylim(1, )
    plt.legend()
    plt.savefig(filename)
    plt.show()
        
        



