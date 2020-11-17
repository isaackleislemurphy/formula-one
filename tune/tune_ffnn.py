#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 22:42:11 2020

@author: isaac.kleisle-murphy
"""

import warnings
warnings.filterwarnings('ignore')

SEAS_MIN=2006
_TRAIN_IDX = -31

import numpy as np
from utils import save_pickle
from ingest import make_races
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from model_ffnn import build_ffnn, NNGridSearchCV, plot_validation_ffnn





def main(seas_min=SEAS_MIN):


    races = make_races(seas_min)
    races_xy = [item.to_xy_1d() for item in races]
    x_all = np.stack([item[0] for item in races_xy], axis=0)
    y_all = races_y = np.stack([np.stack(item[1], axis=0) for item in races_xy], axis=0)
    
    x_train, x_test = x_all[:_TRAIN_IDX, ], x_all[_TRAIN_IDX:, ]
    y_train, y_test = y_all[:_TRAIN_IDX, ], y_all[_TRAIN_IDX:, ]
    
    scaler = StandardScaler().fit(np.vstack(x_train))
    
    x_train_sc = np.stack([scaler.transform(item) for item in x_train], axis=0)
    x_test_sc = np.stack([scaler.transform(item) for item in x_test], axis=0)
    
    
    
    
    model_param_grd = {'hidden_layers': [[25, 25, 25, 25], 
                                         [15, 15, 15, 15], 
                                         [25, 25, 25],
                                         [25, 25],
                                         [50, 50], 
                                         [100, 100], 
                                         [500, 500], 
                                         [400, 250, 125]],
                       'activation': ['sigmoid', 'relu'],
                       'd': [.1, .2],
                       'lr': [.001]}
    fit_param_grd = {'epochs': [125],
                     'batch_size': [16, 64, 256],
                     'verbose': [False]}
    
    
    grid_search = NNGridSearchCV(model_param_grd, fit_param_grd, build_ffnn)
    grid_search.fit(x_train_sc, y_train, k = 5)
    
    
    
    
    print("\n\n" + "-"*10 + ' Full Results: '+ "-"*10)
    for ii in range(len(grid_search.tune_params)):
        print("\nParams:")
        print(grid_search.tune_params[ii])
        print("Scoring:")
        print(grid_search.get_tune_results()[ii])
        
        
    
    print("\n\n" + "*"*10 + ' Best Result: '+ "*"*10)
    print(np.min(grid_search.get_best_tune_results().get_scores()[0]))
    
    print("\n\n" + "*"*10 + ' Corresponding Params: '+ "*"*10)
    print(grid_search.get_best_tune_params())
    
    print("\n\n" + "*"*10 + ' Opt Epochs: '+ "*"*10)
    print(grid_search.get_opt_epochs())
    
    plot_validation_ffnn(grid_search)
    


if __name__=='__main__':
    main()
    