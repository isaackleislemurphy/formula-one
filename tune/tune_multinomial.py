#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 22:42:11 2020

@author: isaac.kleisle-murphy
"""
import warnings
warnings.filterwarnings('ignore')

_ITEMS = ['lap_times', 'races', 'qualifying', 'results', 'seasons', 
          'constructors', 'drivers', 'circuits', 'status', 'constructor_standings']
_SEAS_MIN = 2006
_TRAIN_IDX = -31

import numpy as np
import pandas as pd
from utils import save_pickle
from ingest import make_races
from sklearn.preprocessing import StandardScaler

from model_mn import MNGridSearchCV
from sklearn.linear_model import LogisticRegression





def main(seas_min=_SEAS_MIN, interact=True):

    races = make_races(seas_min)
    races_xy = [item.to_xy_1d() for item in races]
    x_all = np.stack([item[0] for item in races_xy], axis=0)
    y_all = races_y = np.stack([np.stack(item[1], axis=0) for item in races_xy], axis=0)
    
    x_train, x_test = x_all[:_TRAIN_IDX, ], x_all[_TRAIN_IDX:, ]
    y_train, y_test = y_all[:_TRAIN_IDX, ], y_all[_TRAIN_IDX:, ]
    
    scaler = StandardScaler().fit(np.vstack(x_train))
    
    x_train_sc = np.stack([scaler.transform(item) for item in x_train], axis=0)
    x_test_sc = np.stack([scaler.transform(item) for item in x_test], axis=0)
    
    
    
    
    clf = LogisticRegression(penalty='l2', solver='saga', multi_class='multinomial', max_iter=1e8)
    param_grd = {'C': [.05, .025, .01, .005, .001, .0005, .0001]}
    
    grid_search = MNGridSearchCV(param_grd, clf)
    grid_search.fit(x_train_sc, y_train, k=5)
    
    
    print("*"*10 + " Optimal Tune " + "*"*10)
    print(grid_search.get_best_tune_params())
    
    print("-"*10 + " Scoring " + "-"*10)
    for item in grid_search.get_all_results():
        print(item)
        
    print("\n\n" + "*"*10 + " Optimal Tune " + "*"*10)
    print(grid_search.get_best_tune_params())
    


if __name__=='__main__':
    main()
    