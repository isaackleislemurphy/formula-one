#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 22:42:11 2020

@author: isaac.kleisle-murphy
"""

_ITEMS = ['lap_times', 'races', 'qualifying', 'results', 'seasons', 
          'constructors', 'drivers', 'circuits', 'status', 'constructor_standings']
SEAS_MIN = 2006
_TRAIN_IDX = -31

import numpy as np
from utils import save_pickle
from ingest import make_races
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from model_cnn import MultiOutputGridSearchCV, build_cnn
from processing import ImageScaler
from model_cnn import plot_validation_cnn




def main(seas_min=SEAS_MIN):


    races = make_races(seas_min)
    
    races_xy = [item.to_xy_3d() for item in races]
    races_images = np.stack([item[0] for item in races_xy], axis=0)
    races_features = np.vstack([item[1] for item in races_xy])
    races_y = np.stack([np.stack(item[2], axis=0) for item in races_xy], 
                       axis = 0)
    train_img, test_img = races_images[:_TRAIN_IDX, ], races_images[_TRAIN_IDX:, ]
    train_feat, test_feat = races_features[:_TRAIN_IDX, ], races_features[_TRAIN_IDX:, ]
    train_y, test_y = races_y[:_TRAIN_IDX, ], races_y[_TRAIN_IDX:, ]
    
    
    image_scale = ImageScaler('absvalue')
    image_scale.fit(train_img, q=.95)
    feat_scale = StandardScaler().fit(train_feat)
    
    
    train_img_sc = image_scale.transform(train_img)
    test_img_sc = image_scale.transform(test_img)
    
    train_feat_sc = feat_scale.transform(train_feat)
    test_feat_sc = feat_scale.transform(test_feat)





    model_param_grd = {'conv_neurons': [[8, 8], [16, 16], [32, 16]],
                       'conv_dropout': [.1, .2], 
                       'pool_sizes': [(2,2), (1, 1)],
                       'merged_dropout': [.1, .175, .25],
                       'output_dropout': [.1],
                       'feature_neurons': [25, 50, 100], 
                       'output_neurons': [[10], [20, 10], [50, 50]],
                       'merged_neurons': [[15, 15, 15, 12], [50, 50], [100, 100], [400, 250, 125]],
                       'lr': [.001]}
    fit_param_grd = {'epochs': [150],
                     'batch_size': [32],
                     'verbose': [False]}
    
    
    
    
    



    grid_search = MultiOutputGridSearchCV(model_param_grd, fit_param_grd, build_cnn)
    grid_search.fit(X=[train_img_sc, train_feat_sc], y=train_y)
    
    
    
    print("\n Optimal Epochs: ")
    print(np.argmin(grid_search.get_best_tune_results().weigh_scores()[0]) + 1)
    
    print("\n Optimal Scoring")
    print(np.min(grid_search.get_best_tune_results().weigh_scores()[0]))
    
    print("\nOptimal Params: ")
    print(grid_search.get_best_tune_params())

    plot_validation_cnn(grid_search)
    
    grid_search.get_all_tune_results()

if __name__=='__main__':
    main()
    