#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 22:42:11 2020

@author: isaac.kleisle-murphy
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from utils import save_pickle, cross_entropy
from ingest import make_races
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from processing import read_data, ImageScaler
import matplotlib.pyplot as plt
from model_cnn import build_cnn


SEAS_MIN=2006
_TRAIN_IDX = -31

_ITEMS = ['lap_times', 'races', 'qualifying', 'results', 'seasons', 
          'constructors', 'drivers', 'circuits', 'status', 'constructor_standings']

racedata = {item: read_data('./data/{item}.csv'.format(item=item)) 
            for item in _ITEMS}

MODEL_PARAMS = {'conv_neurons': [8, 8],
                'conv_dropout': .1, 
                'pool_sizes': (2,2),
                'merged_dropout': .25,
                'output_dropout': .15,
                'feature_neurons': 25, 
                'output_neurons': [10],
                'merged_neurons': [15, 15, 15, 12],
                'lr': .001}
FIT_PARAMS = {'epochs': 87,
              'batch_size': 32,
              'verbose': False}




def train_prerace(race_idx, races, seas_min=SEAS_MIN, feature_scaler=MinMaxScaler()):
    
    races_xy = [item.to_xy_3d() for item in races]
    races_images = np.stack([item[0] for item in races_xy], axis=0)
    races_features = np.vstack([item[1] for item in races_xy])
    races_y = np.stack([np.stack(item[2], axis=0) for item in races_xy], 
                       axis = 0)
    train_img, test_img = races_images[:race_idx, ], races_images[race_idx:, ]
    train_feat, test_feat = races_features[:race_idx, ], races_features[race_idx:, ]
    train_y, test_y = races_y[:race_idx, ], races_y[race_idx:, ]
    
    train_y_ = [train_y[:, j, ] for j in range(train_y.shape[1])]
    

    image_scale = ImageScaler('absvalue')
    image_scale.fit(train_img, q=.95)
    feat_scale = feature_scaler.fit(train_feat)
    
    
    train_img_sc = image_scale.transform(train_img)
    test_img_sc = image_scale.transform(test_img)
    
    train_feat_sc = feat_scale.transform(train_feat)
    test_feat_sc = feat_scale.transform(test_feat)
    
    
    
    model = build_cnn(train_img_sc, train_feat_sc, train_y, **MODEL_PARAMS)
    model.fit([train_img_sc, train_feat_sc], train_y_, **FIT_PARAMS)
    
    
    
    yhat = model.predict([test_img_sc, test_feat_sc])
    yhat = np.vstack([item[0, :] for item in yhat])
    
    cce = cross_entropy(yhat, test_y[0, ])
    
    result = pd.concat([
        pd.DataFrame(np.round(yhat, 2), columns = list(map(lambda x: 'p' + str(x), range(1, yhat.shape[1] + 1)))),
        races[race_idx].start_df[['raceId', 'driverId', 'grid', 'positionText']].reset_index(drop=True)
        ],
        axis=1)
    
    return result, cce



def main(seas_min=SEAS_MIN):
    
    races = make_races(seas_min)
    
    results, cces = [], []
    
    for rc in range(_TRAIN_IDX, -1):
        race_name = racedata['races'].query(f'raceId == {races[rc].race_id}').name.iloc[0]
        print(f'\n Making CNN prediction for: {race_name}')
        race_result = train_prerace(rc, races)
        results.append(race_result[0])
        cces.append(race_result[1])
        #range(_TRAIN_IDX, 1)
        
    results_full = pd.concat(results).\
        merge(racedata['drivers'][['driverId', 'surname']]).\
        merge(racedata['races'][['raceId', 'name', 'year', 'round']]).\
        rename(columns={'name':'race', 'surname':'driver', 'grid':'start', 'positionText':'finish'})
    results_full = results_full[
        ['year', 'round', 'race', 'driver', 'start', 'finish'] + ['p'+str(i) for i in range(1, 12)]
        ].\
        rename(columns={'p11': 'p11+'}).\
        sort_values(['year', 'round', 'start'])
    
    results_full.to_csv('predictions_cnn.csv')
    save_pickle(cces, 'scoring_cnn.p')


if __name__=='__main__':
    main()
    