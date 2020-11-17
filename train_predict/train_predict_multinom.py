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
from sklearn.preprocessing import StandardScaler
from processing import read_data
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

SEAS_MIN=2006
_TRAIN_IDX = -31

_ITEMS = ['lap_times', 'races', 'qualifying', 'results', 'seasons', 
          'constructors', 'drivers', 'circuits', 'status', 'constructor_standings']

racedata = {item: read_data('./data/{item}.csv'.format(item=item)) 
            for item in _ITEMS}


FIT_PARAMS = {'C': .005}


def train_prerace(race_idx, x_all, y_all, races, seas_min=SEAS_MIN):
    
    x_train, x_test = x_all[:race_idx, ], x_all[race_idx:, ]
    y_train, y_test = y_all[:race_idx, ], y_all[race_idx:, ]
    y_train_sparse = y_train.dot(range(0, y_train.shape[2]))
    
    scaler = StandardScaler().fit(np.vstack(x_train))
    
    x_train_sc = np.stack([scaler.transform(item) for item in x_train], axis=0)
    x_test_sc = np.stack([scaler.transform(item) for item in x_test], axis=0)

    model = LogisticRegression(penalty='l2', solver='saga', 
                               multi_class='multinomial', max_iter=1e8, **FIT_PARAMS)
    model.fit(np.vstack(x_train_sc), y_train_sparse.ravel())

        
    yhat = model.predict_proba(x_test_sc[0, ])
    
    cce = cross_entropy(yhat, y_test[0, ])
    
    result = pd.concat([
        pd.DataFrame(np.round(yhat, 2), columns = list(map(lambda x: 'p' + str(x), range(1, yhat.shape[1] + 1)))),
        races[race_idx].start_df[['raceId', 'driverId', 'grid', 'positionText']].reset_index(drop=True)
        ],
        axis=1)
    
    return result, cce



def main(seas_min=SEAS_MIN):
    
    races = make_races(seas_min)
    races_xy = [item.to_xy_1d() for item in races]
    x_all = np.stack([item[0] for item in races_xy], axis=0)
    y_all = races_y = np.stack([np.stack(item[1], axis=0) for item in races_xy], axis=0)
    
    results, cces = [], []
    
    for rc in range(_TRAIN_IDX, -1):
        race_name = racedata['races'].query(f'raceId == {races[rc].race_id}').name.iloc[0]
        print(f'\n Making multinomial prediction for: {race_name}')
        race_result = train_prerace(rc, x_all, y_all, races)
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
        
        
    results_full.to_csv('predictions_multinomial.csv')
    save_pickle(cces, 'scoring_multinomial.p')
    



if __name__=='__main__':
    main()
    