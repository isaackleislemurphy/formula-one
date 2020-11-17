#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 15:12:24 2020

@author: isaac.kleisle-murphy
"""


import warnings
warnings.filterwarnings('ignore')

_ITEMS = ['lap_times', 'races', 'qualifying', 'results', 'seasons', 
          'constructors', 'drivers', 'circuits', 'status', 'constructor_standings']
_SEAS_MIN = 2006

import numpy as np
import pandas as pd
from utils import save_pickle
from processing import read_data, make_circuit_dict
from processing import compute_best_quali_pace, compute_purple_quali_deltas
from processing import compute_purple_race_deltas, compute_seasonal_avg_deltas
from processing import Race, ImageScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from processing import configure_grid


def make_races(seas_min=_SEAS_MIN):
    
    racedata = {item: read_data('./data/{item}.csv'.format(item=item)) 
                for item in _ITEMS}
    
    racedata['qualifying'] = racedata['results'][['raceId', 'driverId']].\
        merge(racedata['qualifying'],
              how = 'left',
              left_on = ['raceId', 'driverId'],
              right_on = ['raceId', 'driverId'])
        
    
    racedata['results']['constructorId'] =  racedata['results']['constructorId'].apply(lambda x: 13 if x == 14 else x)
        
    racedata['results']['fastestLapSpeedOvr'] = racedata['results']. \
        groupby(['raceId'], as_index=False)['fastestLapSpeed']. \
        transform('max')
        
    racedata['results']['fastestLapSpeedTeam'] = racedata['results']. \
        groupby(['raceId', 'constructorId'], as_index=False)['fastestLapSpeed']. \
        transform('max')
        
    racedata['results']['fastestLapSpeedTeam'] = [
            np.nan if racedata['results']['fastestLapSpeedOvr'].iloc[i] - 20 > racedata['results']['fastestLapSpeedTeam'].iloc[i]
            else racedata['results']['fastestLapSpeedTeam'].iloc[i] 
        for i in range(racedata['results'].shape[0])]
        
        
    races_model = racedata['races'].sort_values(['year', 'round']).\
        query(f'year>={seas_min} & raceId <= 1040 & raceId != 79 & raceId != 926')['raceId'].\
        unique()
        
    circuit_dict = make_circuit_dict(racedata, seas_min)
    
    
    
    quali_deltas_byrace = {race_id: compute_best_quali_pace(racedata['qualifying'].query('raceId == {race_id}'.format(race_id = str(race_id))))
                           for race_id in races_model}
    
    quali_deltas_season = [compute_seasonal_avg_deltas(
            df = racedata['qualifying'], 
            df_race = racedata['races'].query('raceId <= 1040'), 
            season = ii,
            varname = 'avg_quali_delta', 
            delta_func = compute_purple_quali_deltas
        ) for ii in range(seas_min, 2021)]
    quali_deltas_season = {k: v for d in quali_deltas_season for k, v in d.items()}
    
    
    
    top_speed_deltas_season = [compute_seasonal_avg_deltas(
            df = racedata['results'], 
            df_race = racedata['races'].query('raceId <= 1040'), 
            season = ii,
            varname = 'avg_top_speed_delta', 
            delta_func = compute_purple_race_deltas
        ) for ii in range(seas_min, 2021)]
    top_speed_deltas_season = {k: v for d in top_speed_deltas_season for k, v in d.items()}



    # assemble races and split into train/test


    races = [Race(ii, 
                  racedata, 
                  quali_deltas_byrace, 
                  quali_deltas_season, 
                  top_speed_deltas_season,
                  circuit_dict
                  ) for ii in races_model]
    
    
    return races