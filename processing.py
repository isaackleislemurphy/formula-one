#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 11:07:44 2020

@author: isaac.kleisle-murphy
"""

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from wpearson import wpearson




_TIME_COLS = ['q1', 'q2', 'q3', 'time']
_EMPTY_FILL = 1.2/1.07

_POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1] + list(np.zeros(20))

# by year: recent --> old
_WET_RACE_IDS = [
    1020,
    999, 
    970, 982,
    953, 957, 967,
    934, 942, 
    910, 914,
    881,
    861, 879,
    847, 849, 851,
    338, 340, 349, 353,
    2, 3,
    23, 26, 30, 31, 35,
    45, 50, 51,
    65, 68
    ]

def build_time(x):
    x_split = x.split(':')
    time = float(x_split[0])* 60 + float(x_split[1])
    return time


def read_data(filename, time_cols=_TIME_COLS):
    
    df_raw = pd.read_csv(filename)

    for col in time_cols:
        if col in df_raw:
            #fix the time formatting
            times = []
            for item in df_raw[col].to_list():
                try:
                    result = build_time(item)
                except:
                    result = np.nan
                times.append(result)
            df_raw[col] = times
            
    if 'fastestLapSpeed' in df_raw.columns:
        df_raw['fastestLapSpeed'] = [np.nan if ('N' in item) else float(item) for item in df_raw['fastestLapSpeed']]
    
    return df_raw


def compute_best_quali_pace(df):
    
    pace_delta_dict = {}

    for _, row in df.iterrows():
        driver_id = int(row[['driverId']])
        flap_driver = np.nanmin(row[['q1', 'q2', 'q3']].values)
        pace_delta_dict[driver_id] = flap_driver

    return pace_delta_dict




def compute_purple_quali_deltas(df):

    flap = np.nanmin(df[['q1', 'q2', 'q3']].values)
    pace_delta_dict = {}

    for _, row in df.iterrows():
        driver_id = int(row[['driverId']])
        flap_driver = np.nanmin(row[['q1', 'q2', 'q3']].values)
        pace_delta = flap_driver/flap
        pace_delta_dict[driver_id] = pace_delta - 1

    return pace_delta_dict


def compute_purple_race_deltas(df, benchmark = 'fastestLapSpeedTeam'):

    flap = np.nanmax(df[benchmark].values)
    pace_delta_dict = {}

    for _, row in df.iterrows():
        driver_id = int(row[['driverId']])
        flap_driver = row[benchmark]
        if benchmark == 'fastestLapSpeedTeam':
            pace_delta = flap - flap_driver
        else:
            pass
            
        pace_delta_dict[driver_id] = pace_delta

    return pace_delta_dict



def compute_avg_deltas(df, race_ids, delta_func = compute_purple_quali_deltas):
    
    df=df.copy()
    driver_dict = {driver: np.array([]) for driver in df.loc[(df.raceId).isin(race_ids), :].driverId.unique()}

    for race in race_ids:

        race_df = df.loc[(df.raceId==race), :]
        deltas_race = delta_func(race_df)

        for driver, delta in deltas_race.items():
            driver_dict[driver] = np.append(driver_dict[driver], delta)

    driver_delta_means = {k: np.nanmean(v) for k,v in driver_dict.items()}
    
    return driver_delta_means

def compute_seasonal_avg_deltas(df, df_race, season, varname, **kwargs):
    
    df, df_race = df.copy(), df_race.copy()
    df_races_season = df_race.loc[(df_race.year==season), :].sort_values('round')
    result = {}

    for _, rd in df_races_season.iterrows():

        prev_races = df_races_season.loc[(df_races_season['round'] <= rd['round']), :]
        avg_deltas = compute_avg_deltas(df, race_ids = prev_races.raceId.unique(), **kwargs)

#         temp_df = pd.DataFrame(avg_deltas, range(len(avg_deltas))).T.iloc[:, 0].reset_index()
#         temp_df.columns = ['driverId', varname] #
#         temp_df['round'] = rd['round']
#         temp_df['raceId'] = rd['raceId']
#         temp_df['year'] = rd['year']
#         deltas_byround.append(temp_df)
        
        result[rd['raceId']] = avg_deltas

    #result = pd.concat(deltas_byround, sort=True).reset_index(drop=True)

    return result


def assemble_delta_matrix(driver_dict, driver_order, operation = np.subtract):

    #delta matrix: row := grid position from, column := grid position to
    delta_matrix = []
    ordered_values = np.array([driver_dict[drv] for drv in driver_order])
    for drv_value in ordered_values:
        delta_matrix.append(operation(drv_value, ordered_values))
    delta_matrix = np.vstack(delta_matrix)
    
    return delta_matrix




def make_circuit_dict(racedata, seas_min=2006):
    
    circuits_model = racedata['races'].\
    query(f'year>={seas_min}').\
    groupby('circuitId', as_index=False).\
    count().\
    iloc[:, 0:2]
    circuits_model.columns = ['circuit_id', 'num_races']
    circuits_dict={}
    ctr = 1
    for _, n in circuits_model.iterrows():
        if n['num_races'] >= 3:
            circuits_dict[n['circuit_id']] = ctr
            ctr+=1
        else:
            circuits_dict[n['circuit_id']] = 0
    
    return circuits_dict



def configure_grid(racedata, race_id, max_starters=24):
    result_df = racedata['results'].query(f'raceId == {race_id}')
    grid_df = result_df.query('grid >=1 & positionText != "W"')
    pitlane_df = result_df.query('grid == 0 & positionText != "W"')
    no_start_df = result_df.query('positionText == "W"')
        
    start_df = pd.concat([grid_df.sort_values('grid'), 
                                   pitlane_df,
                                   no_start_df
                                   ], axis=0).iloc[0:max_starters, ]
    
    return start_df











def tally_career_starts(racedata, race_id, max_starters=24):
    
    rd = racedata['races'].query(f'raceId == {race_id}')['round'].iloc[0]
    yr = racedata['races'].query(f'raceId == {race_id}')['year'].iloc[0]
    prev_race_ids = racedata['races'].query(f'year < {yr} | (year == {yr} & round < {rd})')['raceId'].unique().tolist()
    prev_races = racedata['results'].loc[(racedata['results'].raceId).isin(prev_race_ids)]
    
    career_starts = prev_races.\
        groupby('driverId', as_index=False)['resultId'].\
        count()
     
    result = []
    grid_order = configure_grid(racedata, race_id, max_starters).driverId.tolist()
    for driver in grid_order:
        if driver in career_starts.driverId.values:
            result.append(career_starts.query(f'driverId=={driver}')['resultId'].iloc[0])
        else:
            result.append(0)
            
    return np.array(result + list(-np.ones(max_starters - len(grid_order))))


def tally_season_wins(racedata, race_id, max_starters=24):
    
    rd = racedata['races'].query(f'raceId == {race_id}')['round'].iloc[0]
    yr = racedata['races'].query(f'raceId == {race_id}')['year'].iloc[0]
    prev_race_ids = racedata['races'].query(f'year == {yr} & round < {rd}')['raceId'].unique().tolist()
    prev_races = racedata['results'].loc[(racedata['results'].raceId).isin(prev_race_ids)].copy()
    prev_races['is_win'] = prev_races['positionOrder'].apply(lambda x: 1 if x == 1 else 0)
    
    if rd==1:
        return np.zeros(max_starters)
    
    win_counts = prev_races.groupby('driverId', as_index=False)['is_win'].sum()
        
    result = []
    grid_order = configure_grid(racedata, race_id, max_starters).driverId.tolist()
    for driver in grid_order:
        if driver in win_counts.driverId.values:
            result.append(win_counts.query(f'driverId=={driver}')['is_win'].iloc[0])
        else:
            result.append(0)
            
    return np.array(result + list(-np.ones(max_starters - len(grid_order))))






def tally_season_podiums(racedata, race_id, max_starters=24):
    
    '''
    EXCLUDING WINS
    '''
    
    rd = racedata['races'].query(f'raceId == {race_id}')['round'].iloc[0]
    yr = racedata['races'].query(f'raceId == {race_id}')['year'].iloc[0]
    prev_race_ids = racedata['races'].query(f'year == {yr} & round < {rd}')['raceId'].unique().tolist()
    prev_races = racedata['results'].loc[(racedata['results'].raceId).isin(prev_race_ids)].copy()
    prev_races['is_podium'] = prev_races['positionOrder'].apply(lambda x: 1 if x in [2, 3] else 0)
    
    if rd==1:
        return np.zeros(max_starters)
    
    podium_counts = prev_races.groupby('driverId', as_index=False)['is_podium'].sum()
        
    result = []
    grid_order = configure_grid(racedata, race_id, max_starters).driverId.tolist()
    for driver in grid_order:
        if driver in podium_counts.driverId.values:
            result.append(podium_counts.query(f'driverId=={driver}')['is_podium'].iloc[0])
        else:
            result.append(0)
            
    return np.array(result + list(-np.ones(max_starters - len(grid_order))))





def tally_season_points(racedata, race_id, max_starters=24, points=_POINTS):
    
    '''
    '''
    
    rd = racedata['races'].query(f'raceId == {race_id}')['round'].iloc[0]
    yr = racedata['races'].query(f'raceId == {race_id}')['year'].iloc[0]
    prev_race_ids = racedata['races'].query(f'year == {yr} & round < {rd}')['raceId'].unique().tolist()
    prev_races = racedata['results'].loc[(racedata['results'].raceId).isin(prev_race_ids)].copy()
    prev_races['points'] = prev_races['positionOrder'].apply(lambda x: points[int(x-1)])
    
    if rd==1:
        return np.zeros(max_starters)
    
    points_tally = prev_races.groupby('driverId', as_index=False)['points'].sum()
        
    result = []
    grid_order = configure_grid(racedata, race_id, max_starters).driverId.tolist()
    for driver in grid_order:
        if driver in points_tally.driverId.values:
            result.append(points_tally.query(f'driverId=={driver}')['points'].iloc[0])
        else:
            result.append(0)
            
    return np.array(result + list(-np.ones(max_starters - len(grid_order))))



def tally_recent_crashes(racedata, race_id, max_starters=24, lb=3):
    
    '''
    '''
    
    
    rd = racedata['races'].query(f'raceId == {race_id}')['round'].iloc[0]
    yr = racedata['races'].query(f'raceId == {race_id}')['year'].iloc[0]
    prev_race_ids = racedata['races'].query(
        f'(year < {yr} & year > {yr}-3)| (year == {yr} & round < {rd})'
        )['raceId'].unique().tolist()
    prev_races = racedata['results'].loc[(racedata['results'].raceId).isin(prev_race_ids)].copy()
    prev_races['crashes'] = prev_races['statusId'].apply(
        lambda x: 1 if x in [3, 4, 130, 137, 138, 41, 20] else 0
        )
    
    crash_tally = prev_races.groupby('driverId', as_index=False)['crashes'].sum()
        
    result = []
    grid_order = configure_grid(racedata, race_id, max_starters).driverId.tolist()
    for driver in grid_order:
        if driver in crash_tally.driverId.values:
            result.append(crash_tally.query(f'driverId=={driver}')['crashes'].iloc[0])
        else:
            result.append(0)
            
    return np.array(result + list(-np.ones(max_starters - len(grid_order))))






def engineer_driver_features(racedata, race_id, max_starters=24,
                             funs=[tally_career_starts, tally_season_wins, 
                                   tally_season_podiums, tally_season_points,
                                   tally_recent_crashes]):
    
    result = np.hstack([fun(racedata, race_id, max_starters) for fun in funs])
    
    return result



def engineer_standings_features(racedata, race_id, max_starters=24):
    
    yr = racedata['races'].query(f'raceId == {race_id}')['year'].iloc[0] - 1
    last_race = racedata['races'].query(f'year=={yr}').sort_values('round').tail(1).raceId.iloc[0]
    prev_season_standings = racedata['constructor_standings'].query(f'raceId=={last_race}')
    
    grd = configure_grid(racedata, race_id, max_starters)
    #TORO ROSSO -> ALPHA TAURI // ALFA ROMEO --> SAUBER
    grd['constructorId'] = grd['constructorId'].apply(lambda x: 5 if x in [212, 213] else 15 if x==51 else x)
    grd = grd.\
        merge(prev_season_standings[['constructorId', 'position']],
              how='left', left_on='constructorId', right_on='constructorId',
              suffixes = ['', '_cons']).\
        fillna(value={'position_cons':16}) #16 indicates new constructor
        
    return np.array(list(grd.position_cons) + list(20* np.ones(max_starters - len(grd))))


        

def engineer_reliability_features(racedata, race_id, max_starters=24):
    
    retirement_status = racedata['status'].loc[['+' not in item for item in racedata['status'].status], :]\
        ['statusId'].\
        unique()
    retirement_status = set(retirement_status).difference(set([1,2,3,4, 20, 29, 41, 59, 130, 137, 138]))    
    
    '''Gets constructor stats UP TO race_id'''
    
    rd = racedata['races'].query(f'raceId == {race_id}')['round'].iloc[0]
    if rd == 1:
        return np.zeros(max_starters)
    
    yr = racedata['races'].query(f'raceId == {race_id}')['year'].iloc[0]
    prev_race_ids = racedata['races'].query(f'year == {yr} & round < {rd}')['raceId'].unique().tolist()
    prev_races = racedata['results'].loc[(racedata['results'].raceId).isin(prev_race_ids)]
    prev_races['rt'] = prev_races['statusId'].apply(
        lambda x: 1 if x in retirement_status else 0
        ).values
    
    
    constructor_retirements = prev_races.groupby('constructorId', as_index=False)['rt'].sum()
    results_df = racedata['results'].query(f'raceId=={race_id}').\
        merge(constructor_retirements, how='left', left_on='constructorId', right_on='constructorId')
    
    grid_order = configure_grid(racedata, race_id, max_starters).driverId.tolist()
    driver_dict = {}
    
    for _, row in results_df.iterrows():
        driver_dict[row['driverId']] = row['rt'] if not np.isnan(row['rt']) else 0
    
    return np.array([driver_dict[i] for i in grid_order] + list(-np.ones(max_starters - len(grid_order))))


def engineer_race_features(racedata, race_id, lb=8):
    
    rd = racedata['races'].query(f'raceId == {race_id}')['round'].iloc[0]
    yr = racedata['races'].query(f'raceId == {race_id}')['year'].iloc[0]
    circuit = racedata['races'].query(f'raceId == {race_id}')['circuitId'].iloc[0]
    
    prev_races_all = racedata['races'].query(
        f'year < {yr} & circuitId == {circuit}').raceId.tolist()
    
    if len(prev_races_all):
    
        prev_races_recent = racedata['races'].query(
            f'year < {yr} & circuitId == {circuit} & year >= {yr}-{lb}').raceId.tolist()
        
        prev_results_all = racedata['results'].loc[(racedata['results'].raceId).isin(prev_races_all)]
        
        
        prev_results_recent = racedata['results'].loc[(racedata['results'].raceId).isin(prev_races_recent)].\
            query('grid > 0')
            
        prev_results_recent['points'] = [_POINTS[int(ii - 1)] + .01 for ii in prev_results_recent.grid]
        
        try:
            start_finish_corr = pearsonr(prev_results_recent.grid, prev_results_recent.positionOrder)[0]
            start_finish_wcorr = wpearson(
                        prev_results_recent.query('positionText!="R"').grid.values, 
                        prev_results_recent.query('positionText!="R"').positionOrder.values, 
                        prev_results_recent.query('positionText!="R"').points
                        )
        except:
            start_finish_corr = 0
            start_finish_wcorr = 0
            
        #driver Ids of those in the race
        race_drivers = racedata['results'].query(f'raceId == {race_id}').driverId
        pct_first_time = 1 - len(set(prev_results_all.driverId).intersection(race_drivers))/len(race_drivers)
    
    else:
        pct_first_time, start_finish_corr, start_finish_wcorr = 1, 0, 0
    
    return np.array([rd, pct_first_time, start_finish_corr, start_finish_wcorr])




    
    


def resize_matrix(A, max_size, m = 1):
            
    bigger_matrix = np.zeros((max_size, max_size))
    bigger_matrix[0:A.shape[0], 0:A.shape[1]] = A
    bigger_matrix[A.shape[1]:max_size, A.shape[1]:max_size] = np.eye(max_size - A.shape[1])*m
    
    return bigger_matrix





def check_images(images):
    
    flag_matrix = []
    for ii in range(images.shape[0]):
        iter_flag = []
        for dd in range(images.shape[3]):
            iter_flag.append(np.sum(np.isnan(images[ii, :, :, dd]) > 0))
        flag_matrix.append(iter_flag)
        
    return np.vstack(flag_matrix)


def check_features(features):

    return np.isnan(features).sum(axis=0)



class Race:
    
    def __init__(self, 
                 race_id, 
                 racedata, 
                 quali_deltas_byrace,
                 quali_deltas_season,
                 top_speed_deltas_season,
                 circuit_dict = None,
                 max_starters=24, 
                 pitlane_time_flag=1, 
                 max_y=10):
        #TODO: drop Australia speeds
        
        
        self.race_id=race_id
        self.round = racedata['races'].query(f'raceId=={self.race_id}')['round'].iloc[0]
        self.quali_deltas = copy.deepcopy(quali_deltas_byrace[race_id])
        self.quali_deltas_season = copy.deepcopy(quali_deltas_season[race_id])
        self.top_speed_deltas_season = copy.deepcopy(top_speed_deltas_season[race_id])
        self.max_starters=max_starters
        self.pitlane_time_flag=5
        self.y={}
        
        
        
        self.start_df = configure_grid(racedata, race_id, max_starters)
        self.n_starters = self.start_df.query('grid >= 1 & positionText != "F" & positionText != "W"').shape[0]
        self.starter_vec = np.hstack([np.zeros(self.n_starters), np.ones(max_starters - self.n_starters)])
        
        #process quali times
        
        self.time_107 = min(self.quali_deltas.values()) * 1.07
        #self.fill_value = (_EMPTY_FILL - 1) * min(self.quali_deltas.values())

        self.quali_deltas_adjusted = {k: v if (not np.isnan(v) and v <= self.time_107) 
                                      else self.time_107 + pitlane_time_flag
                                      for k, v in self.quali_deltas.items()}
        
        self.quali_deltas_season_adjusted = {k:v if not np.isnan(v)
                                             else _EMPTY_FILL * max(list(self.quali_deltas_season.values()))
                                             for k, v in self.quali_deltas_season.items()}
        
        
        self.quali_deltas_matrix = assemble_delta_matrix(self.quali_deltas_adjusted, 
                                                         self.start_df.driverId.to_list())
        
        self.quali_deltas_season_matrix = assemble_delta_matrix(self.quali_deltas_season_adjusted, 
                                                                self.start_df.driverId.to_list())
        
        self.top_speed_deltas_season_matrix = assemble_delta_matrix(self.top_speed_deltas_season, 
                                                                    self.start_df.driverId.to_list())
        # fill in Australia 2011/2012 non-starters as if they're image padding
        self.top_speed_deltas_season_matrix[np.isnan(self.top_speed_deltas_season_matrix)] = 0
            
        
        
        if max_starters > self.quali_deltas_matrix.shape[1]:
            
            #bigger_matrix = np.tril(self.fill_value * np.ones((max_starters, max_starters)), -1) + \
            #    np.triu(self.fill_value * np.ones((max_starters, max_starters)), 1)
            self.quali_deltas_matrix = resize_matrix(self.quali_deltas_matrix, max_size=max_starters)
            self.quali_deltas_season_matrix = resize_matrix(self.quali_deltas_season_matrix, max_size=max_starters)
            self.top_speed_deltas_season_matrix = resize_matrix(self.top_speed_deltas_season_matrix, max_size=max_starters)
            
            
        
        # ENGINEER 1D FEATURES
        
        self.season = racedata['races'].query(f'raceId == {race_id}').year.iloc[0]
        
        if circuit_dict:
            self.circuit_dict = circuit_dict
            race_circuit = racedata['races'].query(f'raceId=={race_id}')['circuitId'].iloc[0]
            self.circuit_array=np.zeros(len(circuit_dict))
            self.circuit_array[circuit_dict[race_circuit]] = 1
            
        self.reliability_features = engineer_reliability_features(racedata, race_id, max_starters)
        self.race_features = engineer_race_features(racedata, race_id, lb=5)
        self.driver_features = engineer_driver_features(racedata, race_id, max_starters)
        self.prev_constructor_standings = engineer_standings_features(racedata, race_id, max_starters)
        self.is_rain = 1 if race_id in _WET_RACE_IDS else 0
        
        # ENGINEER TARGETS

        #for pos in range(0, max_y):
        #    
        #    flag_order = list(self.start_df.positionOrder.values) + \
        #            list(set(range(1, self.max_starters+1)).difference(
        #                self.start_df.positionOrder.values
        #                ))
        #            
        #    self.y[pos+1] = np.array(np.array(flag_order) == pos+1).astype(int)
        y_template = np.zeros(max_y + 1)
        for pos in range(max_starters):
            y_working = y_template.copy()
            if pos < len(self.start_df):
                finish_pos = self.start_df.positionOrder.values[pos]
                y_working[min(finish_pos-1, len(y_template) - 1)] = 1
            else:
                y_working[-1] = 1
            self.y[pos+1] = y_working
            
            
    def to_xy_2d(self):
        
        conv_layers = np.stack([
            self.quali_deltas_matrix,
            self.quali_deltas_season_matrix],
            axis=2
            )
        
        feature_layers = np.hstack(
            [self.circuit_array,
             self.driver_features,
             self.reliability_features,
             self.race_features,
             self.prev_constructor_standings,
             self.starter_vec,
             self.is_rain]
            )
        
        
        y = [self.y[pos] for pos in np.sort(list(self.y.keys()))]
        
        return [conv_layers, feature_layers, y]
    
    
    def to_xy_3d(self):
        
        conv_layers = np.stack([
            self.quali_deltas_matrix,
            self.quali_deltas_season_matrix,
            self.top_speed_deltas_season_matrix],
            axis=2
            )
        
        feature_layers = np.hstack(
            [self.circuit_array,
             self.driver_features,
             self.reliability_features,
             self.race_features,
             self.prev_constructor_standings,
             self.starter_vec,
             self.is_rain]
            )
        
        
        y = [self.y[pos] for pos in np.sort(list(self.y.keys()))]
        
        return [conv_layers, feature_layers, y]
    
    
    def to_xy_1d(self):
        
        x_tall = []
        
        for drv in range(self.max_starters):
            
            #indicators for starting position on grid
            x = list(np.zeros(self.max_starters))
            x[drv] = 1
            
            #indicator if DNS
            dns_flag = 1 if  self.starter_vec[drv] else 0
            x.append(dns_flag)
            
            #time-based features
            x += list(self.quali_deltas_matrix[:, drv])
            x += list(self.quali_deltas_season_matrix[:, drv])
            x += list(self.top_speed_deltas_season_matrix[:, drv])
            
            x += list(self.circuit_array)
            x += list(self.race_features)
            
            drv_features = self.driver_features[list(range(drv, len(self.driver_features), self.max_starters))]
            drv_features[drv_features < 0] = 0
            x += list(drv_features)
            
            rel_features = self.reliability_features[list(range(drv, len(self.reliability_features), self.max_starters))]/\
                    self.round
            rel_features[rel_features < 0] = 0
            x += list(rel_features)
            
            x += [
                self.prev_constructor_standings[drv] if self.prev_constructor_standings[drv] < 16 else 0,
                1 if self.prev_constructor_standings[drv] == 16 else 0
                ]
            x += [self.is_rain]
            
            x_tall.append(x)
            
        x_tall = np.vstack(x_tall)
        
        y = [self.y[pos] for pos in np.sort(list(self.y.keys()))]
        
        return x_tall, y
            
            
            
            
        
        


class ImageScaler:
    
    def __init__(self, method = 'spread'):
        
        self.X=None
        self.is_fited=False
        self.method=method
    
    def fit(self, X, q = .95, p = 15):
        
        self.x_fit = X
        self.image_depth = X[0].shape[-1]
        image_scales = []
        
        if self.method == 'minmax':
        
            for lyr in range(self.image_depth):
                
                lyr_min = np.min(X[:, :, :, lyr])
                lyr_max = np.max(X[:, :, :, lyr])
                image_scales.append([lyr_min, lyr_max])
                
        elif self.method == 'absvalue':
            
            for lyr in range(self.image_depth):
                
                lyr_min = 0
                lyr_max = np.abs(np.quantile(X[:, :, :, lyr], q))
                image_scales.append([lyr_min, lyr_max])
                
        elif self.method == 'spread':
            
            for lyr in range(self.image_depth):
                
                lyr_min = 0
                lyr_max = np.abs(np.mean(X[:, p-1, 0, lyr]))
                image_scales.append([lyr_min, lyr_max])
            
            
        self.scales = np.vstack(image_scales)
        
        
    def transform(self, X):
        
        x_copy = X.copy()
        for lyr in range(self.image_depth):
            lyr_min, lyr_max = self.scales[lyr][0], self.scales[lyr][1]
            x_copy[:, :, :, lyr] = (x_copy[:, :, :, lyr] - lyr_min)/(lyr_max - lyr_min)
            
        return x_copy
    
    
    def fit_transform(self, X):
        
        self.fit(X)
        
        return self.transform(X)  