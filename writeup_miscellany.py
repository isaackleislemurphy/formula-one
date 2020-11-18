#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 23:03:30 2020

@author: isaac.kleisle-murphy
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from ingest import make_races
from utils import save_pickle, load_pickle, cross_entropy
from processing import read_data
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

cmap = sns.diverging_palette(220, 20, as_cmap=True)

SEAS_MIN=2006
_TRAIN_IDX = -31

_ITEMS = ['lap_times', 'races', 'qualifying', 'results', 'seasons', 
          'constructors', 'drivers', 'circuits', 'status', 'constructor_standings']

racedata = {item: read_data('./data/{item}.csv'.format(item=item)) 
            for item in _ITEMS}


def image_plot(mat, label = 'Time Delta (Seconds)', filename = None):
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(mat, cmap=cmap,
                yticklabels = range(1, 21),
                xticklabels = range(1, 21),
                cbar_kws={'label': label})
    plt.xlabel('Grid Position')
    plt.ylabel('Grid Position')
    plt.yticks(rotation=0)
    
    if filename:
        plt.savefig(filename)
    
    

def get_spearman_acc(result_df):
    result_df = result_df.copy()
    result_df['likely_finish'] = [np.argmax(item) + 1 for item in 
                                  result_df[list(map(lambda x: f'p{x}', 
                                                     list(range(1, 11))+['11+']
                                                     ))].values
                                  ]
    result_df['obs_finish'] = [11 if item not in [str(i) for i in range(1, 11)] else int(item)
                               for item in result_df.finish]
    
    p1_acc_df = result_df.\
        sort_values(['race', 'round', 'p1'], ascending = False).\
        groupby(['race', 'round'], as_index=False).\
        head(1)
    
    spr = spearmanr(result_df.likely_finish, result_df.obs_finish)[0]
    acc = np.mean(result_df.likely_finish == result_df.obs_finish)
    p1_acc = np.mean(p1_acc_df.obs_finish == 1)
    
    return spr, acc, p1_acc



def predict_naive(races, race_idx):
    
    race_results = np.stack([item.to_xy_3d()[2] for item in races], axis=0)[:race_idx, ]
    yhat = np.vstack([race_results[:, i, :].mean(axis=0) for i in range(race_results.shape[1])])
    yhat[yhat==0] = .0001
    
    
    result = pd.concat([
        pd.DataFrame(np.round(yhat, 2), columns = list(map(lambda x: 'p' + str(x), range(1, yhat.shape[1] + 1)))),
        races[race_idx].start_df[['raceId', 'driverId', 'grid', 'positionText']].reset_index(drop=True)
        ],
        axis=1)
    
    cce = cross_entropy(yhat, race_results[race_idx + 1, ])
    
    
    results_full = result.\
        merge(racedata['drivers'][['driverId', 'surname']]).\
        merge(racedata['races'][['raceId', 'name', 'year', 'round']]).\
        rename(columns={'name':'race', 'surname':'driver', 'grid':'start', 'positionText':'finish'})
    results_full = results_full[
        ['year', 'round', 'race', 'driver', 'start', 'finish'] + ['p'+str(i) for i in range(1, 12)]
        ].\
        rename(columns={'p11': 'p11+'}).\
        sort_values(['year', 'round', 'start'])
    
    return results_full, cce




def main():
    
    races = make_races()
    
    
    ##################################################################
    # EXAMPLE VISUALIZATIONS FOR WRITEUP: Russia 2020 as example
    ##################################################################
    
    
    sochi_img = races[-1].to_xy_3d()[0]
    
    quali_deltas = sochi_img[0:20, 0:20, 0]
    quali_deltas_season = sochi_img[0:20, 0:20, 1]
    mph_deltas = sochi_img[0:20, 0:20, 2]
    
    # qualifying deltas
    image_plot(quali_deltas, filename = './pngs/quali_deltas.png')
    plt.clf()

    # qualifying seasonal deltas    
    image_plot(quali_deltas_season, 
               label = 'Avg. % Difference',
               filename = './pngs/quali_deltas_season.png')
    plt.clf()
    
    # top speed deltas
    image_plot(mph_deltas, label = 'Max Speed Delta (MPH)', filename = './pngs/mph_deltas_season.png')
    plt.clf()
    
    ##################################################################
    # EXAMPLE VISUALIZATIONS FOR POSTER: MONACO
    ################################################################## 
    
    monaco_img = races[-26].to_xy_3d()[0]
    
    quali_deltas = monaco_img[0:20, 0:20, 0]
    image_plot(quali_deltas, filename = './pngs/quali_deltas_monaco.png')
    # plt.show()
    plt.clf()
    
    # qualifying deltas
    image_plot(quali_deltas, filename = './pngs/quali_deltas.png')
    plt.clf()
    
    
    ##################################################################
    # PARSE RESULTS FOR 2020 RACES//TABLE-MAKING
    ##################################################################
    
    results_naive = pd.concat([predict_naive(races, j)[0]for j in range(_TRAIN_IDX, -1)], axis=0)
    cce_naive = [predict_naive(races, j)[1]for j in range(_TRAIN_IDX, -1)]
    
    cce_multinom = load_pickle('scoring_multinomial.p')
    results_multinom = pd.read_csv('./predictions/predictions_multinomial.csv').sort_values(['year', 'round'])
    
    cce_ffnn = load_pickle('scoring_ffnn.p')
    results_ffnn = pd.read_csv('./predictions/predictions_ffnn.csv').sort_values(['year', 'round'])
    
    cce_cnn = load_pickle('scoring_cnn.p')
    results_cnn = pd.read_csv('./predictions/predictions_cnn.csv').sort_values(['year', 'round'])
    
    
    spr_acc_naive = get_spearman_acc(results_naive)
    spr_acc_multinom = get_spearman_acc(results_multinom)
    spr_acc_ffnn = get_spearman_acc(results_ffnn)
    spr_acc_cnn = get_spearman_acc(results_cnn)
    
    print('\nNaive Performance:')
    print(f'\tCCE: \t{np.mean(cce_naive)}')
    print(f'\tSpearman: \t{spr_acc_naive[0]}')
    print(f'\tAcc: \t{spr_acc_naive[1]}')
    print(f'\tP1 Acc: \t{spr_acc_naive[2]}')
    
    
    
    print('\nMultinomial Performance:')
    print(f'\tCCE: \t{np.mean(cce_multinom)}')
    print(f'\tSpearman: \t{spr_acc_multinom[0]}')
    print(f'\tAcc: \t{spr_acc_multinom[1]}')
    print(f'\tP1 Acc: \t{spr_acc_multinom[2]}')
    
    
    
    print('\nFeed-Forward Performance:')
    print(f'\tCCE: \t{np.mean(cce_ffnn)}')
    print(f'\tSpearman: \t{spr_acc_ffnn[0]}')
    print(f'\tAcc: \t{spr_acc_ffnn[1]}')
    print(f'\tP1 Acc: \t{spr_acc_ffnn[2]}')
    
    
    print('\nCNN Performance:')
    print(f'\tCCE: \t{np.mean(cce_cnn)}')
    print(f'\tSpearman: \t{spr_acc_cnn[0]}')
    print(f'\tAcc: \t{spr_acc_cnn[1]}')
    print(f'\tP1 Acc: \t{spr_acc_cnn[2]}')
    
    
    ##################################################################
    # PARSE RESULTS FOR 2020 RACES//TABLE-MAKING
    ##################################################################
    
    # Model Comparison for Lewis Hamilton: P1
    ham_multinom = results_multinom.query('driver=="Hamilton"').reset_index()
    ham_ffnn = results_ffnn.query('driver=="Hamilton"').reset_index()
    ham_cnn = results_cnn.query('driver=="Hamilton"').reset_index()
    
    n_races = len(ham_multinom)
    race_range = range(1, n_races + 1)
    
    ymax = np.max(list(ham_multinom.p1) + list(ham_ffnn.p1) + list(ham_cnn.p1))
    
    plt.plot(race_range, ham_multinom.p1, label = 'Multinomial Logistic')
    plt.plot(race_range, ham_ffnn.p1, label = 'Feed-Forward NN')
    # plt.plot(race_range, ham_cnn.p1, label = 'Convolutional NN')
    plt.xlim(.5, )
    plt.ylim(0, )
    plt.xlabel('Race Number')
    plt.ylabel('Win Probability')
    plt.vlines([21], linestyle='dashed', alpha = .5, ymin = 0, ymax=ymax, colors=['black'],
               label = 'End of 2019 Season')
    plt.legend(loc=2)
    #plt.show()
    plt.savefig('./pngs/ham.png')
    plt.clf()
    
    
    
    # Model Comparison for Daniel Ricciardo
    ric_multinom = results_multinom.query('driver=="Ricciardo"').reset_index()
    ric_ffnn = results_ffnn.query('driver=="Ricciardo"').reset_index()
    ric_cnn = results_cnn.query('driver=="Ricciardo"').reset_index()
    
    ymax = np.max(list(ric_multinom[['p1', 'p2', 'p3']].values.sum(axis=1)) + \
        list(ric_ffnn[['p1', 'p2', 'p3']].values.sum(axis=1)) + \
        list(ric_cnn[['p1', 'p2', 'p3']].values.sum(axis=1)))
    
    plt.plot(race_range, ric_multinom[['p1', 'p2', 'p3']].values.sum(axis=1), label = 'Multinomial Logistic')
    plt.plot(race_range, ric_ffnn[['p1', 'p2', 'p3']].values.sum(axis=1), label = 'Feed-Forward NN')
    # plt.plot(race_range, ric_cnn[['p1', 'p2', 'p3']].values.sum(axis=1), label = 'Convolutional NN')
    plt.xlim(.5, )
    plt.ylim(0, )
    plt.xlabel('Race Number')
    plt.ylabel('Podium Probability')
    plt.vlines([21], linestyle='dashed', alpha = .5, ymin = 0, ymax=ymax, colors=['black'],
               label = 'End of 2019 Season')
    plt.legend(loc=2)
    #plt.show()
    plt.savefig('./pngs/ric.png')
    plt.clf()
    
    
    
    # Model Comparison for George Russell: Points probability
    rus_multinom = results_multinom.query('driver=="Russell"').reset_index()
    rus_ffnn = results_ffnn.query('driver=="Russell"').reset_index()
    rus_cnn = results_cnn.query('driver=="Russell"').reset_index()
    
    ymax = np.max(list(1-rus_multinom['p11+']) + list(1-rus_ffnn['p11+']) + list(1-rus_cnn['p11+']))
    
    
    plt.plot(race_range, 1-rus_multinom['p11+'], label = 'Multinomial Logistic')
    plt.plot(race_range, 1-rus_ffnn['p11+'], label = 'Feed-Forward NN')
    #plt.plot(race_range, 1-rus_cnn['p11+'], label = 'Convolutional NN')
    plt.xlim(.5, )
    plt.ylim(0, )
    plt.xlabel('Race Number')
    plt.ylabel('Points Probability')
    plt.vlines([21], linestyle='dashed', alpha = .5, ymin = 0, ymax=ymax, colors=['black'],
               label = 'End of 2019 Season')
    plt.legend(loc=2)
    # plt.show()
    plt.savefig('./pngs/rus.png')
    plt.clf()
    
    
    
    
    race_range = range(1, 22)
    
    ric_multinom = results_multinom.query('driver=="Ricciardo" & year==2019').reset_index()
    ric_ffnn = results_ffnn.query('driver=="Ricciardo" & year==2019').reset_index()
    
    hul_multinom = results_multinom.query('(driver=="Hülkenberg") & year==2019')
    hul_ffnn = results_ffnn.query('(driver=="Hülkenberg") & year==2019')
    
    plt.plot(race_range, ric_multinom[['p1', 'p2', 'p3']].values.sum(axis=1), 
             label = 'Multinomial Logistic', linestyle='dashed', color='black')
    
    plt.plot(race_range, ric_ffnn[['p1', 'p2', 'p3']].values.sum(axis=1), 
             label = 'Feed-Forward NN', color='black')
    
    
    plt.plot(race_range, hul_multinom[['p1', 'p2', 'p3']].values.sum(axis=1), 
             label = 'Multinomial Logistic', linestyle='dashed', color='#FFF500')
    
    plt.plot(race_range, hul_ffnn[['p1', 'p2', 'p3']].values.sum(axis=1), 
             label = 'Feed-Forward NN', color='#FFF500')
    
    plt.legend()
    
    
