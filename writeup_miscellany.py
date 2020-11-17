#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 23:03:30 2020

@author: isaac.kleisle-murphy
"""

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import plot_model


###################
# FFNN





###################
# CNN

opt_ep=np.argmin(grid_search.get_best_tune_results().weigh_scores()[0]) + 1

for ii in range(len(grid_search.tune_results[0].history)):
    tune = grid_search.tune_results[0].history[ii]
    
    val_keys = [f'val_output_p{j}_categorical_crossentropy' for j in range(1, 25)]
    tr_keys = [f'output_p{j}_categorical_crossentropy' for j in range(1, 25)]
    
    val_losses = np.vstack(tune[k] for k in val_keys).mean(axis=0)
    tr_losses = np.vstack(tune[k] for k in tr_keys).mean(axis=0)
    
    plt.plot(val_losses, color = 'red', alpha = .225)
    plt.plot(tr_losses, color = 'blue', alpha = .225)



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
plt.show()








model_param_viz = {'conv_neurons': [32, 16],
                       'conv_dropout': .15, 
                       'pool_sizes': (2,2),
                       'merged_dropout': .15,
                       'output_dropout': .15,
                       'feature_neurons': 25, 
                       'output_neurons': [15],
                       'merged_neurons': [25, 25, 25],
                       'lr': .001}
model_viz = build_cnn(train_img, train_feat, train_y,  **model_param_viz)
plot_model(model_viz)

