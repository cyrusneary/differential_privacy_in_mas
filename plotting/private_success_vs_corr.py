import matplotlib.pyplot as plt
import numpy as np

import os, sys, time
sys.path.append('..')

import tikzplotlib

import pickle

# Plotting parameters
fontsize = 12
linewidth = 3
markersize = 15
num_data_points = 80

tikz_save_path = os.path.abspath(os.path.join(os.path.curdir, 'tikz'))

base_path = os.path.abspath(os.path.join(os.path.curdir, '..', 'experiments', 'results'))

save_file_name = '2023-01-11-14-12-04_ma_gridworld_minimum_dependency_0p05.pkl'

save_str = os.path.join(base_path, save_file_name)

with open(save_str, 'rb') as f:
    exp_logger = pickle.load(f)

success_prob_reachability = exp_logger['max_reachability_results']['success_prob'] * np.ones((num_data_points,))
empirical_private_reachability = exp_logger['max_reachability_results']['empirical_private_success_rate']
# print(exp_logger['max_reachability_results'].keys())
total_corr_reachability = exp_logger['max_reachability_results']['total_corr_reach']

# Get the relevant data in numpy format
iters_indexes = exp_logger['results'].keys()

total_corr = []
success_prob = []
empirical_private = []
iters = []
for key in range(num_data_points):
    total_corr.append(exp_logger['results'][key]['total_corr'])
    success_prob.append(exp_logger['results'][key]['success_prob'])
    empirical_private.append(exp_logger['results'][key]['empirical_private_success_rate'])
    iters.append(key)

bound = np.array(success_prob) - np.sqrt(1 - np.exp(-np.array(total_corr)))

# Plot 
fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(total_corr, empirical_private,
            color='blue', linewidth=linewidth,
            label=r'Minimal-Dependency Policy')
# ax.plot(total_corr_reachability, empirical_private_reachability,
#             color='red', marker='x', markersize=markersize,
#             label='Baseline Policy')

ax.grid()
# ax.set_ylabel('Probability of Team Success', fontsize=fontsize)
# ax.set_xlabel('Total Correlation of the Synthesized Minimum-Dependency Policy', fontsize=fontsize)
# ax.set_title('Success Probability of Privatized Policy Execution ($\epsilon = 1$) \n as a Function of the Total Correlation of the Joint Policy', fontsize=fontsize)
# plt.legend(fontsize=fontsize)

# tikz_file_str = os.path.join(tikz_save_path, 'success_prob_vs_corr_two_agent_navigation.tex')
# tikzplotlib.save(tikz_file_str)

plt.show()