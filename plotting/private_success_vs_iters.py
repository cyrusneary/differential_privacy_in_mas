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
# save_file_name = '2022-12-23-12-02-14_ma_gridworld_minimum_dependency_0p05.pkl'
# save_file_name = '2022-12-27-18-59-30_ma_gridworld_minimum_dependency_0p05.pkl'
# save_file_name = '2023-01-05-21-30-29_ma_gridworld_minimum_dependency_0p05.pkl'
# save_file_name = '2023-01-05-22-05-14_ma_gridworld_minimum_dependency_0p05.pkl'
# save_file_name = '2023-01-07-12-48-55_ma_gridworld_minimum_dependency_0p05.pkl'
# save_file_name = '2023-01-10-17-40-25_sys_admin_minimum_dependency.pkl'
# save_file_name = '2023-01-11-14-12-04_ma_gridworld_minimum_dependency_0p05.pkl'
save_file_name = '2023-01-11-14-12-04_ma_gridworld_minimum_dependency_0p05.pkl'
save_str = os.path.join(base_path, save_file_name)

with open(save_str, 'rb') as f:
    exp_logger = pickle.load(f)

success_prob_reachability = exp_logger['max_reachability_results']['success_prob'] * np.ones((num_data_points,))
empirical_private_reachability = exp_logger['max_reachability_results']['empirical_private_success_rate'] * np.ones((num_data_points,))

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
ax.plot(iters, success_prob, 
            color='blue', marker='.', linewidth=linewidth, markersize=markersize,
            label='No Privacy -- Minimal-Dependency Policy')
ax.plot(iters, success_prob_reachability,
            color='red', marker='.', linewidth=linewidth, markersize=markersize,
            label='No privacy -- Baseline Policy')

ax.plot(iters, empirical_private,
            color='blue', linestyle='-', linewidth=linewidth,
            label=r'Privatized Policy Execution ($\epsilon = 1$) -- Minimal-Dependency Policy')
ax.plot(iters, empirical_private_reachability,
            color='red', linestyle='-', linewidth=linewidth,
            label='Privatized Policy Execution ($\epsilon = 1$) -- Baseline Policy')

# ax.plot(iters, bound,
#             color='black', 
#             label='Theoretical Lower Bound on Success Probability of Privatized Policy Execution')

print('MD Policy no privacy: {}'.format(np.max(success_prob)))
print('MD Policy with privacy: {}'.format(np.max(empirical_private)))

print('baseline policy no privacy: {}'.format(np.max(success_prob_reachability)))
print('baseline policy with privacy: {}'.format(np.max(empirical_private_reachability)))

# ax.grid()
# ax.set_ylabel('Probability of Team Success', fontsize=fontsize)
# ax.set_xlabel('Number of Iterations of Policy Synthesis Algorithm', fontsize=fontsize)
# plt.legend(fontsize=fontsize)

tikz_file_str = os.path.join(tikz_save_path, 'success_prob_vs_iters_two_agent_navigation.tex')
tikzplotlib.save(tikz_file_str)

# plt.show()