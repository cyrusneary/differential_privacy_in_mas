import matplotlib.pyplot as plt
import numpy as np

import os, sys, time
sys.path.append('..')

import tikzplotlib

import pickle

# Plotting parameters
fontsize = 12
num_data_points = 80 # 51

tikz_save_path = os.path.abspath(os.path.join(os.path.curdir, 'tikz'))

base_path = os.path.abspath(os.path.join(os.path.curdir, '..', 'experiments', 'results'))
# save_file_name = '2022-12-21-10-14-18_ma_gridworld_total_corr_add_end_state_0p05.pkl'
save_file_name = '2022-12-22-19-05-27_ma_gridworld_minimum_dependency_0p05.pkl'
save_file_name = '2023-01-10-17-40-25_sys_admin_minimum_dependency.pkl'
save_file_name = '2023-01-11-14-12-04_ma_gridworld_minimum_dependency_0p05.pkl'
save_str = os.path.join(base_path, save_file_name)

with open(save_str, 'rb') as f:
    exp_logger = pickle.load(f)

success_prob_reachability = exp_logger['max_reachability_results']['success_prob'] * np.ones((num_data_points,))
empirical_imag_reachability = exp_logger['max_reachability_results']['empirical_imag_success_rate'] * np.ones((num_data_points,))

# Get the relevant data in numpy format
iters_indexes = exp_logger['results'].keys()

total_corr = []
success_prob = []
empirical_imag = []
iters = []
for key in range(num_data_points):
    total_corr.append(exp_logger['results'][key]['total_corr'])
    success_prob.append(exp_logger['results'][key]['success_prob'])
    empirical_imag.append(exp_logger['results'][key]['empirical_imag_success_rate'])
    iters.append(key)

bound = np.array(success_prob) - np.sqrt(1 - np.exp(-np.array(total_corr)))

# Plot the total correlation as a function of optimization iterations
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(iters, total_corr,
        color='blue', marker='.')
# ax.plot(iters, exp_logger_reachability.results[0]['total_corr'] * np.ones(len(iters)), color='magenta')
ax.grid()
ax.set_ylabel('Total Correlation Value', fontsize=fontsize)
ax.set_xlabel('Number of Convex-Concave Iterations', fontsize=fontsize)
ax.set_title('Total Correlation During Policy Synthesis', fontsize=fontsize)

tikz_file_str = os.path.join(tikz_save_path, 'corr_vs_iters_three_agents.tex')
tikzplotlib.save(tikz_file_str)

# plt.show()

# # Plot 
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(iters, success_prob, 
#             color='blue', marker='.', label='Success Probability [Full Comms]')
# ax.plot(iters, success_prob_reachability,
#             color='red', marker='.', 
#             label='Success Probability [Full Comms], \n Optimal Reachability Policy')
# ax.plot(iters, bound,
#             color='black', 
#             label='Theoretical Lower Bound on Success Probability of Imaginary Play')
# ax.plot(iters, empirical_imag,
#             color='blue', linestyle='--', 
#             label='Empirically Measured Success Probability of Imaginary Play')
# ax.plot(iters, empirical_imag_reachability,
#             color='red', linestyle='--',
#             label='Empirically Measured Success Probability of Imaginary Play \n Optimal Reachability Policy')
# ax.grid()
# ax.set_ylabel('Total Correlation Value', fontsize=fontsize)
# ax.set_xlabel('Number of Convex-Concave Iterations', fontsize=fontsize)
# ax.set_title('Policy Success Probability During Synthesis', fontsize=fontsize)
# plt.legend(fontsize=fontsize)

# # tikz_file_str = os.path.join(tikz_save_path, 'success_prob_vs_iters_three_agent_aux_action.tex')
# # tikzplotlib.save(tikz_file_str)

# plt.show()