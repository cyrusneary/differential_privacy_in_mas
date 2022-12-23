import matplotlib.pyplot as plt
import numpy as np

import os, sys, time
sys.path.append('..')

import tikzplotlib
import pickle

from environments.ma_gridworld import MAGridworld
from environments.trajectory_runners import empirical_success_rate_private

# Plotting parameters
fontsize = 12
linewidth = 3
markersize = 15

tikz_save_path = os.path.abspath(os.path.join(os.path.curdir, 'tikz'))

##### Load the saved experiment file

base_path = os.path.abspath(os.path.join(os.path.curdir, '..', 'experiments', 'results'))
save_file_name = '2022-12-23-12-02-14_ma_gridworld_minimum_dependency_0p05.pkl'
save_str = os.path.join(base_path, save_file_name)

with open(save_str, 'rb') as f:
    exp_logger = pickle.load(f)

##### Create the gridworld from the logged parameters

t_start = time.time()
gridworld = MAGridworld(**exp_logger['environment_settings'])
print('Constructed the gridworld in {} seconds.'.format(time.time() - t_start))

epsilon_list = np.linspace(0.01, 10.0, num=11)
md_success_probs = []
base_success_probs = []
lower_bound_list = []

for e in epsilon_list:
    md_policy = exp_logger['results'][max(exp_logger['results'].keys())]['policy']
    base_policy = exp_logger['max_reachability_results']['policy']

    md_policy_val = exp_logger['results'][max(exp_logger['results'].keys())]['success_prob']
    md_total_corr = exp_logger['results'][max(exp_logger['results'].keys())]['total_corr']

    md_success_prob = \
        empirical_success_rate_private(gridworld, md_policy,
                                            num_trajectories=1000,
                                            max_steps_per_trajectory=200,
                                            epsilon=e, k=3)
    base_success_prob = \
        empirical_success_rate_private(gridworld, base_policy,
                                            num_trajectories=1000,
                                            max_steps_per_trajectory=200, 
                                            epsilon=e, k=3)
    # lower_bound = md_policy_val - np.sqrt(1 - np.exp(-e * md_total_corr))

    print('Finished simulating $\epsilon$ = {}'.format(e))

    md_success_probs.append(md_success_prob)
    base_success_probs.append(base_success_prob)
    # lower_bound_list.append(lower_bound)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(epsilon_list, md_success_probs, 
        color='blue', 
        marker='.', 
        linewidth=linewidth, 
        markersize=markersize, 
        label='Minimum Dependency Policy')
ax.plot(epsilon_list, base_success_probs, 
        color='red', 
        marker='.', 
        linewidth=linewidth, 
        markersize=markersize,
        label='Baseline Policy')
# ax.plot(epsilon_list, lower_bound_list, color='black')

ax.grid()
ax.set_xlabel('Privacy Parameter $\epsilon$', fontsize=fontsize)
ax.set_ylabel('Task Success Probability', fontsize=fontsize)
ax.legend(fontsize=fontsize)

# tikz_file_str = os.path.join(tikz_save_path, 'plot_intermittent_play_data_aux_action.tex')
# tikzplotlib.save(tikz_file_str)

plt.show()