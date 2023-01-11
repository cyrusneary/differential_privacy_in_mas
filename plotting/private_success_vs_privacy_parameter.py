import matplotlib.pyplot as plt
import numpy as np

import os, sys, time
sys.path.append('..')

import tikzplotlib
import pickle

from environments.ma_gridworld import MAGridworld
from environments.sys_admin import SysAdmin
from environments.trajectory_runners import empirical_success_rate_private

from optimization_problems.minimum_dependency_policy import \
    marginalize_policy, compute_product_policy, kl_divergence_policies, kl_divergence_joint_and_marginalized_policies

# Plotting parameters
fontsize = 12
linewidth = 3
markersize = 15

tikz_save_path = os.path.abspath(os.path.join(os.path.curdir, 'tikz'))

##### Load the saved experiment file

base_path = os.path.abspath(os.path.join(os.path.curdir, '..', 'experiments', 'results'))
# save_file_name = '2022-12-23-12-02-14_ma_gridworld_minimum_dependency_0p05.pkl'
save_file_name = '2022-12-27-18-59-30_ma_gridworld_minimum_dependency_0p05.pkl'
save_file_name = '2023-01-04-17-25-06_ma_gridworld_minimum_dependency_0p05.pkl'
save_file_name = '2023-01-07-12-48-55_ma_gridworld_minimum_dependency_0p05.pkl'
save_file_name = '2023-01-10-17-40-25_sys_admin_minimum_dependency.pkl'
save_str = os.path.join(base_path, save_file_name)

with open(save_str, 'rb') as f:
    exp_logger = pickle.load(f)

##### Create the gridworld from the logged parameters

t_start = time.time()

if exp_logger['environment_type'] == 'sys_admin':
    pass

elif exp_logger['environment_type'] == 'ma_gridworld':
    gridworld = MAGridworld(**exp_logger['environment_settings'])
    print('Constructed the gridworld in {} seconds.'.format(time.time() - t_start))

epsilon_list = np.linspace(0.01, 10.0, num=11)
md_success_probs = []
base_success_probs = []
md_success_probs_marginalized_policies = []
base_success_probs_marginalized_policies = []

md_policy = exp_logger['results'][max(exp_logger['results'].keys())]['policy']
base_policy = exp_logger['max_reachability_results']['policy']

mdp = gridworld.build_mdp()
N_agents = exp_logger['environment_settings']['N_agents']

marginalized_md_policy_list = marginalize_policy(md_policy, mdp, N_agents)
marginalized_base_policy_list = marginalize_policy(base_policy, mdp, N_agents)

md_policy_kl = kl_divergence_joint_and_marginalized_policies(md_policy, marginalized_md_policy_list, mdp, N_agents)
base_policy_kl = kl_divergence_joint_and_marginalized_policies(base_policy, marginalized_base_policy_list, mdp, N_agents)

print("KL divergence between the minimum dependency policy and the marginalized policies: {}".format(md_policy_kl))
print("KL divergence between the baseline policy and the marginalized policies: {}".format(base_policy_kl))

for e in epsilon_list:

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
        
    md_success_prob_marginalized_policies = \
        empirical_success_rate_private(gridworld, marginalized_md_policy_list,
                                            num_trajectories=1000,
                                            max_steps_per_trajectory=200,
                                            epsilon=e, k=3,
                                            use_marginalized_policies=True)
    base_success_prob_marginalized_policies = \
        empirical_success_rate_private(gridworld, marginalized_base_policy_list,
                                            num_trajectories=1000,
                                            max_steps_per_trajectory=200, 
                                            epsilon=e, k=3,
                                            use_marginalized_policies=True)

    print('Finished simulating $\epsilon$ = {}'.format(e))

    md_success_probs.append(md_success_prob)
    base_success_probs.append(base_success_prob)
    md_success_probs_marginalized_policies.append(md_success_prob_marginalized_policies)
    base_success_probs_marginalized_policies.append(base_success_prob_marginalized_policies)

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
ax.plot(epsilon_list, md_success_probs_marginalized_policies, 
        color='blue', 
        marker='d', 
        linewidth=linewidth, 
        markersize=markersize, 
        label='Minimum Dependency Policy Marginalized Policies')
ax.plot(epsilon_list, base_success_probs_marginalized_policies, 
        color='red', 
        marker='d', 
        linewidth=linewidth, 
        markersize=markersize,
        label='Baseline Policy Marginalized Policies')
# ax.plot(epsilon_list, lower_bound_list, color='black')

ax.grid()
ax.set_xlabel('Privacy Parameter $\epsilon$', fontsize=fontsize)
ax.set_ylabel('Task Success Probability', fontsize=fontsize)
ax.legend(fontsize=fontsize)

# tikz_file_str = os.path.join(tikz_save_path, 'plot_intermittent_play_data_aux_action.tex')
# tikzplotlib.save(tikz_file_str)

plt.show()