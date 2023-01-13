import matplotlib.pyplot as plt
import numpy as np

import os, sys, time
sys.path.append('..')

import tikzplotlib
import pickle

from environments.sys_admin import SysAdmin
from environments.trajectory_runners import empirical_success_rate_private, empirical_success_rate

from optimization_problems.minimum_dependency_policy import \
    marginalize_policy, compute_product_policy, kl_divergence_policies, kl_divergence_joint_and_marginalized_policies

# Plotting parameters
fontsize = 12
linewidth = 3
markersize = 15

tikz_save_path = os.path.abspath(os.path.join(os.path.curdir, 'tikz'))

##### Load the saved experiment file
save_str_list = []
base_path = os.path.abspath(os.path.join(os.path.curdir, '..', 'experiments', 'results'))
save_str_list.append(os.path.join(base_path, '2023-01-13-16-19-38_sys_admin_minimum_dependency_0001.pkl'))
save_str_list.append(os.path.join(base_path, '2023-01-13-16-19-38_sys_admin_minimum_dependency_1111.pkl'))
save_str_list.append(os.path.join(base_path, '2023-01-13-16-19-38_sys_admin_minimum_dependency_0022.pkl'))
save_str_list.append(os.path.join(base_path, '2023-01-13-16-19-38_sys_admin_minimum_dependency_0033.pkl'))
save_str_list.append(os.path.join(base_path, '2023-01-13-16-19-38_sys_admin_minimum_dependency_2233.pkl'))

exp_logger_list = []
for save_str in save_str_list:
    with open(save_str, 'rb') as f:
        exp_logger_list.append(pickle.load(f))

results = {}

##### Create the gridworld from the logged parameters

t_start = time.time()

if exp_logger_list[0]['environment_type'] == 'sys_admin':
    env = SysAdmin(**exp_logger_list[0]['environment_settings'])
    print('Constructed the sys admin environment in {} seconds.'.format(time.time() - t_start))

mdp = env.build_mdp()
N_agents = exp_logger_list[0]['environment_type']['N_agents']

# Iterate through the different trials

epsilon_list = [0.1, 1.0, 5.0, 10.0]
md_success_probs = []
base_success_probs = []
md_success_probs_marginalized_policies = []
base_success_probs_marginalized_policies = []

md_policy = exp_logger['results'][max(exp_logger['results'].keys())]['policy']
base_policy = exp_logger['max_reachability_results']['policy']

marginalized_md_policy_list = marginalize_policy(md_policy, mdp, N_agents)
marginalized_base_policy_list = marginalize_policy(base_policy, mdp, N_agents)

# md_policy_kl = kl_divergence_joint_and_marginalized_policies(md_policy, marginalized_md_policy_list, mdp, N_agents)
# base_policy_kl = kl_divergence_joint_and_marginalized_policies(base_policy, marginalized_base_policy_list, mdp, N_agents)

# print("KL divergence between the minimum dependency policy and the marginalized policies: {}".format(md_policy_kl))
# print("KL divergence between the baseline policy and the marginalized policies: {}".format(base_policy_kl))

# for e in epsilon_list:
        
#     md_success_prob_marginalized_policies = \
#         empirical_success_rate_private(gridworld, marginalized_md_policy_list,
#                                             num_trajectories=1000,
#                                             max_steps_per_trajectory=200,
#                                             epsilon=e, 
#                                             k=exp_logger['empirical_eval_settings']['adjacency_parameter'],
#                                             use_marginalized_policies=True)
#     base_success_prob_marginalized_policies = \
#         empirical_success_rate_private(gridworld, marginalized_base_policy_list,
#                                             num_trajectories=1000,
#                                             max_steps_per_trajectory=200, 
#                                             epsilon=e, 
#                                             k=exp_logger['empirical_eval_settings']['adjacency_parameter'],
#                                             use_marginalized_policies=True)

#     print('Finished simulating $\epsilon$ = {}'.format(e))

#     # md_success_probs.append(md_success_prob)
#     # base_success_probs.append(base_success_prob)
#     md_success_probs_marginalized_policies.append(md_success_prob_marginalized_policies)
#     base_success_probs_marginalized_policies.append(base_success_prob_marginalized_policies)

# tikz_file_str = os.path.join(tikz_save_path, 'success_vs_privacy_parameter_two_agent_navigation.tex')
# tikzplotlib.save(tikz_file_str)
