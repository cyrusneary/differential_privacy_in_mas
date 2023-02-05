import matplotlib.pyplot as plt
import numpy as np

import os, sys, time
sys.path.append('..')

import tikzplotlib
import pickle

from tqdm import tqdm

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
save_str_list.append(os.path.join(base_path, '2023-01-13-16-11-11_sys_admin_minimum_dependency_0001.pkl'))
save_str_list.append(os.path.join(base_path, '2023-01-13-16-16-51_sys_admin_minimum_dependency_1111.pkl'))
save_str_list.append(os.path.join(base_path, '2023-01-13-16-14-02_sys_admin_minimum_dependency_0022.pkl'))
save_str_list.append(os.path.join(base_path, '2023-01-13-16-09-07_sys_admin_minimum_dependency_0033.pkl'))
save_str_list.append(os.path.join(base_path, '2023-01-13-16-19-38_sys_admin_minimum_dependency_2233.pkl'))

exp_loggers = {}
results = {}
for save_str in save_str_list:
    with open(save_str, 'rb') as f:
        exp_loggers[save_str[save_str.index('.pkl') - 4 : save_str.index('.pkl')]] = pickle.load(f)
        results[save_str[save_str.index('.pkl') - 4 : save_str.index('.pkl')]] = {'md' : [], 'base' : []}

##### Create the gridworld from the logged parameters

t_start = time.time()

if exp_loggers[list(exp_loggers.keys())[0]]['environment_type'] == 'sys_admin':
    env = SysAdmin(**exp_loggers[list(exp_loggers.keys())[0]]['environment_settings'])
    print('Constructed the sys admin environment in {} seconds.'.format(time.time() - t_start))

mdp = env.build_mdp()
N_agents = exp_loggers[list(exp_loggers.keys())[0]]['environment_settings']['N_agents']

# Iterate through the different trials

epsilon_list = [0.1, 1.0, 10.0]

num_trajectories = 1000
max_steps_per_trajectory = 1000

for trial in tqdm(exp_loggers.keys()):
    
    md_policy = exp_loggers[trial]['results'][max(exp_loggers[trial]['results'].keys())]['policy']
    base_policy = exp_loggers[trial]['max_reachability_results']['policy']

    marginalized_md_policy_list = marginalize_policy(md_policy, mdp, N_agents)
    marginalized_base_policy_list = marginalize_policy(base_policy, mdp, N_agents)

    for e in epsilon_list:
        # Compute the empirical success rate for the minimum dependency policy
        md_success_prob = empirical_success_rate_private(
                            env, 
                            marginalized_md_policy_list,
                            num_trajectories=num_trajectories,
                            max_steps_per_trajectory=max_steps_per_trajectory,
                            epsilon=e, 
                            k=exp_loggers[trial]['empirical_eval_settings']['adjacency_parameter'],
                            use_marginalized_policies=True,
                        )
        # Compute the empirical success rate for the baseline policy
        base_success_prob = empirical_success_rate_private(
                                env, 
                                marginalized_base_policy_list,
                                num_trajectories=num_trajectories,
                                max_steps_per_trajectory=max_steps_per_trajectory, 
                                epsilon=e, 
                                k=exp_loggers[trial]['empirical_eval_settings']['adjacency_parameter'],
                                use_marginalized_policies=True,
                            )
        results[trial]['md'].append(md_success_prob)
        results[trial]['base'].append(base_success_prob)

    results[trial]['md'].append(exp_loggers[trial]['results'][max(exp_loggers[trial]['results'].keys())]['success_prob'])
    results[trial]['base'].append(exp_loggers[trial]['max_reachability_results']['success_prob'])

# Now plot the results in a nice bar chart

with open('bar_chart_data.pkl', 'wb') as f:
    pickle.dump(results, f)