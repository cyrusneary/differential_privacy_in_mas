from cvxpy.error import SolverError
import numpy as np
import datetime
import sys, os, time
import pickle

sys.path.append('../')

from environments.ma_gridworld import MAGridworld
from environments.sys_admin import SysAdmin

from optimization_problems.minimum_dependency_policy import *
from optimization_problems.max_entropy_policy import build_joint_entropy_program
from optimization_problems.max_reachability_policy import build_reachability_LP
from optimization_problems.random_policy import build_random_policy_program

from environments.trajectory_runners import empirical_success_rate, empirical_success_rate_private
from environments.environment_factory import get_environment

from markov_decision_process.policies import JointPolicy, LocalPolicies, \
    LocalPoliciesAcyclicDependencies

##########################
#### Experimental settings
##########################

# Set the configuration file to load
config_file = 'ma_gridworld_config'

if config_file == 'ma_gridworld_config':
    from experiments.configurations.ma_gridworld_config import exp_logger
elif config_file == 'sysadmin_config':
    from experiments.configurations.sys_admin_config import exp_logger

# Use the current date and time to make a unique experiment name.
curr_datetime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
exp_name = curr_datetime + '_' + exp_logger['exp_name']
print(exp_name)

exp_logger['experiment_name'] = exp_name

###########################################
#### Construct the experimental environment
###########################################

# Build the gridworld
print('Building environment')
t_start = time.time()

env = get_environment(exp_logger['environment_type'], exp_logger['environment_settings'])

print('Constructed environment in {} seconds'.format(time.time() - t_start))

# Sanity check on the transition matrix
for s in range(env.Ns_joint):
    for a in range(env.Na_joint):
        assert(np.abs(np.sum(env.T[s, a, :]) - 1.0) <= 1e-12)

# Save the constructed gridworld
save_file_str = os.path.join(os.path.abspath(os.path.curdir),
                                '..', 'environments',
                                'saved_environments', 
                                exp_logger['environment_type'] + '.pkl')
env.save(save_file_str)

# Construct the corresponding MDP
mdp = env.build_mdp()

#####################################################
#### Construct the optimziation problems of interest.
#####################################################

# Construct an easier policy synthesis optimization problem to use to get an
# initial guess for the solution.
if exp_logger['initial_soln_guess_setup']['type'] == 'random':
    # Construct and solve for a random initial policy
    rand_init = np.random.rand(mdp.Ns, (env.Na_local + 1)**env.N_agents) * 20
    prob_init, vars_init, _ = build_random_policy_program(mdp, 
                                                        rand_init=rand_init,
                                                        N_agents=exp_logger['environment_settings']['N_agents'])
elif exp_logger['initial_soln_guess_setup']['type'] == 'entropy':
    # Construct and solve for an initial policy maximizing entropy
    # exp_len_coef = 0.0, entropy_coef = 0.01
    prob_init, vars_init, _ = build_joint_entropy_program(mdp, env.N_agents,
                        **exp_logger['initial_soln_guess_setup']['settings'])
elif exp_logger['initial_soln_guess_setup']['type'] == 'reachability':
    # Construct and solve the reachability LP
    # exp_len_coef = 0.01
    prob_init, vars_init, params_init = build_reachability_LP(mdp, 
                        **exp_logger['initial_soln_guess_setup']['settings'])

# Construct the minimum-dependency policy synthesis optimization problem
agent_state_size_list = []
agent_action_size_list = []
for agent_id in range(env.N_agents):
    agent_state_size_list.append(env.Ns_local)
    agent_action_size_list.append(env.Na_local)

t = time.time()
prob, vars, params, f_grad, g_grad = \
    build_linearized_program(mdp, 
        env.N_agents,
        agent_state_size_list,
        agent_action_size_list,
        env.check_agent_state_action_with_aux,
        env.check_agent_state,
        reachability_coef=exp_logger['optimization_params']['reachability_coef'],
        exp_len_coef=exp_logger['optimization_params']['exp_len_coef'],
        total_corr_coef=exp_logger['optimization_params']['total_corr_coef'])
print('Constructed optimization problem in {} seconds.'.format(time.time()-t)) 

#######################################################
##### Solve the initial problem to get an initial guess
#######################################################

t = time.time()
# prob_init.solve(verbose=True, solver='ECOS')
prob_init.solve(verbose=True)
print('Solved for initial guess in {} seconds'.format(time.time() - t)) 

# Save the initial policy and its statistics.
occupancy_vars_start = process_occupancy_vars(vars_init[0])
policy_start = JointPolicy(mdp, env.N_agents, occupancy_vars_start)
success_prob = success_probability_from_occupancy_vars(mdp, occupancy_vars_start, env.N_agents)
expected_len = expected_len_from_occupancy_vars(mdp, occupancy_vars_start)
joint_entropy = compute_joint_entropy(mdp, occupancy_vars_start, env.N_agents)
total_corr = compute_total_correlation(mdp,
                            N_agents=env.N_agents,
                            agent_state_size_list=agent_state_size_list,
                            agent_action_size_list=agent_action_size_list,
                            f_grad=f_grad,
                            g_grad=g_grad,
                            x=occupancy_vars_start)

exp_logger['results'][-1] = {
    'occupancy_vars' : occupancy_vars_start,
    'policy' : policy_start,
    'opt_val' : prob_init.solution.opt_val,
    'success_prob' : success_prob,
    'expected_len' : expected_len,
    'joint_entropy' : joint_entropy,
    'total_corr' : total_corr,
}

print(('Success probability: {}, \n \
        expected length: {}, \n \
        joint entropy: {}\n \
        total correlation: {}'.format(success_prob, 
                                    expected_len, 
                                    joint_entropy, 
                                    total_corr)))

x_start = occupancy_vars_start

##################################################################
###### Solve for the (baseline) maximum reachability joint policy.
##################################################################

# Set the parameters in the optimization objective to ignore 
# total correlation and expected trajectory length.
params[0].value = 1.0
params[1].value = 0.0
params[2].value = 0.0
params[3].value = x_start
 
t = time.time()
# Use ECOS instead of MOSEK to solve the pure reachability LP.
prob.solve(verbose=True, solver='ECOS') 
print('Solved for maximum reachability policy in {} seconds'.format(time.time() - t)) 

# Save the max reachability policy and its statistics.
occupancy_vars_reach = process_occupancy_vars(vars[0])
policy_reach = JointPolicy(mdp, env.N_agents, occupancy_vars_reach)
success_prob_reach = success_probability_from_occupancy_vars(mdp, occupancy_vars_reach, env.N_agents)
expected_len_reach = expected_len_from_occupancy_vars(mdp, occupancy_vars_reach)
joint_entropy_reach = compute_joint_entropy(mdp, occupancy_vars_reach, env.N_agents)
total_corr_reach = compute_total_correlation(mdp,
                            N_agents=env.N_agents,
                            agent_state_size_list=agent_state_size_list,
                            agent_action_size_list=agent_action_size_list,
                            f_grad=f_grad,
                            g_grad=g_grad,
                            x=occupancy_vars_reach)

# Empirically test the success rate during truthful communication.
empirical_rate_reach = empirical_success_rate(
    env,
    policy_reach,
    num_trajectories=exp_logger['empirical_eval_settings']['num_trajectories'],
    max_steps_per_trajectory=exp_logger['empirical_eval_settings']['max_steps_per_trajectory']
)

# Get the marginalized policies to test private communication.
if exp_logger['empirical_eval_settings']['policy_type'] == 'local':
    eval_policy = LocalPolicies(mdp, env.N_agents, x=occupancy_vars_reach)
    policy_kl = eval_policy.kl_divergence_joint_and_marginalized_policies(policy_reach)
elif exp_logger['empirical_eval_settings']['policy_type'] == 'joint':
    eval_policy = policy_reach
elif exp_logger['empirical_eval_settings']['policy_type'] == 'acyclic':
    eval_policy = LocalPoliciesAcyclicDependencies(
            mdp, 
            env.N_agents, 
            exp_logger['empirical_eval_settings']['dependency_structure'],
            x=occupancy_vars_reach,
        )
  
# Empirically test the success rate during private communication.  
private_rate_reach = empirical_success_rate_private(
        env,
        eval_policy,
        num_trajectories=exp_logger['empirical_eval_settings']['num_trajectories'],
        max_steps_per_trajectory=exp_logger['empirical_eval_settings']['max_steps_per_trajectory'],
        epsilon=exp_logger['empirical_eval_settings']['privacy_parameter'],
        k=exp_logger['empirical_eval_settings']['adjacency_parameter'],
        policy_type=exp_logger['empirical_eval_settings']['policy_type'],
    )

# Save the results
exp_logger['max_reachability_results'] = {
    'occupancy_vars' : occupancy_vars_reach,
    'policy' : policy_reach,
    'eval_policy' : eval_policy,
    'opt_val' : prob.solution.opt_val,
    'success_prob' : success_prob_reach,
    'expected_len' : expected_len_reach,
    'joint_entropy' : joint_entropy_reach,
    'total_corr_reach' : total_corr_reach,
    'empirical_imag_success_rate' : empirical_rate_reach,
    'empirical_private_success_rate' : private_rate_reach,
}

if exp_logger['empirical_eval_settings']['policy_type'] == 'local':
    exp_logger['max_reachability_results']['policy_kl'] = policy_kl
    print('KL divergence between joint and marginalized policies: {}'.format(policy_kl))

# Print the results to the terminal
print(('Success probability: {}, \n \
        Imaginary Play success prob: {}, \n \
        Private play success prob: {}, \n \
        expected length: {}, \n \
        total correlation: {}, \n \
        joint entropy: {}'.format(success_prob_reach, 
                                  empirical_rate_reach,
                                  private_rate_reach,
                                    expected_len_reach, 
                                    total_corr_reach,
                                    joint_entropy_reach)))

# Set the parameters back to the original values.
params[0].value = exp_logger['optimization_params']['reachability_coef']
params[1].value = exp_logger['optimization_params']['exp_len_coef']
params[2].value = exp_logger['optimization_params']['total_corr_coef']

########################################################################################
##### Solve the minimum-dependency policy synthesis problem via convex-concave procedure
########################################################################################

x_last = x_start
for i in range(80):
    params[3].value = x_last
    # prob.solve(verbose=True, solver='ECOS') 
    prob.solve(verbose=False, solver='MOSEK')

    # Compute the results of the current iteration
    occupancy_vars = process_occupancy_vars(vars[0])
    policy = JointPolicy(mdp, env.N_agents, occupancy_vars)
    success_prob = success_probability_from_occupancy_vars(mdp, occupancy_vars, env.N_agents)
    expected_len = expected_len_from_occupancy_vars(mdp, occupancy_vars)
    joint_entropy = compute_joint_entropy(mdp, occupancy_vars, env.N_agents)
    total_corr = compute_total_correlation(mdp,
                                N_agents=env.N_agents,
                                agent_state_size_list=agent_state_size_list,
                                agent_action_size_list=agent_action_size_list,
                                f_grad=f_grad,
                                g_grad=g_grad,
                                x=occupancy_vars)
    
    # Empirically test the success rate of the joint policy under truthful communication.
    empirical_rate = empirical_success_rate(
        env,
        policy,
        num_trajectories=exp_logger['empirical_eval_settings']['num_trajectories'],
        max_steps_per_trajectory=exp_logger['empirical_eval_settings']['max_steps_per_trajectory']
    )
    
    # Now get the marginalized policies to run private play.
    if exp_logger['empirical_eval_settings']['policy_type'] == 'local':
        eval_policy = LocalPolicies(mdp, env.N_agents, x=occupancy_vars)
        policy_kl = eval_policy.kl_divergence_joint_and_marginalized_policies(policy)
    elif exp_logger['empirical_eval_settings']['policy_type'] == 'joint':
        eval_policy = policy
    elif exp_logger['empirical_eval_settings']['policy_type'] == 'acyclic':
        eval_policy = LocalPoliciesAcyclicDependencies(
                mdp, 
                env.N_agents, 
                exp_logger['empirical_eval_settings']['dependency_structure'],
                x=occupancy_vars,
            )

    # Empirically test the success rate of the policies under private communication.
    private_rate = empirical_success_rate_private(
        env,
        eval_policy,
        num_trajectories=exp_logger['empirical_eval_settings']['num_trajectories'],
        max_steps_per_trajectory=exp_logger['empirical_eval_settings']['max_steps_per_trajectory'],
        epsilon=exp_logger['empirical_eval_settings']['privacy_parameter'],
        k=exp_logger['empirical_eval_settings']['adjacency_parameter'],
        policy_type=exp_logger['empirical_eval_settings']['policy_type'],
    )

    # Save the results of this iteration
    exp_logger['results'][i] = {
        'occupancy_vars' : occupancy_vars,
        'policy' : policy,
        'eval_policy' : eval_policy,
        'opt_val' : prob.solution.opt_val,
        'x_last' : x_last,
        'success_prob' : success_prob,
        'expected_len' : expected_len,
        'joint_entropy' : joint_entropy,
        'total_corr' : total_corr,
        'empirical_imag_success_rate' : empirical_rate,
        'empirical_private_success_rate' : private_rate,
    }
    
    if exp_logger['empirical_eval_settings']['policy_type'] == 'local':
        exp_logger['results'][i]['policy_kl'] = policy_kl        
        print('KL divergence between joint and marginalized policies: {}'.format(policy_kl))
    
    # Print the results to the terminal
    print('\n [{}]: Success probability: {}, Expected length: {}, total correlation: {}'.format(
                                    i, 
                                    exp_logger['results'][i]['success_prob'],
                                    exp_logger['results'][i]['expected_len'],
                                    exp_logger['results'][i]['total_corr']
                                )
        )
    
    print('Private Play success prob: {}'.format(exp_logger['results'][i]['empirical_private_success_rate']))
    
    print('Imaginary Play success prob: {}'.format(
        exp_logger['results'][i]['empirical_imag_success_rate']))
    
    print('TC: {}'.format(vars[1].value))

    print('KL divergence with marginalized policies: {}'.format(policy_kl))
    
    print('||x - x_last||: {}'.format(
        np.linalg.norm(
            exp_logger['results'][i]['occupancy_vars'] - \
                exp_logger['results'][i]['x_last']
            )
        )
    )
    
    print('Grad: {}'.format(np.linalg.norm(vars[2].value)))
    
    print('Expected Difference: {}'.format(
        np.sum(
            np.multiply(
                vars[2].value, exp_logger['results'][i]['occupancy_vars'] - \
                    exp_logger['results'][i]['x_last']))))
    
    print('Convex term: {}'.format(-vars[3].value))
    
    if i >= 1:
        print('Last total_corr: {}'.format(-vars[3].value - exp_logger['results'][i-1]['joint_entropy']))

    x_last = occupancy_vars

    save_folder_str = os.path.join(os.path.abspath(os.path.curdir), 'results')
    save_file_name = exp_logger['experiment_name'] + '.pkl'
    save_str = os.path.abspath(os.path.join(save_folder_str, 
                                                    save_file_name))   
    exp_logger['save_folder_str'] = save_folder_str
    exp_logger['save_file_name'] = save_file_name
    exp_logger['save_str'] = save_str
    
    with open(save_str, 'wb') as f:
        pickle.dump(exp_logger, f) 