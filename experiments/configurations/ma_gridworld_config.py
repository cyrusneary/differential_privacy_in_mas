exp_logger = {
    'exp_name' : 'ma_gridworld_minimum_dependency_0p05',
    'results' : {},
    'environment_type' : 'ma_gridworld',
    'environment_settings' : {
        'N_agents' : 2,
        'Nr' : 5,
        'Nc' : 5,
        'slip_p' : 0.05, # 0.05
        'initial_state' : (4,0,4,4),
        'target_states' : [(4, 3, 4, 1)],
        'dead_states' : [],
        'lava' : [(0, 0), (0,3), (0,1)],
        'walls' : [(0,2), (2,2), (4,2)],
        'seed' : 42,
    },
    'initial_soln_guess_setup' : {
        'type' : 'entropy', # reachability, entropy
        'settings' : {
            'exp_len_coef' : 0.1, 
            'entropy_coef' : 0.1,
            'max_length_constr' : 20,
            },
    },
    'optimization_params' : {
        'reachability_coef' : 10.0, # 10.0
        'exp_len_coef' : 0.1, # 0.1
        'total_corr_coef' : 4.0 # 4.0
    },
    'empirical_eval_settings' : {
        'num_trajectories' : 1000,
        'max_steps_per_trajectory' : 200,
        'privacy_parameter' : 1,
        'adjacency_parameter' : 3,
        'policy_type' : 'acyclic', # joint, local, acyclic
        'dependency_structure' : [(1, 0)],
    },
}