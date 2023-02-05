exp_logger = {
    'exp_name' : 'sys_admin_minimum_dependency_2233',
    'results' : {},
    'environment_type' : 'sys_admin',
    'environment_settings' : {
        'N_agents' : 4,
        'p_repair' : 0.9, # 0.9
        'p_unhealthy' : 0.1, # 0.1
        'p_down' : 0.1, # 0.1
        'allowable_simultaneous_repair' : 2,
        'allowable_simultaneous_down' : 2,
        'initial_state' : (2, 2, 3, 3), # (0, 0, 3, 3)
        'load_file_str' : '',
        'seed' : 0
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
        'reachability_coef' : 100.0, # 10.0
        'exp_len_coef' : 1.0, # 0.1
        'total_corr_coef' : 1.0 # 4.0
    },
    'empirical_eval_settings' : {
        'num_trajectories' : 1000,
        'max_steps_per_trajectory' : 200,
        'privacy_parameter' : 1.0,
        'adjacency_parameter' : 1,
        'use_marginalized_policies' : True,
    },
}