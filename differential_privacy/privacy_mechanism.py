import sys, os
sys.path.append('..')
from environments.ma_gridworld import MAGridworld

import numpy as np

def build_privacy_mechanism(env, epsilon, k):
    """
    Construct the differential privacy mechanism.
    
    Parameters
    ----------
    env :
        The environment for which the privacy mechanism is to be constructed.
    epsilon : float
        The privacy parameter.
    k : int
        The adjacency parameter.
    """
    # make another enviornement with one agent, we need the transition matrix for one agent and this is just easy for now
    one_agent_gridworld = MAGridworld(N_agents= 1,
                                        Nr =env.Nr,
                                        Nc =env.Nc,
                                        slip_p =env.slip_p,
                                        initial_state =(0,0),
                                        target_states =[(2,3)],
                                        dead_states =[],
                                        lava =env.lava,
                                        walls= env.walls,
                                        load_file_str = '',
                                        seed = env.seed
                                        )
    # empty policy
    mu = np.zeros([env.Ns_local, env.Ns_local, env.Ns_local])
    
    for s in range(env.Ns_local):
        for s_o_t_1 in range(env.Ns_local):
            # Calculate the number of feasible states we can transition to from s_o_t_1
            # Need to iterate over each action in the MDP, redundant but more general
            
            possible_transitions = set() # all the states we can transition to
            for row in one_agent_gridworld.T[s_o_t_1,:,:]:
                # Add the index of all states we can transition to
                [possible_transitions.add(s_ ) for s_ in np.where(row!=0)[0]] 
            N_s_o_t_1 = len(possible_transitions)
            tau = 1/((N_s_o_t_1-1)*np.exp(-epsilon/k)+1)
            
            for s_o_t in range(env.Ns_local):
                if s == s_o_t and s in possible_transitions:
                    mu[s,s_o_t_1,s_o_t] = tau
                elif s != s_o_t and s_o_t in possible_transitions:
                    mu[s,s_o_t_1,s_o_t] = (1-tau*int(s in possible_transitions))/(N_s_o_t_1-int(s in possible_transitions))
    return mu