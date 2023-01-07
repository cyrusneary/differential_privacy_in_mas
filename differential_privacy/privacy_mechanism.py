import sys, os
sys.path.append('..')
from environments.ma_gridworld import MAGridworld

import numpy as np

def build_privacy_mechanism(T_local : np.ndarray, epsilon : float, k : int):
    """
    Construct the differential privacy mechanism.
    
    Parameters
    ----------
    T_local : np.ndarray
        The local transition matrix of the agent we are constructing the privacy mechanism for.
    epsilon : float
        The privacy parameter.
    k : int
        The adjacency parameter.
    """
    Ns_local = T_local.shape[0]

    # empty policy
    mu = np.zeros([Ns_local, Ns_local, Ns_local])
    
    for s in range(Ns_local):
        for s_o_t_1 in range(Ns_local):
            # Calculate the number of feasible states we can transition to from s_o_t_1
            # Need to iterate over each action in the MDP, redundant but more general
            
            possible_transitions = set() # all the states we can transition to
            for row in T_local[s_o_t_1,:,:]:
                # Add the index of all states we can transition to
                [possible_transitions.add(s_ ) for s_ in np.where(row!=0)[0]] 
            N_s_o_t_1 = len(possible_transitions)
            tau = 1/((N_s_o_t_1-1)*np.exp(-epsilon/k)+1)
            
            for s_o_t in range(Ns_local):
                if s == s_o_t and s in possible_transitions:
                    mu[s,s_o_t_1,s_o_t] = tau
                elif s != s_o_t and s_o_t in possible_transitions:
                    mu[s,s_o_t_1,s_o_t] = (1-tau*int(s in possible_transitions))/(N_s_o_t_1-int(s in possible_transitions))
    return mu