import numpy as np
from scipy.stats import bernoulli

from differential_privacy.privacy_mechanism import build_privacy_mechanism

import sys
sys.path.append('../')

from markov_decision_process.policies import JointPolicy, LocalPolicies

import time

def run_trajectory(env, policy : np.ndarray, max_steps : int = 50):
    """
    Run a trajectory from the joint initial state implementing the
    specified policy with full communication.

    Parameters
    ----------
    env :
        The environment on which the trajectory is to be run.
    policy : 
        Representation of the team policy. 
        Either a JointPolicy or a LocalPolicy object.

    Returns
    -------
    traj : list
        List of indexes of states. 
    """
    traj = []
    traj.append(env.initial_index)
    s = env.initial_index

    while ((s not in env.target_indexes) and (s not in env.dead_indexes)
                and len(traj) <= max_steps):
        a = policy.sample_joint_action(s)
        s = np.random.choice(np.arange(env.Ns_joint), p=env.T[s,a,:])
        traj.append(s)

    return traj

def run_trajectory_private(
        env, 
        policy, 
        mu : np.ndarray, 
        max_steps : int = 50,
        policy_type : str = 'joint',
    ):
    """
    Run a trajectory from the joint initial state under private 
    play implementing the specified joint policy.

    Parameters
    ----------
    env :
        The environment in which to generate the trajectory.
    policy : 
        Representation of the team policy. 
        Either a JointPolicy or a LocalPolicy object.
    mu : 
        The privacy policy.
    max_steps : 
        Maximum number of steps to take in the trajectory.
    epsilon :
        The privacy parameter.
    k :
        The adjacency parameter.
    policy_type :
        Whether or not to run the privatized policy execution with a 
        joint, local, or acyclic local policies.

    Returns
    -------
    traj : list
        List of indexes of states. 
    """   
    
    if policy_type == 'local':
        assert (isinstance(policy, LocalPolicies) and len(policy.policies) == env.N_agents)
    elif policy_type == 'joint':
        assert (isinstance(policy, JointPolicy) and policy.policy.shape == (env.Ns_joint, env.Na_joint))
    
    traj = []
    agent_s_tuples = {} # This is where we will store the private copy of everyones state
    agent_s_inds = {} # private indexes
    agent_a_inds = {} #
    
    actions = np.arange(env.Na_joint)
    actions_local = np.arange(env.Na_local)
    states = np.arange(env.Ns_joint)

    s_tuple = env.pos_from_index[env.initial_index]
    
    for agent_id in range(env.N_agents):
        agent_s_tuples[agent_id] = s_tuple
        agent_s_inds[agent_id] = \
            env.index_from_pos[agent_s_tuples[agent_id]]
    
    s_tuple = ()
    for agent_id in range(env.N_agents):
        if type(agent_s_tuples[agent_id][env.agent_tuple_slice(agent_id)]) is tuple:
            s_tuple = s_tuple \
                + agent_s_tuples[agent_id][env.agent_tuple_slice(agent_id)]
        else:
            s_tuple = s_tuple \
                + tuple([agent_s_tuples[agent_id][env.agent_tuple_slice(agent_id)]])
    s = env.index_from_pos[s_tuple]
    traj.append(s)
    
    #assume initial condition is known
    last_private_state = {}
    for agent_id in range(env.N_agents):
        last_private_state[agent_id] = env.local_index_from_pos[s_tuple[env.agent_tuple_slice(agent_id)]]

    while ((s not in env.target_indexes) 
            and (s not in env.dead_indexes)
            and len(traj) <= max_steps):
        np.random.seed()
        # Before sharing anything, have each agent privatize its state
        private_states = {}
        for agent_id in range(env.N_agents):
            # Convert this agents state to its local index, i.e., 1...Nr*Nc
            true_state = env.local_index_from_pos[s_tuple[env.agent_tuple_slice(agent_id)]]
            
            # Generate private state
            private_states[agent_id] = np.random.choice(np.arange(env.Ns_local), p=mu[true_state, last_private_state[agent_id], :])
        
        # Now each agent:
        # - collects the private info from the other agents to construct s hat
        agents_actions = {}
        for agent_id in range(env.N_agents):
            agent_i_s_hat = ()
            for i in range(env.N_agents):
                if i==agent_id:
                    if type(s_tuple[env.agent_tuple_slice(agent_id)]) is tuple:
                        agent_i_s_hat=(agent_i_s_hat + s_tuple[env.agent_tuple_slice(agent_id)])
                    else:
                        agent_i_s_hat=(agent_i_s_hat + tuple([s_tuple[env.agent_tuple_slice(agent_id)]]))
                    # agent_i_s_hat=(agent_i_s_hat + s_tuple[env.agent_tuple_slice(agent_id)]) # Use your own real state
                else:
                    if type(env.local_pos_from_index[private_states[i]]) is tuple:
                        agent_i_s_hat=(agent_i_s_hat + env.local_pos_from_index[private_states[i]])
                    else:
                        agent_i_s_hat=(agent_i_s_hat + tuple([env.local_pos_from_index[private_states[i]]])) # use everyone elses private data
                    # agent_i_s_hat=(agent_i_s_hat + env.local_pos_from_index[private_states[i]]) 
            
            # Get action disctribution based on the private information
            agent_i_s_hat_idx = env.index_from_pos[agent_i_s_hat]

            if policy_type == 'joint':
                # Get the team's action, as imagined by the agents private information
                act = policy.sample_joint_action(agent_i_s_hat_idx)
                
                # Extract my action from the joint action
                my_action = env.action_tuple_from_index[act][agent_id]
                agents_actions[agent_id] = my_action
            elif policy_type == 'local':
                # local_policy = policy[agent_id]
                # my_action = np.random.choice(actions_local, p=local_policy[agent_i_s_hat_idx,:])
                my_action = policy.sample_local_action(agent_id, agent_i_s_hat_idx)
                agents_actions[agent_id] = my_action
                
        # Now that we have each agents action, make it a tuple joint action
        a_tuple = tuple(agents_actions[agent_id] for agent_id in range(env.N_agents))
        a_joint_id = env.action_index_from_tuple[a_tuple]
        
        s = np.random.choice(np.arange(env.Ns_joint), p=env.T[s,a_joint_id,:])
        s_tuple = env.pos_from_index[s]
        last_private_state=private_states
        traj.append(s)
    return traj

def empirical_success_rate(env,
                            policy,
                            num_trajectories : int = 1000,
                            max_steps_per_trajectory : int = 50):
    """
    Run a trajectory from the joint initial state implementing the
    specified policy with full communication.

    Parameters
    ----------
    policy : 
        Representation of the team policy. 
        Either a JointPolicy or a LocalPolicy object.
    num_trajectories :
        The number of trajectories to include in the gif.
    max_steps_per_trajectory :
        The maximum number of steps to include in each trajectory
        of the gif.

    Returns
    -------
    success_rate : float
        A numerical value between 0 and 1 indicating the frequency
        at which the policy was observed to reach the target set.
    """
    success_count = 0
    for t_ind in range(num_trajectories):
        temp_traj = run_trajectory(env, policy, 
                            max_steps=max_steps_per_trajectory)
        if (temp_traj[-1] in env.target_indexes):
                success_count = success_count + 1

    return success_count / num_trajectories

def empirical_success_rate_private(env,
                                policy,
                                num_trajectories : int = 1000,
                                max_steps_per_trajectory : int = 50,
                                epsilon=1,
                                k=3,
                                policy_type : str = 'joint',
                            ):
    """
    Run a trajectory from the joint initial state implementing the
    specified policy with full communication.

    Parameters
    ----------
    policy : 
        Representation of the team policy. 
        Either a JointPolicy or a LocalPolicy object.
    use_imaginary_play :
        A boolean flag indicating whether or not to use imaginary 
        play when generating the gifs.
    num_trajectories :
        The number of trajectories to include in the gif.
    max_steps_per_trajectory :
        The maximum number of steps to include in each trajectory
        of the gif.
    epsilon :
        The privacy parameter.
    k :
        The adjacency parameter.
    policy_type :
        Whether or not to run the privatized policy execution with a 
        joint, local, or acyclic local policies.

    Returns
    -------
    success_rate : float
        A numerical value between 0 and 1 indicating the frequency
        at which the policy was observed to reach the target set.
    """
    
    #Construct a privacy mechanism for each agent, 
    # everyone has the same epsilon for now.
    # Assuming the local transition matrices are the same for all agents,
    # grab the transition matrix for the first agent.
    T_local = env.local_transition_matrices[0]
    
    # Build the privacy mechanism (currently implemented as being identical for all agents)
    mu = build_privacy_mechanism(T_local, epsilon=epsilon, k=k)  

    success_count = 0
    for t_ind in range(num_trajectories):
        np.random.seed()
        temp_traj = run_trajectory_private(
            env, 
            policy, 
            mu,
            max_steps=max_steps_per_trajectory,
            policy_type=policy_type,
        )
        if (temp_traj[-1] in env.target_indexes):
                success_count = success_count + 1

    return success_count / num_trajectories