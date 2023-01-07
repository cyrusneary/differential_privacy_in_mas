import numpy as np
from scipy.stats import bernoulli

from differential_privacy.privacy_mechanism import build_privacy_mechanism

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
        Matrix representing the policy. policy[s_ind, a_ind] is the 
        probability of taking the action indexed by a_ind from the 
        joint state indexed by s_ind.

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
        a = np.random.choice(np.arange(env.Na_joint), p=policy[s,:])
        s = np.random.choice(np.arange(env.Ns_joint), p=env.T[s,a,:])
        traj.append(s)

    return traj

def run_trajectory_imaginary(env, 
                                policy : np.ndarray, 
                                max_steps : int = 50):
    """
    Run a trajectory from the joint initial state under imaginary 
    play implementing the specified joint policy.

    Parameters
    ----------
    policy : 
        Matrix representing the policy. policy[s_ind, a_ind] is the 
        probability of taking the action indexed by a_ind from the 
        joint state indexed by s_ind.

    Returns
    -------
    traj : list
        List of indexes of states. 
    """
    traj = []
    agent_s_tuples = {}
    agent_s_inds = {}
    agent_a_inds = {}

    actions = np.arange(env.Na_joint)
    states = np.arange(env.Ns_joint)

    s_tuple = env.pos_from_index[env.initial_index]

    for agent_id in range(env.N_agents):
        agent_s_tuples[agent_id] = s_tuple
        agent_s_inds[agent_id] = \
            env.index_from_pos[agent_s_tuples[agent_id]]

    s_tuple = ()
    for agent_id in range(env.N_agents):
        s_tuple = (s_tuple
                    + agent_s_tuples[agent_id][2*agent_id:(2*agent_id+2)])
    s = env.index_from_pos[s_tuple]
    traj.append(s)

    while ((s not in env.target_indexes) 
                and (s not in env.dead_indexes)
                and len(traj) <= max_steps):
        for agent_id in range(env.N_agents):

            s_imag_ind = agent_s_inds[agent_id]
            # Get the agent's action distribution from the policy.
            act_dist = policy[s_imag_ind, :]

            # Get the team's action, as imagined by the agent.
            act = np.random.choice(actions, p=act_dist)


            ##### This is where privacy goes
            # Instead of imagining where the team goes next
            
            # Get the team's next state, as imagined by the agent.
            s_next_ind = np.random.choice(states, p=env.T[s_imag_ind, act, :])
            s_next_tuple = env.pos_from_index[s_next_ind]

            agent_a_inds[agent_id] = act
            agent_s_inds[agent_id] = s_next_ind
            agent_s_tuples[agent_id] = s_next_tuple

        # Construct the true team next state
        s_tuple = ()
        for agent_id in range(env.N_agents):
            s_tuple = (s_tuple
                    + agent_s_tuples[agent_id][2*agent_id:(2*agent_id+2)])
        s = env.index_from_pos[s_tuple]
        traj.append(s)
        
        # Privatize

    return traj

def run_trajectory_intermittent(env, 
                                policy : np.ndarray, 
                                q : float,
                                max_steps : int = 50):
    """
    Run a trajectory from the joint initial state under imaginary 
    play implementing the specified joint policy.

    Parameters
    ----------
    policy : 
        Matrix representing the policy. policy[s_ind, a_ind] is the 
        probability of taking the action indexed by a_ind from the 
        joint state indexed by s_ind.
    q :
        Value in [0,1] representing the parameter of the bernoulli
        distribution modeling the probability of loosing 
        communication at each step.

    Returns
    -------
    traj : list
        List of indexes of states. 
    """
    traj = []
    agent_s_tuples = {}
    agent_s_inds = {}
    agent_a_inds = {}

    actions = np.arange(env.Na_joint)
    states = np.arange(env.Ns_joint)

    s_tuple = env.pos_from_index[env.initial_index]

    for agent_id in range(env.N_agents):
        agent_s_tuples[agent_id] = s_tuple
        agent_s_inds[agent_id] = \
            env.index_from_pos[agent_s_tuples[agent_id]]
    
    s_tuple = ()
    for agent_id in range(env.N_agents):
        s_tuple = (s_tuple
                    + agent_s_tuples[agent_id][2*agent_id:(2*agent_id+2)])
    s = env.index_from_pos[s_tuple]
    traj.append(s)

    timestep = 0
    while ((s not in env.target_indexes) 
                and (s not in env.dead_indexes)
                and len(traj) <= max_steps):
        
        # flag should be true if communication is available
        comm_flag = 1 - bernoulli.rvs(q)

        if comm_flag:
            for agent_id in range(env.N_agents):
                agent_s_inds[agent_id] = s
            a = np.random.choice(np.arange(env.Na_joint), p=policy[s,:])

            for agent_id in range(env.N_agents):
                # Get the team's next state, as imagined by the agent.
                s_next_ind = np.random.choice(states, p=env.T[s, a, :])
                s_next_tuple = env.pos_from_index[s_next_ind]

                agent_a_inds[agent_id] = a
                agent_s_inds[agent_id] = s_next_ind
                agent_s_tuples[agent_id] = s_next_tuple

            # Construct the true team next state
            s_tuple = ()
            for agent_id in range(env.N_agents):
                s_tuple = (s_tuple
                    + agent_s_tuples[agent_id][2*agent_id:(2*agent_id+2)])
            s = env.index_from_pos[s_tuple]
            
        else:
            for agent_id in range(env.N_agents):

                s_imag_ind = agent_s_inds[agent_id]

                # Get the agent's action distribution from the policy.
                act_dist = policy[s_imag_ind, :]

                # Get the team's action, as imagined by the agent.
                act = np.random.choice(actions, p=act_dist)

                # Get the team's next state, as imagined by the agent.
                s_next_ind = np.random.choice(states, p=env.T[s_imag_ind, act, :])
                s_next_tuple = env.pos_from_index[s_next_ind]

                agent_a_inds[agent_id] = act
                agent_s_inds[agent_id] = s_next_ind
                agent_s_tuples[agent_id] = s_next_tuple

            # Construct the true team next state
            s_tuple = ()
            for agent_id in range(env.N_agents):
                s_tuple = (s_tuple
                    + agent_s_tuples[agent_id][2*agent_id:(2*agent_id+2)])
            s = env.index_from_pos[s_tuple]

        traj.append(s)

        timestep = timestep + 1

    return traj

def run_trajectory_private(
        env, 
        policy, 
        mu : np.ndarray, 
        max_steps : int = 50,
        use_marginalized_policies : bool = False,
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
        If use_marginalized_policies is False, policy is an ndarray and
        policy[s_ind, a_ind] is the probability of taking the joint action 
        indexed by a_ind from the joint state indexed by s_ind.
        If use_marginalized_policies is True, policy is a list of ndarrays and
        policy[agent_id][s_ind, a_ind] is the probability of the agent specified
        by agent_id taking the local action a_ind, from the joint state s_ind.
    mu : 
        The privacy policy.
    max_steps : 
        Maximum number of steps to take in the trajectory.
    epsilon :
        The privacy parameter.
    k :
        The adjacency parameter.
    use_marginalized_policies :
        A boolean flag indicating whether or not to use marginalized policies
        when simulating the team's behavior.

    Returns
    -------
    traj : list
        List of indexes of states. 
    """   
    
    traj = []
    agent_s_tuples = {} # This is where we will store the private copy of everyones state
    agent_s_inds = {} #private indexes
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
        s_tuple = (s_tuple
                    + agent_s_tuples[agent_id][2*agent_id:(2*agent_id+2)])
    s = env.index_from_pos[s_tuple]
    traj.append(s)
    
    #assume initial condition is known
    last_private_state = {}
    for agent_id in range(env.N_agents):
        last_private_state[agent_id] = env.local_index_from_pos[s_tuple[2*agent_id:(2*agent_id+2)]]

    while ((s not in env.target_indexes) 
            and (s not in env.dead_indexes)
            and len(traj) <= max_steps):
        np.random.seed()
        # Before sharing anything, have each agent privatize its state
        private_states = {}
        for agent_id in range(env.N_agents):
            # Convert this agents state to its local index, i.e., 1...Nr*Nc
            true_state = env.local_index_from_pos[s_tuple[2*agent_id:(2*agent_id+2)]]
            
            # Generate private state
            private_states[agent_id] = np.random.choice(np.arange(env.Ns_local), p=mu[true_state, last_private_state[agent_id], :])
        
        # Now each agent:
        # - collects the private info from the other agents to construct s hat
        agents_actions = {}
        for agent_id in range(env.N_agents):
            agent_i_s_hat = ()
            for i in range(env.N_agents):
                if i==agent_id:
                    agent_i_s_hat=(agent_i_s_hat + s_tuple[2*agent_id:(2*agent_id+2)]) # Use your own real state
                else:
                    agent_i_s_hat=(agent_i_s_hat + env.local_pos_from_index[private_states[i]]) # use everyone elses private data
            
            # Get action disctribution based on the private information
            agent_i_s_hat_idx = env.index_from_pos[agent_i_s_hat]

            if not use_marginalized_policies:
                act_dist = policy[agent_i_s_hat_idx,:]
                
                # Get the team's action, as imagined by the agents private information
                act = np.random.choice(actions, p=act_dist)
                
                # Extract my action from the joint action
                my_action = env.action_tuple_from_index[act][agent_id]
                agents_actions[agent_id] = my_action
            else:
                local_policy = policy[agent_id]
                my_action = np.random.choice(actions_local, p=local_policy[agent_i_s_hat_idx,:])
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
                                policy : np.ndarray,
                                use_imaginary_play : bool = False,
                                num_trajectories : int = 1000,
                                max_steps_per_trajectory : int = 50):
    """
    Run a trajectory from the joint initial state implementing the
    specified policy with full communication.

    Parameters
    ----------
    policy : 
        A (Ns, Na) Matrix representing the policy. 
        policy[s_ind, a_ind] is the probability of taking the action
        indexed by a_ind from the joint state indexed by s_ind.
    use_imaginary_play :
        A boolean flag indicating whether or not to use imaginary 
        play when generating the gifs.
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
        if use_imaginary_play:
            temp_traj = run_trajectory_imaginary(env, policy, 
                                max_steps=max_steps_per_trajectory)
        else:
            temp_traj = run_trajectory(env, policy, 
                                max_steps=max_steps_per_trajectory)
        if (temp_traj[-1] in env.target_indexes):
                success_count = success_count + 1

    return success_count / num_trajectories

def empirical_intermittent_success_rate(env,
                            policy : np.ndarray,
                            q: float,
                            num_trajectories : int = 1000,
                            max_steps_per_trajectory : int = 50):
    """
    Run a trajectory from the joint initial state implementing the
    specified policy with full communication.

    Parameters
    ----------
    policy : 
        A (Ns, Na) Matrix representing the policy. 
        policy[s_ind, a_ind] is the probability of taking the action
        indexed by a_ind from the joint state indexed by s_ind.
    q :
        Value in [0,1] representing the parameter of the bernoulli
        distribution modeling the probability of loosing 
        communication at each step.            
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
        temp_traj = run_trajectory_intermittent(env, policy, q,
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
                                use_marginalized_policies : bool = False,
                            ):
    """
    Run a trajectory from the joint initial state implementing the
    specified policy with full communication.

    Parameters
    ----------
    policy : 
        Representation of the team policy. 
        If use_marginalized_policies is False, policy is an ndarray and
        policy[s_ind, a_ind] is the probability of taking the joint action 
        indexed by a_ind from the joint state indexed by s_ind.
        If use_marginalized_policies is True, policy is a list of ndarrays and
        policy[agent_id][s_ind, a_ind] is the probability of the agent specified
        by agent_id taking the local action a_ind, from the joint state s_ind.
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
    use_marginalized_policies :
        A boolean flag indicating whether or not to use marginalized policies
        when simulating the team's performance.

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
            use_marginalized_policies=use_marginalized_policies,
        )
        if (temp_traj[-1] in env.target_indexes):
                success_count = success_count + 1

    return success_count / num_trajectories