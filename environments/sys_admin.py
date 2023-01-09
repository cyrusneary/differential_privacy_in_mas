# from cvxpy.expressions.cvxtypes import index
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio
import numpy as np
import pickle
import sys, os, time

sys.path.append('../')
from markov_decision_process.mdp import MDP

from scipy.stats import bernoulli

class SysAdmin(object):

    def __init__(self, 
                N_agents : int = 4,
                p_repair : float = 0.1,
                p_unhealthy : float = 0.1,
                p_down : float = 0.1,
                allowable_simultaneous_repair : int = 2,
                allowable_simultaneous_down : int = 2,
                initial_state : tuple = (0, 0, 3, 3),
                load_file_str : str = '',
                seed : int = 0
                ):
        """
        Initializer for the SysAdmin environment object.

        The team's states and actions are represented as:
        (state1, state2, ..., stateN)
        (action1, action2, ..., actionN)

        Parameters
        ----------
        N_agents :
            Number of agents in the gridworld.
        p_repair :
            The probability of returning to a healthy state when waiting in the repair state.
        p_unhealthy :
            The probability of transitioning from a healthy state to an unhealthy state.
        p_down :
            The probability of transitioning from an unhealthy state to a down state.
        initial_state :
            The initial joint position of the agents in the environment.
        load_file_str :
            String representing the path of a data file to use to load a
            pre-build multiagent gridworld object. If load_file_str = '', 
            then the gridworld is instead constructed from scratch.
        seed :
            The random seed for the environment.
        """

        if load_file_str == '':
            np.random.seed(seed)

            # Local states:
            # 0 : healthy
            # 1 : unhealthy
            # 2 : repairing
            # 3 : down
            self.Ns_local = 4
            self.N_agents = N_agents
            self.Ns_joint = (self.Ns_local)**self.N_agents

            # The team is successful when all agents are working (in either a healthy or unhealthy state).
            self.local_target_states = [0, 1] 

            self.allowable_simultaneous_repair = allowable_simultaneous_repair
            self.allowable_simultaneous_down = allowable_simultaneous_down

            # Actions:
            # 0 : wait
            # 1 : repair
            self.Na_local = 2
            self.Na_joint = self.Na_local**self.N_agents

            self.team_state_tuple_shape = tuple(self.Ns_local for i in range(self.N_agents))
            self.team_action_tuple_shape = tuple(self.Na_local for i in range(self.N_agents))
            self.team_action_tuple_shape_with_aux = tuple(self.Na_local + 1 for i in range(self.N_agents))

            self._construct_state_space()
            self._construct_action_space()

            # Check that the initial state is valid.
            assert len(initial_state) == self.N_agents
            self.initial_state = initial_state
            self.initial_index = self.index_from_pos[self.initial_state]
            assert self.initial_index <= self.Ns_joint - 1

            self._construct_target_states()
            self._construct_dead_states()

            assert p_repair <= 1.0 and p_repair >= 0.0
            assert p_unhealthy <= 1.0 and p_unhealthy >= 0.0
            assert p_down <= 1.0 and p_down >= 0.0

            self.p_repair = p_repair
            self.p_unhealthy = p_unhealthy
            self.p_down = p_down

            self._build_transition_matrix()

            self.seed = seed

        else:
            self.load(load_file_str)

    def save(self, save_file_str : str):
        """
        Save the multiagent gridworld object.
        """
        save_dict = {}
        save_dict['p_repair'] = self.p_repair
        save_dict['p_unhealthy'] = self.p_unhealthy
        save_dict['p_down'] = self.p_down
        save_dict['allowable_simultaneous_repair'] = self.allowable_simultaneous_repair
        save_dict['allowable_simultaneous_down'] = self.allowable_simultaneous_down
        save_dict['N_agents'] = self.N_agents
        save_dict['Ns_local'] = self.Ns_local
        save_dict['Ns_joint'] = self.Ns_joint
        save_dict['Na_local'] = self.Na_local
        save_dict['Na_joint'] = self.Na_joint
        save_dict['initial_state'] = self.initial_state
        save_dict['initial_index'] = self.initial_index
        save_dict['local_target_states'] = self.local_target_states
        save_dict['target_states'] = self.target_states
        save_dict['target_indexes'] = self.target_indexes
        save_dict['team_state_tuple_shape'] = self.team_state_tuple_shape
        save_dict['team_action_tuple_shape'] = self.team_action_tuple_shape
        save_dict['team_action_tuple_shape_with_aux'] = self.team_action_tuple_shape_with_aux
        save_dict['dead_states'] = self.dead_states
        save_dict['dead_indexes'] = self.dead_indexes
        save_dict['T'] = self.T
        save_dict['seed'] = self.seed

        with open(save_file_str, 'wb') as f:
            pickle.dump(save_dict, f)

    def load(self, load_file_str : str):
        """
        Load the multiagent gridworld data from a file.
        """
        with open(load_file_str, 'rb') as f:
            save_dict = pickle.load(f)

        self.p_repair = save_dict['p_repair']
        self.p_unhealthy = save_dict['p_unhealthy']
        self.p_down = save_dict['p_down']
        self.allowable_simultaneous_repair = save_dict['allowable_simultaneous_repair']
        self.allowable_simultaneous_down = save_dict['allowable_simultaneous_down']
        self.N_agents = save_dict['N_agents']
        self.Ns_local = save_dict['Ns_local']
        self.Ns_joint = save_dict['Ns_joint']
        self.Na_local = save_dict['Na_local']
        self.Na_joint = save_dict['Na_joint']
        self.initial_state = save_dict['initial_state']
        self.initial_index = save_dict['initial_index']
        self.local_target_states = save_dict['local_target_states']
        self.target_states = save_dict['target_states']
        self.target_indexes = save_dict['target_indexes']
        self.team_state_tuple_shape = save_dict['team_state_tuple_shape']
        self.team_action_tuple_shape = save_dict['team_action_tuple_shape']
        self.team_action_tuple_shape_with_aux = save_dict['team_action_tuple_shape_with_aux']
        self.dead_states = save_dict['dead_states']
        self.dead_indexes = save_dict['dead_indexes']
        self.T = save_dict['T']
        self.seed = save_dict['seed']

        np.random.seed(self.seed)

        self._construct_state_space()
        self._construct_action_space()

    def agent_tuple_slice(self, agent_id):
        """
        Return an array slice representing the agent's indexing within the 
        tuple representation of the team's state.
        """
        return agent_id

    # def index_from_state_tuple(self, state_tuple):
    #     """
    #     Return the joint state index from a tuple of local state indexes.
    #     """
    #     return np.ravel_multi_index(state_tuple, self.team_state_tuple_shape)

    # def state_tuple_from_index(self, index):
    #     """
    #     Return the tuple of local state indexes from a joint state index.
    #     """
    #     return np.unravel_index(index, self.team_state_tuple_shape)

    # def action_index_from_tuple(self, action_tuple):
    #     """
    #     Return the joint action index from a tuple of local action indexes.
    #     """
    #     return np.ravel_multi_index(action_tuple, self.team_action_tuple_shape)

    # def action_tuple_from_index(self, index):
    #     """
    #     Return the tuple of local action indexes from a joint action index.
    #     """
    #     return np.unravel_index(index, self.team_action_tuple_shape)

    # def action_index_from_tuple_with_aux(self, action_tuple):
    #     """
    #     Return the joint action index from a tuple of local action indexes.
    #     """
    #     return np.ravel_multi_index(action_tuple, self.team_action_tuple_shape_with_aux)

    # def action_tuple_from_index_with_aux(self, index):
    #     """
    #     Return the tuple of local action indexes from a joint action index.
    #     """
    #     return np.unravel_index(index, self.team_action_tuple_shape_with_aux)

    def _construct_target_states(self):
        """
        Construct the target states.
        """
        self.target_states = []
        self.target_indexes = []
        for s_ind in range(self.Ns_joint):
            state_tuple = self.pos_from_index[s_ind]
            if all([state_tuple[agent_ind] in self.local_target_states for agent_ind in range(self.N_agents)]):
                self.target_states.append(state_tuple)
                self.target_indexes.append(s_ind)

    def _construct_dead_states(self):
        """
        Construct the dead states.
        """
        self.dead_states = []
        self.dead_indexes = []
        for s_ind in range(self.Ns_joint):
            state_tuple = self.pos_from_index[s_ind]

            num_rebooting = np.sum(np.array(state_tuple) == 2)
            num_down = np.sum(np.array(state_tuple) == 3)

            if num_rebooting > self.allowable_simultaneous_repair or num_down > self.allowable_simultaneous_down:
                self.dead_states.append(state_tuple)
                self.dead_indexes.append(s_ind)

    def _construct_state_space(self):
        """
        Construct the state space.
        """
        self.pos_from_index = {}
        self.index_from_pos = {}

        for i in range(self.Ns_joint):
            self.pos_from_index[i] = np.unravel_index(i, self.team_state_tuple_shape)
            self.index_from_pos[self.pos_from_index[i]] = i
            
        # Local state index from alternate state representation.
        # In this case, the local state index is the same as the joint state index.
        # Keep this part of the implementation for external functions which will 
        # try and access this field.
        self.local_index_from_pos = {}
        self.local_pos_from_index = {}

        for i in range(self.Ns_local):
            self.local_pos_from_index[i] = i
            self.local_index_from_pos[i] = i

    def _construct_action_space(self):
        """
        Construct the action space.
        """
        self.action_index_from_tuple = {}
        self.action_tuple_from_index = {}

        action_shape = tuple(self.Na_local for i in range(self.N_agents))

        for i in range(self.Na_joint):
            self.action_tuple_from_index[i] = np.unravel_index(i, action_shape)
            self.action_index_from_tuple[self.action_tuple_from_index[i]] = i

        self.action_index_from_tuple_with_aux = {}
        self.action_tuple_from_index_with_aux = {}

        action_shape = tuple(self.Na_local + 1 for i in range(self.N_agents))

        Na_aux = (self.Na_local + 1) ** self.N_agents
        for i in range(Na_aux):
            self.action_tuple_from_index_with_aux[i] = np.unravel_index(i, action_shape)
            self.action_index_from_tuple_with_aux[self.action_tuple_from_index_with_aux[i]] = i

    def _build_transition_matrix(self):
        """
        Build the transition matrix storing the dynamics of the 
        gridworld environment. 
        
        self.T[s,a,s'] = probability of reaching joint state s' from 
                            joint state s under joint action a.

        The Local transitions of each agent are assumed to be independent.
        
        Given an action, each agent proceeds to the desired next 
        state with prob 1.0 - slip_p. The remaining slip_p is 
        distributed among the remaining adjacent states. If the selected
        action moves into a wall, then all of the probability is 
        assigned to the available adjacent states.
        """
        # Initialize the joint transition matrix
        self.T = np.zeros((self.Ns_joint, self.Na_joint, self.Ns_joint))

        # Initialize the local transition matrices for each agent.
        self.local_transition_matrices = {}
        for agent_id in range(self.N_agents):
            self.local_transition_matrices[agent_id] = \
                np.zeros((self.Ns_local, self.Na_local, self.Ns_local))

        for s in range(self.Ns_joint):

            # Get the tuple of row and column positions of all agents
            pos = self.pos_from_index[s] 

            # For each agent, find the next available local positions
            # and the probabilities of the individual agents 
            # transitioning to those states.
            local_trans_funcs = {}
            for agent_id in range(self.N_agents):

                local_state = pos[agent_id]

                # dictionary containing local transition functions mapping
                # local_trans_funcs[agent_id][local_action][local_next_state] 
                #           = prob value
                local_trans_funcs[agent_id] = {}

                # Local states:
                # 0 : healthy
                # 1 : unhealthy
                # 2 : repairing
                # 3 : down

                # action 0 : wait
                # action 1 : repair

                if local_state == 0:
                    self.local_transition_matrices[agent_id][local_state, 0, :] = \
                        [1 - self.p_unhealthy, self.p_unhealthy, 0.0, 0.0]
                    self.local_transition_matrices[agent_id][local_state, 1, :] = \
                        [0.0, 0.0, 1.0, 0.0]
                if local_state == 1:
                    self.local_transition_matrices[agent_id][local_state, 0, :] = \
                        [0.0, 1 - self.p_down, 0.0, self.p_down]
                    self.local_transition_matrices[agent_id][local_state, 1, :] = \
                        [0.0, 0.0, 1.0, 0.0]
                if local_state == 2:
                    self.local_transition_matrices[agent_id][local_state, 0, :] = \
                        [self.p_repair, 0.0, 1 - self.p_repair, 0.0]
                    self.local_transition_matrices[agent_id][local_state, 1, :] = \
                        [0.0, 0.0, 1.0, 0.0]
                if local_state == 3:
                    self.local_transition_matrices[agent_id][local_state, 0, :] = \
                        [0.0, 0.0, 0.0, 1.0]
                    self.local_transition_matrices[agent_id][local_state, 1, :] = \
                        [0.0, 0.0, 1.0, 0.0]

            # Now that we have the local transition functions of all 
            # the agents, construct the joint transition function using
            # the assumption that all local transition probabilities
            # are independent.
            for a_ind in range(self.Na_joint):
                action_tuple = self.action_tuple_from_index[a_ind]

                for next_s_ind in range(self.Ns_joint):
                    next_s_tuple = self.pos_from_index[next_s_ind]

                    prob_transition = 1.0

                    for agent_id in range(self.N_agents): 
                        local_state = pos[agent_id]
                        local_action = action_tuple[agent_id]
                        local_next_state = next_s_tuple[agent_id]

                        prob_local_trans = \
                            self.local_transition_matrices[agent_id][local_state, local_action, local_next_state]

                        prob_transition = prob_transition * prob_local_trans

                    self.T[s, a_ind, next_s_ind] = prob_transition
                    
        # # Set all target states to be absorbing
        # for state in self.target_indexes:
        #     for action in range(self.Na_joint):
        #         self.T[state, action, state] = 1.0

        # # Set all dead states to be absorbing
        # for state in self.dead_indexes:
        #     for action in range(self.Na_joint):
        #         self.T[state, action, state] = 1.0

    def check_agent_state_action(self,
                                agent_id : int,
                                local_state : int,
                                team_state_ind : int,
                                local_action : int,
                                team_action_ind : int):
        """
        Function to check whether a particular agent is occupying a 
        particular local state-action pair, when the team is occupying
        a particular joint state-action pair.

        Parameters
        ----------
        agent_id :
            The index of the agent.
        local_state :
            The index of the local state of the agent.
        team_state_ind :
            The index of the joint state of the team.
        local_action :
            The index of the local action of the agent.
        team_action_ind :
            The index of the joint action of the team.

        Returns
        -------
        tf : bool
            Return True if the agent's local state-action pair agrees
            with the joint state-action pair, and false otherwise.
        """
        team_state_tuple = self.pos_from_index[team_state_ind]
        team_action_tuple = self.action_tuple_from_index[team_action_ind]

        if (local_state == team_state_tuple[agent_id]
            and team_action_tuple[agent_id] == local_action):
            return True
        else:
            return False

    def check_agent_state_action_with_aux(self,
                                        agent_id : int,
                                        local_state : int,
                                        team_state_ind : int,
                                        local_action : int,
                                        team_action_ind : int):
        """
        Function to check whether a particular agent is occupying a 
        particular local state-action pair, when the team is occupying
        a particular joint state-action pair.

        Parameters
        ----------
        agent_id :
            The index of the agent.
        local_state_ind :
            The index of the local state of the agent.
        team_state_ind :
            The index of the joint state of the team.
        local_action :
            The index of the local action of the agent.
        team_action_ind :
            The index of the joint action of the team.

        Returns
        -------
        tf : bool
            Return True if the agent's local state-action pair agrees
            with the joint state-action pair, and false otherwise.
        """
        team_state_tuple = self.pos_from_index[team_state_ind]
        team_action_tuple = self.action_tuple_from_index_with_aux[team_action_ind]

        if (local_state == team_state_tuple[agent_id]
            and team_action_tuple[agent_id] == local_action):
            return True
        else:
            return False

    def check_agent_state(self,
                        agent_id : int,
                        local_state : int,
                        team_state_ind : int):
        """
        Function to check whether a particular agent is occupying a 
        particular local state, when the team is occupying a particular 
        joint state.

        Parameters
        ----------
        agent_id :
            The index of the agent.
        local_state :
            The index of the local state of the agent.
        team_state_ind :
            The index of the joint state of the team.

        Returns
        -------
        tf : bool
            Return True if the agent's local state agrees with the joint
            state, and false otherwise.
        """
        team_state_tuple = self.pos_from_index[team_state_ind]

        if local_state == team_state_tuple[agent_id]:
            return True
        else:
            return False

    def build_mdp(self, gamma : float = 1.0):
        """Build an MDP model of the environment."""
        return MDP(self.Ns_joint, 
                    self.Na_joint, 
                    self.T, 
                    self.initial_index,
                    self.target_indexes,
                    self.dead_indexes,
                    gamma=gamma)

    #################### Visualization Methods
        
def main():
    ##### BUILD THE ENVIRONMENT FROM SCRATCH AND SAVE IT

    # Build the gridworld
    print('Building environment')
    t_start = time.time()
    env = SysAdmin(N_agents = 4,
                    p_repair = 0.1,
                    p_unhealthy = 0.1,
                    p_down = 0.1,
                    allowable_simultaneous_repair = 2,
                    allowable_simultaneous_down = 2,
                    initial_state = (0, 0, 3, 3),
                    load_file_str = '',
                    seed = 0
                    )

    print('Constructed environment in {} seconds'.format(time.time() - t_start))

    # Sanity check on the transition matrix
    for s in range(env.Ns_joint):
        for a in range(env.Na_joint):
            assert(np.abs(np.sum(env.T[s, a, :]) - 1.0) <= 1e-12)

    print(env.dead_indexes)
    print(env.target_indexes)

    # Save the constructed gridworld
    save_file_str = os.path.join(os.path.abspath(os.path.curdir), 
                                    'saved_environments', 'sys_admin_env.pkl')
    env.save(save_file_str)

if __name__ == '__main__':
    main()