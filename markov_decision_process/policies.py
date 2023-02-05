import numpy as np
import networkx as nx
import sys
sys.path.append('..')

from markov_decision_process.mdp import MDP

from scipy.stats import entropy

class JointPolicy:

    def __init__(
        self, 
        mdp : MDP,
        num_agents : int,
        x : np.ndarray = None) -> None:
        """
        Initialize the joint policy.
        
        Parameters
        ----------
        mdp :
            An object representing the multiagent MDP on which the policies are to be defined.
        x : np.ndarray
            An array of occupancy variables.
        num_agents : int
            The number of agents in the system.
        """
        self.mdp = mdp
        self.num_agents = num_agents
        
        self.x = x
        
        if x is not None:
            self.construct_joint_policy_from_occupancy_vars(x)

    def sample_joint_action(self, joint_state : int):
        """
        Return the joint action for a given joint state.
        
        Parameters
        ----------
        joint_state : int
            The joint state for which the joint action is to be sampled.
        """
        return np.random.choice(np.arange(self.mdp.Na), p=self.policy[joint_state, :])

    def construct_joint_policy_from_occupancy_vars(self, x : np.ndarray) -> None:
        """
        Construct a joint policy from a vector of occupancy variables.
        
        Parameters
        ----------
        x : np.ndarray
            An array of occupancy variables.
        """
        self.x = x
        
        Na_local = round(pow(self.mdp.Na, 1/self.num_agents))
        Na = (Na_local + 1) ** self.num_agents

        action_shape = tuple([Na_local + 1 for i in range(self.num_agents)])
        
        local_auxiliary_action_index = Na_local

        true_actions = []
        partially_auxiliary_actions = []
        all_auxiliary_actions = []
        for a_joint in range(Na):
            a_tuple = np.unravel_index(a_joint, action_shape)
            if np.sum(np.array(a_tuple) == local_auxiliary_action_index) == 0:
                true_actions.append(a_joint)
            elif np.sum(np.array(a_tuple) == local_auxiliary_action_index) == self.num_agents:
                all_auxiliary_actions.append(a_joint)
            else:
                partially_auxiliary_actions.append(a_joint)

        x_mod = np.copy(x)[:, true_actions]

        policy = np.zeros((self.mdp.Ns, self.mdp.Na))
        for s in range(self.mdp.Ns):
            for a in range(self.mdp.Na):
                if not (np.sum(x_mod[s,:]) == 0.0):
                    policy[s,a] = x_mod[s, a] / np.sum(x_mod[s, :])
                else:
                    policy[s,a] = 1.0 / len(x_mod[s,:])
                    
        self.policy = policy

def kl_divergence_policies(
        policy_1 : np.ndarray, 
        policy_2 : np.ndarray,
    ) -> float:
        """
        Compute the KL divergence between two policies.
        
        Parameters
        ----------
        policy_1 : ndarray
            Array built such that policy_1[s,a] represents the probability
            of taking action a from state s under this policy.
            Note that policy_1[s,a] is a uniform distribution over a if s is absorbing.
        policy_2 : ndarray

        Returns
        -------
        KL_divergence : float
            The KL divergence between the two policies.
        """
        return entropy(policy_1.flatten(), policy_2.flatten())

class LocalPolicies:
    """
    A class to represent a collection of local policies.
    """
    
    def __init__(
        self,
        mdp : MDP,
        num_agents : int,
        x : np.ndarray = None,
        joint_policy : JointPolicy = None,
    ):
        """
        Initialize a collection of local policies.
        
        Parameters
        ----------
        mdp :
            An object representing the multiagent MDP on which the policies are to be defined.
        num_agents : int
            The number of agents in the system.
        x : np.ndarray
            An array of occupancy variables.
        """
        self.mdp = mdp
        self.num_agents = num_agents
        
        self.x = x
        
        if x is not None:
            self.construct_policies_from_occupancy_vars(x)
        elif joint_policy is not None:
            self.construct_policies_from_joint_policy(joint_policy)
            
    def sample_local_action(self, agent_ind : int, joint_state : int):
        """
        Sample a local action for a given agent and local state.
        
        Parameters
        ----------
        agent_ind : int
            The index of the agent for which the local action is to be sampled.
        joint_state : int
            The joint state for which the local action is to be sampled.
        
        Returns
        -------
        action : int
            The sampled local action.
        """
        local_policy = self.policies[agent_ind]
        return np.random.choice(np.arange(self.Na_local), p=local_policy[joint_state,:])
        
    def construct_policies_from_joint_policy(self, joint_policy : JointPolicy):
        """
        Construct the local policies by marginalizing the joint policy.
        
        Parameters
        ----------
        joint_policy : JointPolicy
            The joint policy to be marginalized.
        """        
        local_policies = []
    
        # Want to sum over all the actions corresponding to the possible actions of the other agents
        # while this agent is taking its current action.
        
        self.Na_local = round(pow(self.mdp.Na, 1/self.num_agents))

        action_shape = tuple(self.Na_local for i in range(self.num_agents))
        Na_joint_indexes = np.arange(self.mdp.Na)
            
        for i in range(self.num_agents):
            local_policy = np.zeros((self.mdp.Ns, self.Na_local))
            for s in range(self.mdp.Ns):
                for a in range(self.Na_local):
                    compatible_team_actions = np.where(np.unravel_index(Na_joint_indexes, action_shape)[i] == a)
                    local_policy[s, a] = np.sum(joint_policy.policy[s, compatible_team_actions])
            local_policies.append(local_policy)
            
        self.policies = local_policies

    def construct_policies_from_occupancy_vars(self, x : np.ndarray) -> None:
        """
        Construct the local policies from an array of occupancy variables.
        
        Parameters
        ----------
        x : np.ndarray
            An array of occupancy variables.
        """
        self.x = x
        joint_policy = JointPolicy(self.mdp, self.num_agents, x=x)
        self.construct_policies_from_joint_policy(joint_policy)
        
    def compute_product_policy(self) -> np.ndarray:
        """
        Given a the current collection of local policies (one for each agent), 
        compute the joint policy that results from having each agent 
        independently selecting actions from their local policy.
            
        Returns
        -------
        policy : ndarray
            Array built such that policy[s,a] represents the probability
            of taking joint action a from joint state s under this policy.
            Note that policy[s,a] is a uniform distribution over a if s is absorbing.
        """

        Ns_joint = self.mdp.Ns
        Na_joint = self.mdp.Na
        
        Na_local = round(pow(Na_joint, 1/self.num_agents))
        action_shape = tuple(Na_local for i in range(self.num_agents))
        
        policy = np.zeros((Ns_joint, Na_joint))
        
        for s in range(Ns_joint):
            for a in range(Na_joint):
                for i in range(self.num_agents):
                    local_policy = self.policies[i]
                    local_action = np.unravel_index(a, action_shape)[i]
                                    
                    if i == 0:
                        policy[s, a] = local_policy[s, local_action]
                    else:
                        policy[s,a] *= local_policy[s, local_action]
                    
        return policy

    def kl_divergence_joint_and_marginalized_policies(
        self,
        joint_policy : JointPolicy,
    ) -> float:
        """
        Compute the KL divergence between a joint policy and the product of 
        this collection of local policies.
        
        Parameters
        ----------
        joint_policy : JointPolicy
            A joint policy.
            
        Returns
        -------
        KL_divergence : float
        """
        return kl_divergence_policies(joint_policy.policy, self.compute_product_policy())

class LocalPoliciesAcyclicDependencies:
    """
    A class to represent a collection of local policies with a 
    directed graph encoding dependencices. An edge (i, j) in the 
    dependency graph indicates that agent i's policy depends on 
    agent j's local state.
    """

    def __init__(
        self,
        mdp : MDP,
        num_agents : int,
        dependency_structure : list,
        x : np.ndarray = None,
    ):
        """
        Initialize a collection of local policies with a given dependency structure.
        
        Parameters
        ----------
        mdp :
            An object representing the multiagent MDP on which the policies are to be defined.
        num_agents : int
            The number of agents in the system.
        dependency_structure : list
            A list of tuples (i, j) indicating that agent i's policy depends on agent j's local state.
        x : np.ndarray
            An array of occupancy variables.
        """
        self.mdp = mdp
        self.num_agents = num_agents
        self.x = x
      
        self.dependency_graph = nx.DiGraph()
        self.dependency_graph.add_nodes_from(range(num_agents))
        self.dependency_graph.add_edges_from(dependency_structure)
            
        assert self.check_acyclic_dependency_structure(), "Dependency structure must be acyclic."
        
        if x is not None:
            self.construct_policies_from_occupancy_vars(x)

    def construct_policies_from_occupancy_vars(self, x : np.ndarray):
        """
        Build a policy from the occupancy measure values.

        Parameters
        ----------
        x :
            Array built such that x[s,a] represents the occupancy 
            measure of state-action pair (s,a).

        Returns
        -------
        policy : ndarray
            Array built such that policy[s,a] represents the probability
            of taking action a from state s under this policy.
            Note that policy[s,a] is a uniform distribution over a if s is absorbing.
        """
        self.x = x
        
        Na_local = round(pow(self.mdp.Na, 1/self.num_agents))
        Na = (Na_local + 1) ** self.num_agents

        action_shape = tuple([Na_local + 1 for i in range(self.num_agents)])
        
        local_auxiliary_action_index = Na_local

        true_actions = []
        partially_auxiliary_actions = []
        all_auxiliary_actions = []
        for a_joint in range(Na):
            a_tuple = np.unravel_index(a_joint, action_shape)
            if np.sum(np.array(a_tuple) == local_auxiliary_action_index) == 0:
                true_actions.append(a_joint)
            elif np.sum(np.array(a_tuple) == local_auxiliary_action_index) == self.num_agents:
                all_auxiliary_actions.append(a_joint)
            else:
                partially_auxiliary_actions.append(a_joint)

        x_mod = np.copy(x)[:, true_actions]

        policies = []

        for i in range(self.num_agents):
            pass
            # local_policy = np.zeros((self.mdp.Ns, self.mdp.Na))
            # for s in range(self.mdp.Ns):
            #     for a in range(self.mdp.Na):
            #         if not (np.sum(x_mod[s,:]) == 0.0):
            #             policy[s,a] = x_mod[s, a] / np.sum(x_mod[s, :])
            #         else:
            #             policy[s,a] = 1.0 / len(x_mod[s,:])
        self.policies = policies

    def check_acyclic_dependency_structure(self):
        """
        Check that the dependency structure is acyclic.
        """
        return nx.is_directed_acyclic_graph(self.dependency_graph)

    def sample_local_action(self, joint_state_tuple : tuple, agent_id : int):
        """
        Return the local action for a given agent in a given joint state.
        
        Parameters
        ----------
        joint_state_tuple : tuple
            A tuple representing the joint state. The ith element of the tuple
            represents the local state of the ith agent.
        agent_id : int
            The id of the agent whose local action is to be sampled.
            
        Returns
        -------
        action : int
            The sampled local action of the agent.
        """
        local_policy = self.policies[agent_id]
        
        dependence_inds = self.get_indeces_of_this_agents_dependencies(agent_id)
        local_policy_input = [joint_state_tuple[i] for i in dependence_inds]

        return np.random.choice(np.arange(self.Na_local), p=local_policy[local_policy_input,:])

    def get_indeces_of_this_agents_dependencies(self, agent_id : int):
        """
        Return the indeces of the agents that this agent depends on.
        """
        dependencies = [idx for idx in self.dependency_graph.successors(agent_id)]
        dependencies.append(agent_id)
        return dependencies.sort()


if '__main__' == __name__:
    
    num_agents = 4
    dependency_structure = [(0, 1), (0, 2), (0, 3)]
    acyclic_policies = LocalPolicies(num_agents, dependency_structure)
    print(acyclic_policies.dependency_graph.edges)
    print(acyclic_policies.dependency_graph.nodes)