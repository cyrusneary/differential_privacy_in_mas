import numpy as np
import networkx as nx
from mdp import MDP

class LocalPolicies:
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
        dependency_structure : list = None, 
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
        """
        self.mdp = mdp
        self.num_agents = num_agents
      
        if dependency_structure is None:
            G_fully_connected = nx.complete_graph(num_agents)
            self.dependency_graph = nx.DiGraph(G_fully_connected)
        else:
            self.dependency_graph = nx.DiGraph()
            self.dependency_graph.add_nodes_from(range(num_agents))
            self.dependency_graph.add_edges_from(dependency_structure)

    def construct_policies_from_occupancy_vars(mdp : MDP, x : np.ndarray, N_agents : int):
        """
        Build a policy from the occupancy measure values.

        Parameters
        ----------
        mdp :
            An object representing the MDP on which the reachability problem
            is to be solved.
        x :
            Array built such that x[s,a] represents the occupancy 
            measure of state-action pair (s,a).
        N_agents :
            The number of agents.

        Returns
        -------
        policy : ndarray
            Array built such that policy[s,a] represents the probability
            of taking action a from state s under this policy.
            Note that policy[s,a] is a uniform distribution over a if s is absorbing.
        """
        Na_local = round(pow(mdp.Na, 1/N_agents))
        Na = (Na_local + 1) ** N_agents

        action_shape = tuple([Na_local + 1 for i in range(N_agents)])
        
        local_auxiliary_action_index = Na_local

        true_actions = []
        partially_auxiliary_actions = []
        all_auxiliary_actions = []
        for a_joint in range(Na):
            a_tuple = np.unravel_index(a_joint, action_shape)
            if np.sum(np.array(a_tuple) == local_auxiliary_action_index) == 0:
                true_actions.append(a_joint)
            elif np.sum(np.array(a_tuple) == local_auxiliary_action_index) == N_agents:
                all_auxiliary_actions.append(a_joint)
            else:
                partially_auxiliary_actions.append(a_joint)

        x_mod = np.copy(x)[:, true_actions]

        policy = np.zeros((mdp.Ns, mdp.Na))
        for s in range(mdp.Ns):
            for a in range(mdp.Na):
                if not (np.sum(x_mod[s,:]) == 0.0):
                    policy[s,a] = x_mod[s, a] / np.sum(x_mod[s, :])
                else:
                    policy[s,a] = 1.0 / len(x_mod[s,:])
        return policy

    def check_acyclic_dependency_structure(self):
        """
        Check that the dependency structure is acyclic.
        """
        return nx.is_directed_acyclic_graph(self.dependency_graph)

    def get_local_action(self, joint_state : tuple, agent_id : int):
        """
        Return the local action for a given agent in a given joint state.
        """
        pass

    def get_indeces_of_this_agents_dependencies(self, agent_id : int):
        """
        Return the indeces of the agents that this agent depends on.
        """
        dependencies = [idx for idx in self.dependency_graph.successors(agent_id)]
        dependencies.append(agent_id)
        return dependencies.sort()

class JointPolicy:

    def __init__(self) -> None:
        
        pass

    def get_joint_action(self, joint_state : tuple):
        """
        Return the joint action for a given joint state.
        """
        pass

    def construct_joint_policy_from_occupancy_vars(x : np.ndarray, num_agents : int):
        """
        Construct a joint policy from a vector of occupancy variables.

        Parameters
        ----------
        x : np.ndarray
            An array of occupancy variables.
        num_agents : int
            The number of agents in the system.
        """
        pass

if '__main__' == __name__:
    
    num_agents = 4
    dependency_structure = [(0, 1), (0, 2), (0, 3)]
    acyclic_policies = LocalPolicies(num_agents, dependency_structure)
    print(acyclic_policies.dependency_graph.edges)
    print(acyclic_policies.dependency_graph.nodes)