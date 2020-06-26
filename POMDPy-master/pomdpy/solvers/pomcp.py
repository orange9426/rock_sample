from __future__ import absolute_import
from __future__ import division
from builtins import range
from past.utils import old_div
import time
import numpy as np
from pomdpy.util import console
from pomdpy.action_selection import ucb_action
from pomdpy.action_selection import softmax_action
from .belief_tree_solver import BeliefTreeSolver

module = "pomcp"


class POMCP(BeliefTreeSolver):
    """
    Monte-Carlo Tree Search implementation, from POMCP
    """

    # Dimensions for the fast-UCB table
    UCB_N = 10000
    UCB_n = 100

    def __init__(self, agent):
        """
        Initialize an instance of the POMCP solver
        :param agent:
        :param model:
        :return:
        """
        super(POMCP, self).__init__(agent)

        # Pre-calculate UCB values for a speed-up
        if agent.model.solver == 'POMCP':
            self.fast_UCB = [[None for _ in range(POMCP.UCB_n)] for _ in range(POMCP.UCB_N)]

            for N in range(POMCP.UCB_N):
                for n in range(POMCP.UCB_n):
                    if n is 0:
                        self.fast_UCB[N][n] = np.inf
                    else:
                        self.fast_UCB[N][n] = agent.model.ucb_coefficient * np.sqrt(old_div(np.log(N + 1), n))
        
        elif agent.model.solver == 'ME-POMCP':
            self.fast_lambda = [None for _ in range(POMCP.UCB_N)]

            for N in range(POMCP.UCB_N):
                if N is 0:
                    self.fast_lambda[N] = 1
                else:
                    self.fast_lambda[N] = agent.model.me_epsilon / np.log2(N + 1)

    @staticmethod
    def reset(agent):
        """
        Generate a new POMCP solver

        :param agent:
        Implementation of abstract method
        """
        return POMCP(agent)

    def find_fast_ucb(self, total_visit_count, action_map_entry_visit_count, log_n):
        """
        Look up and return the value in the UCB table corresponding to the params
        :param total_visit_count:
        :param action_map_entry_visit_count:
        :param log_n:
        :return:
        """
        assert self.fast_UCB is not None
        if total_visit_count < POMCP.UCB_N and action_map_entry_visit_count < POMCP.UCB_n:
            return self.fast_UCB[int(total_visit_count)][int(action_map_entry_visit_count)]

        if action_map_entry_visit_count == 0:
            return np.inf
        else:
            return self.model.ucb_coefficient * np.sqrt(old_div(log_n, action_map_entry_visit_count))
    
    # def find_fast_lambda(self, total_visit_count, action_map_entry_visit_count, log_n):
    #     """
    #     Look up and return the value in the UCB table corresponding to the params
    #     :param total_visit_count:
    #     :param action_map_entry_visit_count:
    #     :param log_n:
    #     :return:
    #     """
    #     assert self.fast_UCB is not None
    #     if total_visit_count < POMCP.UCB_N and action_map_entry_visit_count < POMCP.UCB_n:
    #         return self.fast_UCB[int(total_visit_count)][int(action_map_entry_visit_count)]

    #     if action_map_entry_visit_count == 0:
    #         return np.inf
    #     else:
    #         return self.model.ucb_coefficient * np.sqrt(old_div(log_n, action_map_entry_visit_count))

    def select_eps_greedy_action(self, eps, start_time):
        """
        Starts off the Monte-Carlo Tree Search and returns the selected action. If the belief tree
                data structure is disabled, random rollout is used.
        """
        if self.disable_tree:
            self.rollout_search(self.belief_tree_index)
        else:
            self.monte_carlo_approx(eps, start_time)
        return ucb_action(self, self.belief_tree_index, True)

    def simulate(self, belief_node, eps, start_time):
        """
        :param belief_node:
        :return:
        """
        state = belief_node.sample_particle()
        return self.traverse(state, belief_node, 0, start_time)

    def traverse(self, state, belief_node, tree_depth, start_time):
        delayed_reward = 0

        # Time expired
        if time.time() - start_time > self.model.action_selection_timeout:
            console(4, module, "action selection timeout")
            return 0

        if self.model.solver == 'POMCP':
            action = ucb_action(self, belief_node, False)
        elif self.model.solver == 'ME-POMCP':
            action = softmax_action(self, belief_node, False)

        # Search horizon reached
        if tree_depth >= self.model.max_depth:
            console(4, module, "Search horizon reached")
            return 0

        step_result, is_legal = self.model.generate_step(state, action)

        if not step_result.is_terminal:
            child_belief_node, added = belief_node.create_or_get_child(action, step_result.observation)
            tree_depth += 1
            # Add S' to the new belief node
            # Add a state particle with the new state
            if child_belief_node.state_particles.__len__() < self.model.max_particle_count:
                child_belief_node.state_particles.append(step_result.next_state)
            if not added:
                delayed_reward = self.traverse(step_result.next_state, child_belief_node, tree_depth, start_time)
                if self.model.solver == 'ME-POMCP':
                    softmax = self.model.me_tau * np.log(sum([np.exp(action_entry.mean_q_value / self.model.me_tau)
                                                          for action_entry in child_belief_node.action_map.entries.values()]))
                    delayed_reward = softmax
            else:
                delayed_reward = self.rollout(step_result.next_state, child_belief_node)
            tree_depth -= 1
        else:
            console(4, module, "Reached terminal state.")

        # delayed_reward is "Q maximal"
        # current_q_value is the Q value of the current belief-action pair
        action_mapping_entry = belief_node.action_map.get_entry(action.bin_number)

        # off-policy Q learning update rule
        q_value = step_result.reward + self.model.discount * delayed_reward

        action_mapping_entry.update_visit_count(1)
        action_mapping_entry.update_q_value(q_value)

        # Add RAVE ?
        return q_value
