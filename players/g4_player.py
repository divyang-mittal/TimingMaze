from collections import defaultdict
import random
import time
from constants import WAIT, LEFT, UP, RIGHT, DOWN
import os
import pickle
import numpy as np
import logging

import constants
from timing_maze_state import TimingMazeState
from players.g4.gridworld import GridWorld
from players.g4.mcts import MCTS

# from gridworld import GridWorld
# from qtable import QTable
# from q_policy import QPolicy
# from multi_armed_bandit.ucb import UpperConfidenceBounds
from sympy import divisors


# class MCTSNode:

#     # Record a unique node id to distinguish duplicated states
#     next_node_id = 0

#     # Records the number of times states have been visited
#     visits = defaultdict(lambda: 0)

#     def __init__(
#         self, mdp, parent, state, qfunction, bandit, reward=0.0, parent_action=None
#     ):
#         self.mdp = mdp
#         self.parent = parent
#         self.state = state
#         self.id = MCTSNode.next_node_id
#         MCTSNode.next_node_id += 1

#         # The Q function used to store state-action values
#         self.qfunction = qfunction

#         # A multi-armed bandit for this node
#         self.bandit = bandit

#         # The immediate reward received for reaching this state, used for backpropagation
#         self.reward = reward

#         # The action that generated this node
#         self.parent_action = parent_action

#         # A dictionary from actions to a set of node-probability pairs
#         self.children = {}

#     """ Return the value of this node """

#     def get_value(self):
#         max_q_value = self.qfunction.get_max_q(
#             self.state, self.mdp.get_actions(self.state)
#         )
#         return max_q_value

#     """ Get the number of visits to this state """

#     def get_visits(self):
#         return MCTSNode.visits[self.state]

#     """ Return true if and only if all child actions have been expanded """

#     def is_fully_expanded(self):
#         valid_actions = [
#             WAIT,
#             LEFT,
#             UP,
#             RIGHT,
#             DOWN,
#         ]  ### figure out a way to get valid actions, this is okay for now
#         if len(valid_actions) == len(self.children):
#             return True
#         else:
#             return False

#     """ Select a node that is not fully expanded """

#     def select(self):
#         if not self.is_fully_expanded() or self.mdp.is_terminal(self.state):
#             return self
#         else:
#             actions = list(self.children.keys())
#             action = self.bandit.select(self.state, actions, self.qfunction)
#             return self.get_outcome_child(action).select()

#     """ Expand a node if it is not a terminal node """

#     def expand(self):
#         if not self.mdp.is_terminal(self.state):
#             # Randomly select an unexpanded action to expand
#             actions = self.mdp.get_actions(self.state) - self.children.keys()
#             action = random.choice(list(actions))

#             self.children[action] = []
#             return self.get_outcome_child(action)
#         return self

#     """ Backpropogate the reward back to the parent node """

#     def back_propagate(self, reward, child):
#         action = child.action

#         MCTSNode.visits[self.state] = MCTSNode.visits[self.state] + 1
#         MCTSNode.visits[(self.state, action)] = (
#             MCTSNode.visits[(self.state, action)] + 1
#         )

#         q_value = self.qfunction.get_q_value(self.state, action)
#         delta = (1 / (MCTSNode.visits[(self.state, action)])) * (
#             reward - self.qfunction.get_q_value(self.state, action)
#         )
#         self.qfunction.update(self.state, action, delta)

#         if self.parent != None:
#             self.parent.back_propagate(self.reward + reward, self)

#     """ Simulate the outcome of an action, and return the child node """

#     def get_outcome_child(self, action):
#         # Choose one outcome based on transition probabilities
#         (next_state, reward, done) = self.mdp.execute(self.state, action)

#         # Find the corresponding state and return if this already exists
#         for child, _ in self.children[action]:
#             if next_state == child.state:
#                 return child

#         # This outcome has not occured from this state-action pair previously
#         new_child = MCTSNode(
#             self.mdp, self, next_state, self.qfunction, self.bandit, reward, action
#         )

#         # Find the probability of this outcome (only possible for model-based) for visualising tree
#         # probability = 0.0
#         # for (outcome, probability) in self.mdp.get_transitions(self.state, action):
#         #     if outcome == next_state:
#         #         self.children[action] += [(new_child, probability)]
#         #         return new_child


# class MCTS:
#     def __init__(self, mdp, qfunction, bandit):
#         self.mdp = mdp
#         self.qfunction = qfunction
#         self.bandit = bandit

#     """ Execute the MCTS algorithm from the initial state given, with timeout in seconds """

#     def mcts(self, timeout=1, root_node=None):
#         if root_node is None:
#             root_node = self.create_root_node()

#         start_time = time.time()
#         current_time = time.time()
#         while current_time < start_time + timeout:

#             # Find a state node to expand
#             selected_node = root_node.select()
#             if not self.mdp.is_terminal(selected_node):

#                 child = selected_node.expand()
#                 reward = self.simulate(child)
#                 selected_node.back_propagate(reward, child)

#             current_time = time.time()

#         return root_node

#     """ Create a root node representing an initial state """

#     def create_root_node(self):
#         return MCTSNode(
#             self.mdp, None, self.mdp.get_initial_state(), self.qfunction, self.bandit
#         )

#     """ Choose a random action. Heustics can be used here to improve simulations. """

#     def choose(self, state):
#         return random.choice(self.mdp.get_actions(state))

#     """ Simulate until a terminal state """

#     def simulate(self, node):
#         state = node.state
#         cumulative_reward = 0.0
#         depth = 0
#         while not self.mdp.is_terminal(state):
#             # Choose an action to execute
#             action = self.choose(state)

#             # Execute the action
#             (next_state, reward, done) = self.mdp.execute(state, action)

#             # Discount the reward
#             cumulative_reward += pow(self.mdp.get_discount_factor(), depth) * reward
#             depth += 1

#             state = next_state

#         return cumulative_reward


class Player:
    def __init__(
        self,
        rng: np.random.Generator,
        logger: logging.Logger,
        precomp_dir: str,
        maximum_door_frequency: int,
        radius: int,
    ) -> None:
        """Initialize the player with the basic amoeba information

        Args:
            rng (np.random.Generator): numpy random number generator, use this for same player behavior across run
            logger (logging.Logger): logger use this like logger.info("message")
            maximum_door_frequency (int): the maximum frequency of doors
            radius (int): the radius of the drone
            precomp_dir (str): Directory path to store/load pre-computation
        """

        # precomp_path = os.path.join(precomp_dir, "{}.pkl".format(map_path))

        # # precompute check
        # if os.path.isfile(precomp_path):
        #     # Getting back the objects:
        #     with open(precomp_path, "rb") as f:
        #         self.obj0, self.obj1, self.obj2 = pickle.load(f)
        # else:
        #     # Compute objects to store
        #     self.obj0, self.obj1, self.obj2 = _

        #     # Dump the objects
        #     with open(precomp_path, 'wb') as f:
        #         pickle.dump([self.obj0, self.obj1, self.obj2], f)

        self.rng = rng
        self.logger = logger
        self.maximum_door_frequency = maximum_door_frequency
        self.radius = radius
        self.frequencies_per_cell = defaultdict(
            lambda: set(range(maximum_door_frequency + 1))
        )
        self.turn = 0
        self.start = (0,0)
        self.goal = None
        # self.gridworld = GridWorld()
        # self.qfunction = QTable()

    def set_goal(self, maze_state, curr_x, curr_y):
        coords = maze_state.keys()
        far_coords = [coord for coord in coords if abs(coord[0] - curr_x) + abs(coord[1] - curr_y) > (self.radius // 2) + 1]
        goal = self.rng.choice(far_coords)

        return goal

    def move(self, current_percept) -> int:
        """Function which retrieves the current state of the amoeba map and returns an amoeba movement

        Args:
            current_percept(TimingMazeState): contains current state information
        Returns:
            int: This function returns the next move of the user:
                WAIT = -1
                LEFT = 0
                UP = 1
                RIGHT = 2
                DOWN = 3
        """

        curr_x, curr_y = -current_percept.start_x, -current_percept.start_y
        self.turn += 1
        maze_state = {}
        coords = (float('-inf'), float('-inf'))
        factors = set(divisors(self.turn))
        for dX, dY, door, state in current_percept.maze_state:
            if state == constants.CLOSED:
                self.frequencies_per_cell[(curr_x + dX, curr_y + dY, door)] -= factors
            elif state == constants.OPEN:
                self.frequencies_per_cell[(curr_x + dX, curr_y + dY, door)] &= factors
            elif (curr_x + dX, curr_y + dY, door) not in self.frequencies_per_cell.keys() and state == constants.BOUNDARY:
                self.frequencies_per_cell[(curr_x + dX, curr_y + dY, door)] = {0}

            coords = (curr_x + dX, curr_y + dY)

            if coords not in maze_state.keys():
                maze_state[coords] = [(dX, dY, door, state)]
            else:
                maze_state[coords].append((dX, dY, door, state))

        if current_percept.is_end_visible:
            self.goal = (current_percept.end_x + curr_x, current_percept.end_y + curr_y)
        elif self.goal is not None or self.goal == (curr_x, curr_y):
            self.goal = self.set_goal(maze_state, curr_x, curr_y)

        env = GridWorld((curr_x, curr_y), maze_state, self.goal, current_percept.is_end_visible)
        actions = [constants.LEFT, constants.UP, constants.RIGHT, constants.DOWN, constants.WAIT]
        mcts = MCTS(env, actions, self.frequencies_per_cell, self.turn, self.maximum_door_frequency, maze_state)
        best_node = mcts.mcts((curr_x, curr_y), timeout=0.03)
        best_node_actions = list(best_node.parent.children.keys())
        
        cur_cell = sorted(maze_state[(curr_x, curr_y)], key = lambda x : x[2])
        for action in best_node_actions:
            if action == constants.LEFT:
                adj_cell = sorted(maze_state[(curr_x - 1, curr_y)], key = lambda x : x[2])
                adj_action = constants.RIGHT
                if cur_cell[action][-1] == constants.OPEN and adj_cell[adj_action][-1] == constants.OPEN:
                    best_action = action
                    break
            elif action == constants.UP:
                adj_cell = sorted(maze_state[(curr_x, curr_y - 1)], key = lambda x : x[2])
                adj_action = constants.DOWN
                if cur_cell[action][-1] == constants.OPEN and adj_cell[adj_action][-1] == constants.OPEN:
                    best_action = action
                    break
            elif action == constants.RIGHT:
                adj_cell = sorted(maze_state[(curr_x + 1, curr_y)], key = lambda x : x[2])
                adj_action = constants.LEFT
                if cur_cell[action][-1] == constants.OPEN and adj_cell[adj_action][-1] == constants.OPEN:
                    best_action = action
                    break
            elif action == constants.DOWN:
                adj_cell = sorted(maze_state[(curr_x, curr_y + 1)], key = lambda x : x[2])
                adj_action = constants.UP
                if cur_cell[action][-1] == constants.OPEN and adj_cell[adj_action][-1] == constants.OPEN:
                    best_action = action
                    break
            elif action == constants.WAIT:
                best_action = action
                break
        
        # best_action = list(best_node.parent.children.keys())[0]

        return best_action
