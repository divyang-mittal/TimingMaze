import numpy as np
import random
from collections import defaultdict
import time
import math
import constants

# Node class for MCTS
class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self, actions):
        return len(self.children) == len(actions)
    
    def heuristic(self, state, target):
        # manhattan distance
        return abs(state[0] - target[0]) + abs(state[1] - target[1])

    def best_child(self, target=None, c_param=1.4):
        # UCB1 with heuristics
        choices_weights = [
            (child.value / child.visits) + c_param * np.sqrt(np.log(self.visits) / child.visits) - self.heuristic(child.state, target)
            if target is not None else (child.value / child.visits) + c_param * np.sqrt(np.log(self.visits) / child.visits) 
            for child in self.children.values()
        ]
        return list(self.children.values())[np.argmax(choices_weights)]
    
    def expand(self, action, next_state):
        child_node = Node(state=next_state, parent=self)
        self.children[action] = child_node
        return child_node


# Monte Carlo Tree Search Class
class MCTS:
    def __init__(self, env, actions, frequencies, turn, max_freq, maze_state):
        self.env = env
        self.actions = actions
        self.frequencies = frequencies
        self.turn = turn
        self.max_freq = max_freq
        self.maze_state = maze_state

    def mcts(self, root_state=None, timeout=1):
        root = Node(state=root_state)
        start_time = time.time()

        while time.time() - start_time < timeout:
            # 1. Selection
            node = self.selection(root)

            # 2. Expansion
            if not node.is_fully_expanded(self.actions):
                action = self.random_untried_action(node)
                next_state = self.env.get_next_state(node.state, action)
                node = node.expand(action, next_state)

            # 3. Simulation
            reward = self.simulate(node.state)

            # 4. Backpropagation
            self.backpropagate(node, reward)

        return root.best_child(target=self.env.goal, c_param=1.4)

    # navigate tree selecting best children nodes until node with unexplored children reached
    def selection(self, node):
        while node.is_fully_expanded(self.actions):
            node = node.best_child()
        return node

    # randomly select an action which has not yet been explored
    def random_untried_action(self, node):
        tried_actions = set(node.children.keys())
        untried_actions = list(set(self.actions) - tried_actions)
        return random.choice(untried_actions)
        
    # def angular_dist(self, diff):
    #     return min(abs(diff), 360 - abs(diff))
    
    def compare_manhattan_dist(self, cur, adj):
        # compute manhattan distances
        cur_dist = abs(cur[0] - self.env.goal[0]) + abs(cur[1] - self.env.goal[1])
        adj_dist = abs(adj[0] - self.env.goal[0]) + abs(adj[1] - self.env.goal[1])

        if adj_dist < cur_dist:
            return 1
        elif adj_dist == cur_dist:
            return 0
        else:
            return -1
    
    def compute_score(self, cur, adj, action, turn=None, wait=False):
        if turn == None:
            turn = self.turn

        cur_state = sorted(self.maze_state[cur], key=lambda x : x[2])

        # discourage going through boundaries
        if cur_state[action][-1] == constants.BOUNDARY:
            return -10
        
        try:
            adj_state = sorted(self.maze_state[adj[0]], key=lambda x : x[2])
        except:
            return -1

        if not wait:
            # Reward if both doors are open and punish if not
            if cur_state[action][-1] != constants.OPEN or adj_state[adj[1]][-1] != constants.OPEN:
                return -10
            
            if cur_state[action][-1] == constants.OPEN and adj_state[adj[1]][-1] == constants.OPEN:
                return 1
        # else:
        #     open_score = 0.5
        else:
            # give scores based on how many doors are likely to open at given turn
            cur_freq = self.frequencies[(*cur, action)]
            adj_freq = self.frequencies[(*adj[0], adj[1])]
            cum_freq = cur_freq & adj_freq

            # base case
            if len(cum_freq) == self.max_freq:
                return 0
            
            if len(cum_freq) == 0:
                return -1
            
            mod_count = 0
            for freq in cum_freq:
                if freq == 0:
                    continue
                if turn % freq == 0:
                    mod_count += 1
            
            score = mod_count / len(cum_freq)

            print(score)
            return score
    
    def choose_action(self, state, turn=None):
        if turn is None:
            turn = self.turn 

        # xdist = self.env.goal[0] - state[0]
        # ydist = -(self.env.goal[1] - state[1])
        # if xdist != 0:
        #     theta = math.degrees(math.atan(ydist / xdist))
        # else:
        #     theta = 90 * (ydist/abs(ydist))

        # if (xdist < 0 and ydist >= 0) or (xdist < 0 and ydist < 0):
        #     theta += 180
        # elif xdist > 0 and ydist < 0:
        #     theta += 360

        weights = np.zeros(5)

        wait_score = 0
        for action in self.actions[:-1]:
            
            # give scores for each action
            if action == constants.LEFT:
                # ang_dist_score = 1 - self.angular_dist(180 - theta) / 180
                # weights[0] = ang_dist_score
                adj = ((state[0] - 1, state[1]), 2)
                manhattan_score = self.compare_manhattan_dist(state, adj[0])
                weights[0] = manhattan_score
                weights[0] += self.compute_score(state, adj, action, turn, False)

                # give score based on chances if wait a turn then take specified action
                wait_score_left = self.compute_score(state, adj, action, self.turn + 1, True) + manhattan_score
                wait_score = wait_score_left if wait_score_left > wait_score else wait_score

            elif action == constants.UP:
                # ang_dist_score = 1 - self.angular_dist(180 - theta) / 180
                # weights[1] = ang_dist_score
                adj = ((state[0], state[1] - 1), 3)
                manhattan_score = self.compare_manhattan_dist(state, adj[0])
                weights[1] = manhattan_score
                weights[1] += self.compute_score(state, adj, action, turn, False)

                wait_score_up = self.compute_score(state, adj, action, self.turn + 1, True) + manhattan_score
                wait_score = wait_score_up if wait_score_up > wait_score else wait_score

            elif action == constants.RIGHT:
                # ang_dist_score = 1 - self.angular_dist(180 - theta) / 180
                # weights[2] = ang_dist_score
                adj = ((state[0] + 1, state[1]), 0)
                manhattan_score = self.compare_manhattan_dist(state, adj[0])
                weights[2] = manhattan_score
                weights[2] += self.compute_score(state, adj, action, turn, False)

                wait_score_right = self.compute_score(state, adj, action, self.turn + 1, True) + manhattan_score
                wait_score = wait_score_right if wait_score_right > wait_score else wait_score

            elif action == constants.DOWN:
                # ang_dist_score = 1 - self.angular_dist(180 - theta) / 180
                # weights[3] = ang_dist_score
                adj = ((state[0], state[1] + 1), 1)
                manhattan_score = self.compare_manhattan_dist(state, adj[0])
                weights[3] = manhattan_score
                weights[3] += self.compute_score(state, adj, action, turn, False)

                wait_score_down = self.compute_score(state, adj, action, self.turn + 1, True) + manhattan_score
                wait_score = wait_score_down if wait_score_down > wait_score else wait_score

            weights[-1] = wait_score 
        
        # if all weights are 0, then wait
        if max(weights) == 0:
            best_action = -1
        else:
            # check if best action is valid; if not, take next best action
            best_action = np.argmax(weights) if np.argmax(weights) != 4 else -1
            for i in range(len(weights)):
                if not self.is_valid_move(state, best_action):
                    weights[best_action] = float('-inf')

            best_action = np.argmax(weights) if np.argmax(weights) != 4 else -1

        return best_action

    def simulate(self, state):
        current_state = state
        prev_state = None
        cumulative_reward = 0.0
        # while not self.env.is_goal(current_state):
        for i in range(10):
            # very naive way of setting up a reward system
            if self.env.is_goal(current_state) and self.env.is_end_visible:
                cumulative_reward += 100
            elif self.env.is_goal(current_state) and not self.env.is_end_visible:
                cumulative_reward += 1
            elif prev_state is not None:
                # give reward if closer to target
                if self.compare_manhattan_dist(current_state, prev_state) == 1:
                    cumulative_reward += 10
            else:
                cumulative_reward -= 0.01
            
            action = self.choose_action(current_state, self.turn + i)
            prev_state = current_state
            current_state = self.env.get_next_state(current_state, action)
        return cumulative_reward
    
    def is_valid_move(self, state, action):
        curr_cell = sorted(self.maze_state[state], key = lambda x : x[2])

        if action == constants.LEFT:
            adj_cell = sorted(self.maze_state[state[0] - 1, state[1]], key = lambda x : x[2])
            return curr_cell[action][-1] == constants.OPEN and adj_cell[2][-1] == constants.OPEN
        elif action == constants.UP:
            adj_cell = sorted(self.maze_state[state[0], state[1] - 1], key = lambda x : x[2])
            return curr_cell[action][-1] == constants.OPEN and adj_cell[3][-1] == constants.OPEN
        elif action == constants.RIGHT:
            adj_cell = sorted(self.maze_state[state[0] + 1, state[1]], key = lambda x : x[2])
            return curr_cell[action][-1] == constants.OPEN and adj_cell[0][-1] == constants.OPEN
        elif action == constants.DOWN:
            adj_cell = sorted(self.maze_state[state[0], state[1] + 1], key = lambda x : x[2])
            return curr_cell[action][-1] == constants.OPEN and adj_cell[1][-1] == constants.OPEN
        
        return False

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent