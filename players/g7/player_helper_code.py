import random

from dataclasses import dataclass
import networkx as nx
import matplotlib.pyplot as plt
from math import lcm
import constants
from utils import get_divisors
import numpy as np
import heapq


class MemoryDoor:
    def __init__(self):
        self.is_certain_freq = False
        self.observations = {} # {turn : 1 - Closed / 2 - Open / 3 - Boundary}
        self.freq_distribution = {}
    
    def is_open(self, turn):
        # Returns if the door is open at a given turn
        return self.observations.get(turn, 1) == 2
    
    def update_observations(self, door_state, turn):
        # Updates observed freqs, runs get_freq
        if not self.is_certain_freq:
            self.observations[turn] = door_state
            self.freq_distribution = self.get_freq()
            if len(self.freq_distribution) == 1:
                self.is_certain_freq = True
    
    def get_freq(self):
        # Tries to find the frequency given observed, returns a probability distribution
        possible_open_frequencies = set()
        closed_frequencies = set()

        # Iterate over the frequency conditions
        for turn, status in self.observations.items():
            if status == 2:
                # Add all divisors of the frequency if it's open
                if not possible_open_frequencies:
                    possible_open_frequencies = get_divisors(turn)
                else:
                    possible_open_frequencies &= get_divisors(turn)
            else:
                # Add all divisors of the closed frequency
                closed_frequencies |= get_divisors(turn)

        # Remove any closed frequencies from the possible open set
        possible_open_frequencies -= closed_frequencies

        # Assign equal probabilities to each remaining frequency
        total_frequencies = len(possible_open_frequencies)
        probability_distribution = {
            freq: 1/total_frequencies for 
            freq in possible_open_frequencies} if total_frequencies > 0 else {}

        return probability_distribution
    
    def roll_freq(self):
        # Returns a frequency based on the distribution
            # Calculate cumulative distribution
        if self.freq_distribution == {}:
            return 0
        cumulative_dist = []
        cumulative_sum = 0
        for freq, prob in self.freq_distribution.items():
            cumulative_sum += prob
            cumulative_dist.append((cumulative_sum, freq))
        
        rand = np.random.random()
        
        # Choose the frequency based on the random number
        for cumulative_prob, freq in cumulative_dist:
            if rand <= cumulative_prob:
                return freq

        # In case the random number is exactly 1, return the last frequency
        return cumulative_dist[-1][1]
        

class MemorySquare:
    def __init__(self):
        left = MemoryDoor()
        up = MemoryDoor()
        right = MemoryDoor()
        down = MemoryDoor()
        self.doors = {constants.LEFT:left, constants.UP:up, constants.RIGHT:right, constants.DOWN:down}

class PlayerMemory:
    def __init__(self, map_size: int = 100):
        self.memory = [[MemorySquare() for _ in range(map_size * 2)] for _ in range(map_size * 2)]
        self.pos = (map_size, map_size) #(y, x)
    
    def update_memory(self, state, turn):
        # state = [door] = (row_offset, col_offset, door_type, door_status)
        for s in state:
            # if s[0] == -1 and s[1] == 0:
                # print("ya")
            square = self.memory[self.pos[0] + s[1]][self.pos[1] + s[0]]
            door = square.doors[s[2]]
            door_state = s[3]
            door.update_observations(door_state, turn)

    def update_pos(self, move):
        if move == constants.LEFT:
            self.pos = (self.pos[0], self.pos[1] - 1)
        if move == constants.UP:
            self.pos = (self.pos[0] - 1, self.pos[1])
        if move == constants.RIGHT:
            self.pos = (self.pos[0], self.pos[1] + 1)
        if move == constants.DOWN:
            self.pos = (self.pos[0] + 1, self.pos[1])
    
    def is_move_valid(self, move, state):

        if move == constants.LEFT:
            current_square_left_open = False
            left_square_right_open = False
            for s in state: 
                if s[0] == 0 and s[1] == 0 and s[2] == constants.LEFT and s[3] == 2:
                    current_square_left_open = True
                if s[0] == -1 and s[1] == 0 and s[2] == constants.RIGHT and s[3] == 2:
                    left_square_right_open = True
            return current_square_left_open and left_square_right_open

        if move == constants.UP:
            current_square_up_open = False
            up_square_down_open = False
            for s in state:
                if s[0] == 0 and s[1] == 0 and s[2] == constants.UP and s[3] == 2:
                    current_square_up_open = True
                if s[0] == 0 and s[1] == -1 and s[2] == constants.DOWN and s[3] == 2:
                    up_square_down_open = True
            return current_square_up_open and up_square_down_open

        
        if move == constants.RIGHT:
            current_square_right_open = False
            right_square_left_open = False
            for s in state:
                if s[0] == 0 and s[1] == 0 and s[2] == constants.RIGHT and s[3] == 2:
                    current_square_right_open = True
                if s[0] == 1 and s[1] == 0 and s[2] == constants.LEFT and s[3] == 2:
                    right_square_left_open = True
            return current_square_right_open and right_square_left_open
 
        if move == constants.DOWN:
            current_square_down_open = False
            down_square_up_open = False
            for s in state:
                if s[0] == 0 and s[1] == 0 and s[2] == constants.DOWN and s[3] == 2:
                    current_square_down_open = True
                if s[0] == 0 and s[1] == 1 and s[2] == constants.UP and s[3] == 2:
                    down_square_up_open = True
            return current_square_down_open and down_square_up_open
        
        return False

class MazeGraph:
    def __init__(self, graph: dict = {}):
        self.graph = graph  # A dictionary for the adjacency list
        
    def add_edge(self, node1: tuple, node2: tuple, weight: int):
        if node1 not in self.graph:  # Check if the node is already added
            self.graph[node1] = {}   # If not, create the node
        self.graph[node1][node2] = weight  # Else, add a connection to its neighbor
    
    def add_bidirectional_edge(self, node1: tuple, node2: tuple, node1_door_freq: int, node2_door_freq: int):
        """Adds an edge both ways between node1 and node2."""
        self.add_edge(node1, node2, [node1_door_freq, node2_door_freq])
        self.add_edge(node2, node1, [node2_door_freq, node1_door_freq])
    
    def getMazeDimension(self) -> int:
        # Returns the dimension of the maze (assuming it is a square)
        return int(len(self.graph) ** 0.5)
    
    def getNeighbors(self, node) -> dict[tuple, int]:
        return self.graph[node]


    def visualize_graph_in_grid(self, minDistanceArray=None, parent=None, startNode=None, targetNode=None, 
                                row_slice=None, col_slice=None, figsize=30):
        G = nx.Graph()

        # Add nodes and edges from adjacency list
        for node1, neighbors in self.graph.items():
            for node2, weight in neighbors.items():
                node1Freq, node2Freq = weight

                # Calculate the edge weight
                if node1Freq == 0 or node2Freq == 0:
                    weight = float('inf')
                else:
                    # LCM is the combined frequency of the doors
                    weight = lcm(node1Freq, node2Freq)

                # Only add nodes and edges within the specified row/col slice
                if row_slice and col_slice:
                    if (node1[0] in row_slice and node1[1] in col_slice and
                        node2[0] in row_slice and node2[1] in col_slice):
                        G.add_edge(node1, node2, weight=weight)
                else:
                    G.add_edge(node1, node2, weight=weight)

        grid_dim = len(self.graph) ** 0.5  # Square root of the number of nodes
        grid_dim_int = int(grid_dim)  # Ensure square root is an integer for a square grid
        plt.figure(figsize=(figsize, figsize))

        # Define positions for nodes in a grid layout, restricted to the row and column slices
        pos = {(i, j): (j, -i) for i in range(grid_dim_int) for j in range(grid_dim_int)
            if (not row_slice or i in row_slice) and (not col_slice or j in col_slice)}

        # Draw the graph in the grid layout
        nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10)

        # Highlight the path if startNode and targetNode are provided
        if minDistanceArray and parent and startNode and targetNode:
            print(targetNode)
            targetx, targety = targetNode
            if minDistanceArray[targetx][targety] == float('inf'):
                print(f"Target node {targetNode} is unreachable.")
                return
            path = reconstruct_path(parent, startNode, targetNode)

            # Collect the edges in the path
            path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]

            # Get turn information for nodes in the path
            path_turns = {node: parent[node][1] for node in path if node in parent}

            # Highlight the path nodes in a different color
            nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='yellow', node_size=700)

            # Draw the edges in the path with arrows, direction, and increased width
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=3, arrows=True)

            # Label the nodes with the turn number
            for node, (x, y) in pos.items():
                if node in path_turns:
                    plt.text(x, y + 0.1, f"Turn {path_turns[node]}", fontsize=9, ha='center', color='blue')

        # Draw edge labels (combined frequency of the adjacent doors)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        print("Making graph")
        plt.savefig("graph.png", format="png", dpi=300)
        # plt.show()
        
def reconstruct_path(parent, startNode, targetNode):
    """Helper function to reconstruct the path from startNode to targetNode using the parent dictionary."""
    path = []
    # we are storing y, x in startNode (if y is row and x is xol...)
    currentNode = (int(startNode[0] + targetNode[0]), int(startNode[1] + targetNode[1]))
    while currentNode is not None:
        if currentNode not in parent:
            print("Target Node is unreachable")
            return None # TargetNode is unreachable
        path.append(currentNode)
        currentNode = parent.get(currentNode)[0]  # Get the parent node
    path.reverse()  # Reverse the path to get it from start to target
    return path

def build_graph_from_memory(player_memory: PlayerMemory) -> MazeGraph:
    graph = MazeGraph()

    # Iterate through the map_memory and add edges to the graph
    for y in range(len(player_memory.memory)): # i is the row index (basically y)
        for x in range(len(player_memory.memory[0])): # j is the column index (basically x)
            currMemSquare: MemorySquare = player_memory.memory[y][x]

            # Coordinates for neighboring cells
            neighbors = {
                'left': (y, x-1),
                'right': (y, x+1),
                'up': (y-1, x),
                'down': (y+1, x)
            }
            # test = [[(0,0),(0,1), (0,2), (0,3), (0,4)],
            #  [(1,0),(1,1), (1,2), (1,3), (1,4)],
            #  [(2,0),(2,1), (2,2), (2,3), (2,4)],
            #  [(3,0),(3,1), (3,2), (3,3), (3,4)],
            #  [(4,0),(4,1), (4,2), (4,3), (4,4)]]
            
            # Add edges based on door frequencies and valid neighboring cells
            
            # Left neighbor exists 
            if x > 0:
                leftSquare: MemorySquare = player_memory.memory[y][x - 1]
                graph.add_bidirectional_edge((y, x), neighbors['left'],
                                                currMemSquare.doors[constants.LEFT].roll_freq(),
                                                leftSquare.doors[constants.RIGHT].roll_freq())

            # Right neighbor exists 
            if x < len(player_memory.memory[0]) - 1:  
                rightSquare: MemorySquare = player_memory.memory[y][x + 1]
                graph.add_bidirectional_edge((y, x), neighbors['right'],
                                                currMemSquare.doors[constants.RIGHT].roll_freq(),
                                                rightSquare.doors[constants.LEFT].roll_freq())
            
            # Up neighbor exists 
            if y > 0: 
                upSquare: MemorySquare = player_memory.memory[y - 1][x]
                graph.add_bidirectional_edge((y, x), neighbors['up'],
                                                currMemSquare.doors[constants.UP].roll_freq(),
                                                upSquare.doors[constants.DOWN].roll_freq())
            
            # Down neighbor exists
            if y < len(player_memory.memory) - 1:
                downSquare: MemorySquare = player_memory.memory[y + 1][x]
                graph.add_bidirectional_edge((y, x), neighbors['down'],
                                                currMemSquare.doors[constants.DOWN].roll_freq(),
                                                downSquare.doors[constants.UP].roll_freq())

    return graph


def findShortestPathsToEachNode(graph: MazeGraph, startNode: tuple, turnNumber: int):
    dimension = graph.getMazeDimension()

    # Initialize the minDistanceArray with infinity
    minDistanceArray = [[float('inf')] * dimension for _ in range(dimension)]
    minDistanceArray[startNode[0]][startNode[1]] = 0  # Start node has distance 0

    # Initialize the parent dictionary to track the shortest path and the turn when we moved to this node
    # parent[(x, y)] = (parentNode, turnWeMovedToNode)
    parent = {startNode: (None, turnNumber)}  # Initially at startNode at the given turnNumber

    # Min-heap stores (distance, (x, y) node)
    minHeap = [(0, startNode)]  # Start node with distance 0

    visitedNodes = set()

    # Process the heap until it is empty
    while minHeap:
        turnsToCurrentNode, currentNode = heapq.heappop(minHeap)
        # currentXCoord, currentYCoord = currentNode         print("visiting: ", currentNode)

        # Skip node if already visited
        if currentNode in visitedNodes:
            continue

        visitedNodes.add(currentNode)

        # Get neighbors of the current node from the graph
        neighbors: dict[tuple, tuple] = graph.getNeighbors(currentNode)


        for (yCoordNeighbour, xCoordNeighbour), (node1Freq, node2Freq) in neighbors.items():
            # Combined Frequency of the doors
            if node1Freq == 0 or node2Freq == 0:
                combinedFrequencey = float('inf')
            else:
                # LCM is the combined frequency of the doors
                combinedFrequencey = lcm(node1Freq, node2Freq)
    
            # Determine the turn number when we reach this node
            turnWeWillBeAtThisNode = turnNumber + turnsToCurrentNode

            # Calculate the number of turns we need to wait for the door to open
            turnsToWait = (combinedFrequencey - (turnWeWillBeAtThisNode % combinedFrequencey)) % combinedFrequencey
            # at the node at turn 19. 19 % 12 = 7... 12 -7 = 5  % combinedFrequencey = 5

            newTurnsToGetToNeighbor = turnsToCurrentNode + turnsToWait + 1

            # Update the neighbor's distance if a shorter path is found
            if newTurnsToGetToNeighbor < minDistanceArray[yCoordNeighbour][xCoordNeighbour]:
                minDistanceArray[yCoordNeighbour][xCoordNeighbour] = newTurnsToGetToNeighbor
                # Store both the parent node and the turn at which we moved to this neighbor
                parent[(yCoordNeighbour, xCoordNeighbour)] = (currentNode, turnWeWillBeAtThisNode + turnsToWait + 1)
                heapq.heappush(minHeap, (newTurnsToGetToNeighbor, (yCoordNeighbour, xCoordNeighbour)))

    return minDistanceArray, parent  # Return both distances and paths


def print_min_dist_array(minDistanceArray, start_row, end_row, start_col, end_col, width=4):
    for y in range(len(minDistanceArray)):
        if y >= start_row and y <= end_row:
            row = minDistanceArray[y]
            for x in range(len(row)):
                if x >= start_col and x <= end_col:
                    # Print each element with a fixed width
                    print(f"{row[x]:>{width}}", end=" ")
            print()