import random

from dataclasses import dataclass
import networkx as nx
import matplotlib.pyplot as plt
from math import lcm

@dataclass
class Square():
    leftFreq: int
    upFreq: int
    rightFreq: int
    downFreq: int

def getFourRandomInts(lowerBound=0, upperBound=4) -> list[int]:
    return [random.randint(lowerBound, upperBound) for _ in range(4)]

def addEdgeFrequencies(mem_map) -> list[list[Square]]:
    mapDim = len(mem_map)
    for i in range(mapDim):
        for j in range(mapDim):
            if i == 0:
                mem_map[i][j].upFreq = 0
            if i == mapDim - 1:
                mem_map[i][j].downFreq = 0
            if j == 0:
                mem_map[i][j].leftFreq = 0
            if j == mapDim - 1:
                mem_map[i][j].rightFreq = 0
    return mem_map

# Generate array of Square objects
def generateMemoryMap(dim: int) -> list[list[Square]]:
    memMap = []
    for i in range(dim):
        row = []
        for j in range(dim):
            # we can add some logic here to make certain things happen
            row.append(Square(*getFourRandomInts(lowerBound=0, upperBound=4)))
        memMap.append(row)
    memMap = addEdgeFrequencies(memMap)
    return memMap

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
    

    def visualize_graph_in_grid(self, minDistanceArray=None, parent=None, startNode=None, targetNode=None):
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
                
                G.add_edge(node1, node2, weight=weight)
        
        grid_dim = len(self.graph) ** 0.5 # Square root of number of nodes
        grid_dim_int = int(grid_dim) # Square root should always be an int for a square grid.
        plt.figure(figsize=(30, 30))  
        # Define positions for nodes in a grid layout
        pos = {(i, j): (j, -i) for i in range(grid_dim_int) for j in range(grid_dim_int)}
        
        # Draw the graph in the grid layout
        nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=10)

        # Highlight the path if startNode and targetNode are provided
        if minDistanceArray and parent and startNode and targetNode:
            print(targetNode)
            targetx, targety = targetNode
            if minDistanceArray[targetx][targety] == float('inf'):
                print(targetx)
                print(targety)
                print(minDistanceArray[targetx][targety])
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
        plt.show()

def reconstruct_path(parent, startNode, targetNode):
    """Helper function to reconstruct the path from startNode to targetNode using the parent dictionary."""
    path = []
    currentNode = targetNode
    while currentNode is not None:
        path.append(currentNode)
        currentNode = parent.get(currentNode)[0]  # Get the parent node
    path.reverse()  # Reverse the path to get it from start to target
    return path

def build_graph_from_memory(map_memory:list[list[Square]]) -> MazeGraph:
    graph = MazeGraph()

    # Iterate through the map_memory and add edges to the graph
    for i in range(len(map_memory)):
        for j in range(len(map_memory[0])):
            current_square = map_memory[i][j]

            # Coordinates for neighboring cells
            neighbors = {
                'left': (i, j - 1),
                'right': (i, j + 1),
                'up': (i - 1, j),
                'down': (i + 1, j)
            }
            
            # Add edges based on door frequencies and valid neighboring cells
            
            # Left neighbor exists 
            if j > 0:
                leftSquare = map_memory[i][j - 1]
                graph.add_bidirectional_edge((i, j), neighbors['left'], current_square.leftFreq, leftSquare.rightFreq)

            # Right neighbor exists 
            if j < len(map_memory[0]) - 1:  
                rightSquare = map_memory[i][j + 1]
                graph.add_bidirectional_edge((i, j), neighbors['right'], current_square.rightFreq, rightSquare.leftFreq)
            
            # Up neighbor exists 
            if i > 0: 
                upSquare = map_memory[i - 1][j]
                graph.add_bidirectional_edge((i, j), neighbors['up'], current_square.upFreq, upSquare.downFreq)
            
            # Down neighbor exists
            if i < len(map_memory) - 1:
                downSquare = map_memory[i + 1][j]
                graph.add_bidirectional_edge((i, j), neighbors['down'], current_square.downFreq, downSquare.upFreq)

    return graph



