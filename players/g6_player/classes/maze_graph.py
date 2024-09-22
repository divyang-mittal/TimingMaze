import networkx as nx
import math

class MazeGraph:
    def __init__(self):
        # Initialize a graph using networkx
        self.graph = nx.Graph()

    def add_edge(self, node1, node2, weight):
        """
        Add an edge between two nodes (cells) with a given path distance (weight).
        """
        self.graph.add_edge(node1, node2, weight=weight)

    def euclidean_distance(self, node1, node2):
        """
        Heuristic function: calculate the Euclidean distance between two nodes.
        """
        x1, y1 = node1
        x2, y2 = node2
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def astar_shortest_path(self, start, target):
        """Find the shortest path between start and end nodes using A* algorithm."""
        print(type(start), type(target))
        try:
            shortest_path = nx.astar_path(
                self.graph, source=start, target=target, heuristic=self.euclidean_distance, weight='weight'
            )
            path_length = nx.astar_path_length(
                self.graph, source=start, target=target, heuristic=self.euclidean_distance, weight='weight'
            )
            return shortest_path, path_length
        except Exception as e:
            print(f"An error occurred while finding the path: {str(e)}")
            return None, float('inf')

    def get_distinct_nodes(self):
        """Return a list of distinct nodes in the graph."""
        print("Number of nodes: {}".format(len(list(self.graph.nodes()))))
        return # list(self.graph.nodes())

    def display_graph(self):
        """Display the graph's edges and weights."""
        for edge in self.graph.edges(data=True):  # Accessing the 'edges' method of the graph
            print(f"Edge from cell {edge[0]} to cell {edge[1]} with distance {edge[2]['weight']}")
