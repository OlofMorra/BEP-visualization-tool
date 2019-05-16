"""Provides implementations of graph algorithms that, aside from calculating the correct output, also store the state
of the algorithm during intermediate steps.

Algorithms
----------
dijkstra: Dijkstra's shortest path algorithm
"""

import pandas as pd
import numpy as np
from heapq import heapify, heappush, heappop
import networkx as nx
import time
import sys
from datetime import datetime

__author__ = "Freek Rooks and Olof Morra"


class Dijkstra:
    def __init__(self, G:nx.Graph, start:int, weight:str="weight"):
        self.G = G
        self.start = start
        self.weight = weight

    def get_memory_used(self, *args):
        result = 0
        for x in args:
            result += sys.getsizeof(x)
        return result

    def dijkstra(self):
        """Implementation of Dijkstra's shortest path algorithm to find shortest weighted paths from source to all other
        nodes graph G.

        Parameters
        ----------
        G : nx.Graph
            Networkx Graph object consisting of nodes and edges, which can contain extra features
        start : int
            node in G
        weight : str
            key by which the weight of edges is stored in the graph

        Returns
        -------
        (dict, dict)
            The first dictionary takes a node as key and stores the distance between source and key as value.
            The second dictionary takes a node as key and stores the previous node on the shortest path from source to key
            as value.
        """
        Q = list()  # list of (distance, vertex) tuples, maintained as a min heap by distance
        dist = dict()  # key is a node and value the distance between source and key
        prev = dict()  # key is a node and value the previous node on the shortest path from source to key
        dist[self.start] = 0

        # initialize distances and paths
        for v in self.G.nodes:
            if v != self.start:
                dist[v] = float('inf')
            prev[v] = None
            heappush(Q, (dist[v], v))  # insert v, maintaining min heap property

        while Q:
            dist_u, u = heappop(Q)  # extract minimum, maintaining min heap property
            neighs_u = [x for x in nx.neighbors(self.G, u) if x in [y[1] for y in Q]]  # neighbors of u that are still in Q

            for v in neighs_u:
                t_start = time.time()  # keep track of time

                alt = dist_u + self.G.edges[u, v][self.weight]  # dist(source, u) + dist(u, v)

                if alt < dist[v]:  # update dist if alt dist < current dist
                    dist[v] = alt
                    prev[v] = u
                    heappush(Q, (dist[v], v))

                t_elapsed = time.time() - t_start
                timestamp = datetime.now()
                memory_used = self.get_memory_used(self.G, Q, dist, prev, neighs_u)
                yield memory_used, t_elapsed, timestamp, Q, u, neighs_u, dist, prev
        # return dist, prev
