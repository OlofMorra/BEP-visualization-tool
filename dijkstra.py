import pandas as pd
import numpy as np
import networkx as nx
from heapq import heapify, heappush, heappop


def dijkstra(G:nx.Graph, source:int, length:str="length"):
    Q = list()  # list of (distance, vertex) tuples, maintained as a min heap by distance
    dist = dict()  # key is a vertex and value the distance between source and key
    prev = dict()  # key is a vertex and value the previous vertex on the path from source to key

    dist[source] = 0

    # initialize distances and paths
    for v in G.nodes:
        if v != source:
            dist[v] = float('inf')
        prev[v] = None
        heappush(Q, (dist[v], v))  # insert v, maintaining min heap property

    while Q:
        d, u = heappop(Q)  # extract minimum, maintaining min heap property
        neighs_u = [x for x in nx.neighbors(G, u) if x in [y[1] for y in Q]]  # neighbors of u that are still in Q

        for v in neighs_u:
            alt = dist[u] + G.edges[u, v][length]  # dist(source, u) + dist(u, v)

            if alt < dist[v]:  # update dist if alt dist < current dist
                dist[v] = alt
                prev[v] = u
                heappush(Q, (dist[v], v))
    return dist, prev
