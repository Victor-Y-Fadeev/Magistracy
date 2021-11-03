import numpy as np
import pandas as pd
import networkx as nx


def load(path) -> nx.graph:
    """Load edges list by path.
        
    Keyword arguments:
    path -- path to file

    """
    edges = pd.read_csv(path, sep='\s+', names=('source', 'target'))
    edges.insert(len(edges.columns), 'weight', np.ones(len(edges.index)))
    return nx.from_pandas_edgelist(edges, edge_attr='weight')

def modify(graph: nx.graph):
    """Modify edges before learn.
        
    Keyword arguments:
    graph -- friendship network

    """
    for i in nx.nodes(graph):
        graph.add_edge(i, i, weight=-1.0)


def update_responsibility(graph: nx.graph):
    """Update responsibility.
        
    Keyword arguments:
    graph -- friendship network

    """
    for i in nx.nodes(graph):
        prev_max, new_max, index = -1.0, -1.0, 0
        for j in nx.neighbors(graph, i):
            try:
                temp = graph[i][j]['weight'] + graph[i][j]['availability']
            except KeyError:
                temp = graph[i][j]['weight']
            if new_max <= temp:
                prev_max, new_max, index = new_max, temp, j

        for j in nx.neighbors(graph, i):
            if j != index:
                graph.add_edge(i, j, responsibility=graph[i][j]['weight'] - new_max)
            else:
                graph.add_edge(i, j, responsibility=graph[i][j]['weight'] - prev_max)

def update_availability(graph: nx.graph):
    """Update availability.
        
    Keyword arguments:
    graph -- friendship network

    """
    for k in nx.nodes(graph):
        new_sum = min(0, graph[k][k]['responsibility'])
        for j in nx.neighbors(graph, k):
            new_sum += max(0, graph[j][k]['responsibility'])

        for i in nx.neighbors(graph, k):
            if i != k:
                temp = min(0, new_sum - max(0, graph[i][k]['responsibility']))
            else:
                temp = new_sum - graph[k][k]['responsibility']
            graph.add_edge(i, k, availability=temp)


def get_clusters(graph: nx.graph):
    clusters = []
    for i in nx.nodes(graph):
        index = next(nx.neighbors(graph, i))
        new_max = graph[i][index]['availability'] + graph[i][index]['responsibility']

        for k in nx.neighbors(graph, i):
            temp = graph[i][k]['availability'] + graph[i][k]['responsibility']
            if temp > new_max:
                new_max, index = temp, k

        clusters.append(index)
    return clusters




graph = load('./Dataset/Gowalla_edges.txt')
modify(graph)

MAX_ITERATIONS = 10
i = 0

while i < MAX_ITERATIONS:
    print('ITERATION:', i)
    update_responsibility(graph)
    update_availability(graph)
    i += 1

print(len(set(get_clusters(graph))))

#print(graph[0][2]['responsibility'])
#graph.add_edge(0, 2, responsibility=5)
#print(graph[0][2]['responsibility'])
#print(graph[2][2]['responsibility'])

#print(list(nx.neighbors(graph, 0)))

print(graph[0][0])
print(graph[0][1])
print(graph[1][0])

try:
    print(graph[0][0])
except KeyError:
    print('error')

print(graph[0][1]['responsibility'])