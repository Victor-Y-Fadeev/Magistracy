import functools
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
        max_for, max, index = -1.0, -1.0, 0
        for j in nx.neighbors(graph, i):
            try:
                temp = graph[i][j]['weight'] + graph[i][j]['availability']
            except KeyError:
                temp = graph[i][j]['weight']
            if max <= temp:
                max_for, max, index = max, temp, j

        for j in nx.neighbors(graph, i):
            if j != index:
                graph.add_edge(i, j, responsibility=graph[i][j]['weight'] - max)
            else:
                graph.add_edge(i, j, responsibility=graph[i][j]['weight'] - max_for)




graph = load('./Dataset/Gowalla_edges.txt')
modify(graph)

update_responsibility(graph)


#print(graph[0][2]['responsibility'])
#graph.add_edge(0, 2, responsibility=5)
#print(graph[0][2]['responsibility'])
#print(graph[2][2]['responsibility'])

print(graph[0][0])
print(graph[0][1])
print(graph[1][0])

try:
    print(graph[0][0])
except KeyError:
    print('error')

print(graph[0][1]['responsibility'])