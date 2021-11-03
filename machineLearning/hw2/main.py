import functools
import numpy as np
import pandas as pd
import networkx as nx


def modify_decorator(func):
    """Modify decorator.
    
    Keyword arguments:
    func -- load function to decorate

    """
    @functools.wraps(func)
    def wrapper(path) -> nx.graph:
        graph = func(path)
        for i in nx.nodes(graph):
            graph.add_edge(i, i, weight=-1.0)
        return graph
    return wrapper

@modify_decorator
def load_dataset(path) -> nx.graph:
    """Load edges list by path.
        
    Keyword arguments:
    path -- path to file

    """
    edges = pd.read_csv(path, sep='\s+', names=('source', 'target'))
    edges.insert(len(edges.columns), 'weight', np.ones(len(edges.index)))
    return nx.from_pandas_edgelist(edges, edge_attr='weight')


def save_graph(graph: nx.graph, path):
    """Save graph to edges list.
        
    Keyword arguments:
    graph -- friendship network
    path -- path to file

    """
    nx.to_pandas_edgelist(graph).to_csv(path, index_label=False)

def load_graph(path) -> nx.graph:
    """Load graph from edges list.
        
    Keyword arguments:
    path -- path to file

    """
    return nx.from_pandas_edgelist(pd.read_csv(path), edge_attr=True)


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


def learn(graph: nx.graph, iterations):
    """Learn by Affinity propagation.
        
    Keyword arguments:
    graph -- friendship network
    iterations -- max iterations

    """
    i = 0
    while i < iterations:
        print('ITERATION:', i)
        update_responsibility(graph)
        update_availability(graph)
        i += 1

def get_clusters(graph: nx.graph):
    """Extract clusters from graph.
        
    Keyword arguments:
    graph -- friendship network

    """
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


graph = load_dataset('./Dataset/Gowalla_edges.txt')
#graph = load_graph('./Result/graph.csv')

MAX_ITERATIONS = 1
learn(graph, MAX_ITERATIONS)
save_graph(graph, './Result/graph.csv')

clusters = get_clusters(graph)
print('CLUSTERS:', len(set(clusters)))
