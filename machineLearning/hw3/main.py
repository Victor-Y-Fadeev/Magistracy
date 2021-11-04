import functools
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter


MAX_ITERATIONS = 20
DIAGONAL_VALUE = -1.0

HIDDEN_USERS = 100
TOP_LOCATIONS = 10


def modify_decorator(func):
    """Modify decorator.
    
    Keyword arguments:
    func -- load function to decorate

    """
    @functools.wraps(func)
    def wrapper(path) -> nx.graph:
        graph = func(path)
        for i in nx.nodes(graph):
            graph.add_edge(i, i, weight=DIAGONAL_VALUE)
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

def load_checkins(path) -> pd.DataFrame:
    """Load users check-ins by path.
        
    Keyword arguments:
    path -- path to file

    """
    checkins = pd.read_csv(path,
                           sep='\s+',
                           usecols=('user', 'location id'),
                           names=('user',
                                  'check-in time',
                                  'latitude',
                                  'longitude',
                                  'location id'))

    checkins = checkins.groupby('user')['location id']
    checkins = checkins.apply(list).reset_index(name='locations')
    return checkins.set_index('user')


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
        index = i
        new_max = graph[i][i]['availability'] + graph[i][i]['responsibility']

        for k in nx.neighbors(graph, i):
            temp = graph[i][k]['availability'] + graph[i][k]['responsibility']
            if temp > new_max:
                new_max, index = temp, k

        clusters.append(index)
    return clusters

def get_predictions(checkins: pd.DataFrame, clusters) -> pd.DataFrame:
    """Get location predictions for clusters.
        
    Keyword arguments:
    checkins -- check-ins made by users
    clusters -- clusters table

    """
    predictions = checkins.reindex(range(len(clusters)), fill_value=[])
    predictions.insert(0, 'cluster', clusters)

    predictions = predictions.groupby('cluster')['locations']
    predictions = predictions.agg(sum).reset_index('cluster')

    top = lambda x: [key for key, _ in Counter(x).most_common(TOP_LOCATIONS)]
    return predictions['locations'].apply(top)


#graph = load_dataset('./Dataset/Gowalla_edges.txt')
graph = load_graph('./Result/graph.csv')

#learn(graph, MAX_ITERATIONS)
#save_graph(graph, './Result/graph.csv')

clusters = get_clusters(graph)
print('CLUSTERS:', len(set(clusters)))


checkins = load_checkins('./Dataset/Gowalla_totalCheckins.txt')
hidden = checkins.index[np.linspace(0, len(checkins.index), num=HIDDEN_USERS,
                                    endpoint=False, dtype=int)]
predictions = get_predictions(checkins, clusters)


print(predictions[:10])
