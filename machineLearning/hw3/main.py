import functools
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter


MAX_NODES = 100
DIAGONAL_VALUE = -1
NOISE_POWER = -16

GAMMA = 0.95
MAX_ITERATIONS = 1000

HIDDEN_USERS = 10
TOP_LOCATIONS = 10


def noise_decorator(func):
    """Noise decorator.

    Keyword arguments:
    func -- function to decorate

    """
    @functools.wraps(func)
    def wrapper(path) -> nx.graph:
        graph = func(path)
        for i in graph.edges:
            noise = np.random.uniform(high=10.0) * pow(10, NOISE_POWER)
            graph[i[0]][i[1]]['weight'] += noise

        return graph
    return wrapper

def fill_decorator(func):
    """Fill decorator.

    Keyword arguments:
    func -- function to decorate

    """
    @functools.wraps(func)
    def wrapper(path) -> nx.graph:
        graph = func(path)
        for i in nx.nodes(graph):
            for j in set(nx.nodes(graph)).difference(set(nx.neighbors(graph, i))):
                graph.add_edge(i, j, weight=0)

        return graph
    return wrapper

def diagonal_decorator(func):
    """Diagonal decorator.

    Keyword arguments:
    func -- function to decorate

    """
    @functools.wraps(func)
    def wrapper(path) -> nx.graph:
        graph = func(path)
        for i in nx.nodes(graph):
            graph.add_edge(i, i, weight=DIAGONAL_VALUE)

        return graph
    return wrapper

def cut_decorator(func):
    """Cut decorator.

    Keyword arguments:
    func -- function to decorate

    """
    @functools.wraps(func)
    def wrapper(path):
        data = func(path)
        if type(data) is pd.Series:
            data.drop([id for id in data.index if id >= MAX_NODES], inplace=True)
        else:
            data.remove_nodes_from([node for node in data.nodes if node >= MAX_NODES])

        return data
    return wrapper


@noise_decorator
@fill_decorator
@diagonal_decorator
@cut_decorator
def load_dataset(path) -> nx.graph:
    """Load edges list by path.

    Keyword arguments:
    path -- path to file

    """
    edges = pd.read_csv(path, sep='\s+', names=('source', 'target'))
    edges.insert(len(edges.columns), 'weight', np.ones(len(edges.index), dtype=int))
    return nx.from_pandas_edgelist(edges, edge_attr='weight')

@cut_decorator
def load_checkins(path) -> pd.Series:
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
    return checkins.set_index('user')['locations']


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
    was_updated = False
    for i in nx.nodes(graph):
        for k in nx.neighbors(graph, i):
            maximum = graph[i][nx.neighbors(graph, i)[:1]]['weight']
            for j in nx.neighbors(graph, i):
                try:
                    temp = graph[i][j]['weight'] + graph[i][j]['availability']
                except KeyError:
                    temp = graph[i][j]['weight']
                if maximum <= temp:
                    maximum = temp

        temp = graph[i][j]['weight'] - maximum

        try:
            was_updated |= graph[i][j]['responsibility'] != temp
            temp += GAMMA * graph[i][j]['responsibility']
        except KeyError:
            was_updated = True

        graph.add_edge(i, j, responsibility=temp)
    return was_updated

def update_availability(graph: nx.graph):
    """Update availability.

    Keyword arguments:
    graph -- friendship network

    """
    was_updated = False
    for k in nx.nodes(graph):
        new_sum = min(0, graph[k][k]['responsibility'])
        for j in nx.neighbors(graph, k):
            new_sum += max(0, graph[j][k]['responsibility'])

        for i in nx.neighbors(graph, k):
            if i != k:
                temp = min(0, new_sum - max(0, graph[i][k]['responsibility']))
            else:
                temp = new_sum - graph[k][k]['responsibility']

            try:
                was_updated |= graph[i][k]['availability'] != temp
                temp += GAMMA * graph[i][k]['availability']
            except KeyError:
                was_updated = True

            graph.add_edge(i, k, availability=temp)
    return was_updated

def learn(graph: nx.graph, iterations):
    """Learn by Affinity propagation.

    Keyword arguments:
    graph -- friendship network
    iterations -- max iterations

    """
    i = 0
    was_updated = True
    while i < iterations and was_updated:
        print('ITERATION:', i)
        was_updated = update_responsibility(graph)
        was_updated |= update_availability(graph)
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

def get_predictions(checkins: pd.Series, clusters) -> pd.Series:
    """Get location predictions for clusters.

    Keyword arguments:
    checkins -- check-ins made by users
    clusters -- clusters table

    """
    predictions = checkins.to_frame().reindex(range(len(clusters)), fill_value=[])
    predictions.insert(0, 'cluster', clusters)

    predictions = predictions.groupby('cluster')['locations']
    predictions = predictions.agg(sum).reset_index('cluster')

    top = lambda x: [key for key, _ in Counter(x).most_common(TOP_LOCATIONS)]
    return predictions['locations'].apply(top)


graph = load_dataset('./Dataset/Gowalla_edges.txt')
#graph = load_graph('./Result/graph.csv')

learn(graph, MAX_ITERATIONS)
save_graph(graph, './Result/graph.csv')

clusters = get_clusters(graph)
print('CLUSTERS:', len(set(clusters)))


checkins = load_checkins('./Dataset/Gowalla_totalCheckins.txt')
hidden = checkins.index[np.linspace(0, len(checkins.index), num=HIDDEN_USERS,
                                    endpoint=False, dtype=int)]

removed = checkins.drop(hidden)
predictions = get_predictions(removed, clusters)


precision = 0
for id in hidden:
    if clusters[id] in predictions:
        precision += len(set(predictions[clusters[id]]).intersection(set(checkins[id])))

precision /= TOP_LOCATIONS * HIDDEN_USERS
print('PRECISION:', precision)
