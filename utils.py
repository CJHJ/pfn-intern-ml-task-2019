import gnn
import numpy as np
import os
from copy import deepcopy
import random

from operator import methodcaller

TRAIN_PATH = '../datasets/train/'
TEST_PATH = '../datasets/test/'

# Get data index from filename
def get_indices(path):
    filelist = list(map(methodcaller('split', '_'), os.listdir(path)))
    return list(set([a[0] for a in filelist]))

# Convert text into graph
def convert_to_graph(textfile, dim, init_feature=None):
    # Open and get node number
    f = open(textfile)
    node_number = int(f.readline().strip())

    # Init graph and add nodes
    graph = gnn.Graph(dim)
    for i in range(node_number):
        graph.add_node(i, init_feature=init_feature)
    
    # Add edges
    for i, string in enumerate(f):
        edge_list = string.strip().split(' ')
        
        # Only look on lower triangular matrix
        for j in range(i + 1):
            if int(edge_list[j]) == 1:
                graph.add_edge(i, j)

    f.close()

    return graph

# Graph generator
def graph_generator(parent_dir, indices, dim, mb_size=10, train=True, init_feature=None, shuffle=False):
    # Initialize graph and label mini-batch list
    graphs = []
    labels = []
    
    if shuffle:
        random.shuffle(indices)

    # For every index in indices yield a mini batch which size is dictated by mb_size
    for index in indices:        
        # Get a graph
        path_to_file = parent_dir + index
        graph = convert_to_graph(path_to_file + '_graph.txt', dim, init_feature=init_feature)
        if train:
            f = open(path_to_file + '_label.txt')
            label = int(f.readline().strip())
            f.close()
        else:
            label = None

        # print("Index: {}, Label: {}".format(index, label))
        
        graphs.append(deepcopy(graph))
        labels.append(label)

        # Yield element when the length is equal to mb_size
        if len(graphs) == mb_size:
            yield graphs, labels
            
            # Reinit lists
            graphs = []
            labels = []

    # If there are leftover graphs
    if graphs:
        yield graphs, labels

if __name__ == "__main__":
    train_indices = get_indices(TRAIN_PATH)
    test_indices = get_indices(TEST_PATH)
    train_gen = graph_generator(TRAIN_PATH, train_indices, 4)
    test_gen = graph_generator(TEST_PATH, test_indices, 4, train=False)

    for graphs, labels in train_gen:
        print(labels)
        print("---")
