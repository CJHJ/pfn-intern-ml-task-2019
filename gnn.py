import numpy as np
import tqdm
from copy import deepcopy
import random
import pickle
import os
import argparse

import utils

# Helper functions
def relu(x):
    return x * (x > 0)

def sigmoid(s):
    if s < 0:
        a = np.exp(s)
        p = a / (1.0 + a)
    else:
        p = 1.0 / (1.0 + np.exp(-s))
    return p

def trunc(x):
    return 1 if x > 0.5 else 0

def bce(s, y):
    return y * np.logaddexp(0, -s) + (1.0 - y) * np.logaddexp(0, s)

# Node class
class Node:
    # Initialize node
    def __init__(self, init_feature=None):
        self.feature = init_feature
        self.adjacent = {}
    
    # Initialize node using a specified dimension
    @classmethod
    def fromdim(cls, dim):
        feature = np.zeros(dim)
        feature[0] = 1
        return cls(feature)

    # Add adjacent node
    def add_node(self, index, adj_node):
        self.adjacent[index] = adj_node

    # Remove specified node
    def remove_node(self, index):
        del self.adjacent[index]

    # Print node and feature
    def print_adjacent(self):
        for index, node in self.adjacent.items():
            print("Node {}: {}".format(index, node.feature))

# Undirected graph class
class Graph:
    # Initialize graph
    def __init__(self, dim):
        self.dim = dim
        self.nodes = {}

    # Add node
    def add_node(self, index, init_feature=None):
        if init_feature is None:
            self.nodes[index] = Node.fromdim(self.dim)
        else:
            if init_feature.size != self.dim:
                raise Exception("Inputted feature has different dimension. Expected {}, instead got {}".format(self.dim, init_feature.size))
            self.nodes[index] = Node(init_feature)
    
    # Remove node
    def remove_node(self, index):
        # Delete every reference of the node from other nodes
        for _, main_node in self.nodes[index].adjacent.items():
            main_node.remove_node(index)

        del self.nodes[index]

    # Add edge
    def add_edge(self, index_1, index_2):
        self.nodes[index_1].add_node(index_2, self.nodes[index_2])
        self.nodes[index_2].add_node(index_1, self.nodes[index_1])

    # Remove edge
    def remove_edge(self, index_1, index_2):
        self.nodes[index_1].remove_node(index_2)
        self.nodes[index_2].remove_node(index_1)

    # Print graph structure
    def print_graph(self):
        # Print graph
        for index, node in self.nodes.items():
            print("Node {}".format(index))
            print(node.feature)

            print("Adjacent nodes:")
            node.print_adjacent()

# GNN class
class GNN:
    def __init__(self, dim, agg_T=2, init_W=None, init_A=None, init_b=0):
        self.dim = dim
        
        # Hyperparameters
        self.agg_T = agg_T
        self.epsilon = 0.001
        self.alpha = 0.0001
        self.momentum = 0.9

        # Learnable parameters
        if init_W is None:
            self.W = np.identity(self.dim)
        else:
            if init_W.shape != (self.dim, self.dim):
                raise Exception("Wrong weight W size.")
            self.W = init_W

        if init_A is None:
            self.A = np.ones(self.dim)
        else:
            if init_A.shape != (self.dim,):
                raise Exception("Wrong weight A size.")
            self.A = init_A

        self.b = init_b

        # Momentum SGD
        self.w_W = np.zeros((self.dim, self.dim))
        self.w_A = np.zeros(self.dim)
        self.w_b = 0
        
        # Adam
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.m_W = np.zeros((self.dim, self.dim))
        self.m_A = np.zeros(self.dim)
        self.m_b = 0
        self.v_W = np.zeros((self.dim, self.dim))
        self.v_A = np.zeros(self.dim)
        self.v_b = 0

    # Forward pass
    def forward(self, graph, W, A, b):
        # Problem 1
        temp_graph = deepcopy(graph)
        for i in range(self.agg_T):
            temp_graph = self.aggregator_1(temp_graph)
            temp_graph = self.aggregator_2(temp_graph, W)
        h_g = self.readout(temp_graph)

        # Problem 2
        s = self.wsum(h_g, A, b)

        return s

    # Predict the result
    def prediction(self, graph, true_label=None):
        s = self.forward(graph, self.W, self.A, self.b)
        p = sigmoid(s)
        y = trunc(p)

        if true_label is not None:
            loss = bce(s, true_label)
        else:
            loss = None

        return y, loss
    
    # Calculate loss
    def loss(self, graph, true_label, W, A, b):
        s = self.forward(graph, W, A, b)
        loss = bce(s, true_label)

        return loss

    # Train for problem 2
    def train(self, graph, true_label):
        # Calculate the current loss
        cur_loss = self.loss(graph, true_label, self.W, self.A, self.b)

        # For each learnable parameters, calculate the loss gradients
        # W
        update_W = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                temp_W = deepcopy(self.W)
                temp_W[i, j] += self.epsilon

                temp_loss = self.loss(graph, true_label, temp_W, self.A, self.b)
                gradient = (temp_loss - cur_loss) / self.epsilon

                update_W[i, j] = self.W[i, j] - self.alpha * gradient 

        # A
        update_A = np.zeros(self.dim)
        for i in range(self.dim):
            temp_A = deepcopy(self.A)
            temp_A[i] += self.epsilon

            temp_loss = self.loss(graph, true_label, self.W, temp_A, self.b)
            gradient = (temp_loss - cur_loss) / self.epsilon

            update_A[i] = self.A[i] - self.alpha * gradient 

        # b
        temp_b = self.b + self.epsilon
        temp_loss = self.loss(graph, true_label, self.W, self.A, temp_b)
        gradient = (temp_loss - cur_loss) / self.epsilon

        # Update learnable parameters
        self.b = self.b - self.alpha * gradient
        self.W = update_W
        self.A = update_A

        return cur_loss
    
    # Calculate mini-batch gradients
    def calc_gradients(self, graphs, true_labels):
        total = len(graphs)
        run_loss = 0
        run_up_W = np.zeros((self.dim, self.dim))
        run_up_A = np.zeros(self.dim)
        run_up_b = 0
        for graph, true_label in zip(graphs, true_labels):
            # Calculate the current loss
            cur_loss = self.loss(graph, true_label, self.W, self.A, self.b)

            run_loss += cur_loss

            # For each learnable parameters, calculate the loss gradients
            # W
            for i in range(self.dim):
                for j in range(self.dim):
                    temp_W = deepcopy(self.W)
                    temp_W[i, j] += self.epsilon

                    temp_loss = self.loss(graph, true_label, temp_W, self.A, self.b)
                    gradient = (temp_loss - cur_loss) / self.epsilon

                    run_up_W[i, j] += gradient

            # A
            for i in range(self.dim):
                temp_A = deepcopy(self.A)
                temp_A[i] += self.epsilon

                temp_loss = self.loss(graph, true_label, self.W, temp_A, self.b)
                gradient = (temp_loss - cur_loss) / self.epsilon

                run_up_A[i] += gradient
                
            # b
            temp_b = self.b + self.epsilon

            temp_loss = self.loss(graph, true_label, self.W, self.A, temp_b)
            gradient = (temp_loss - cur_loss) / self.epsilon

            run_up_b += gradient

        delta_W = run_up_W / total
        delta_A = run_up_A / total
        delta_b = run_up_b / total
        
        return delta_W, delta_A, delta_b, run_loss

    # Train with SGD
    def train_sgd(self, graphs, true_labels):
        delta_W, delta_A, delta_b, run_loss = self.calc_gradients(graphs, true_labels)
        
        # Update learnable parameters
        self.W -= self.alpha * delta_W
        self.A -= self.alpha * delta_A
        self.b -= self.alpha * delta_b

        return run_loss

    # Momentum SGD functions
    def init_momentum(self):
        self.w_W = np.zeros((self.dim, self.dim))
        self.w_A = np.zeros(self.dim)
        self.w_b = 0

    # Train with Momentum SGD
    def train_momentum_sgd(self, graphs, true_labels):
        delta_W, delta_A, delta_b, run_loss = self.calc_gradients(graphs, true_labels)

        # Update learnable parameters
        temp_change_W = - (self.alpha * delta_W) + (self.momentum * self.w_W)
        temp_change_A = - (self.alpha * delta_A) + (self.momentum * self.w_A)
        temp_change_b = - (self.alpha * delta_b) + (self.momentum * self.w_b)

        self.W += temp_change_W
        self.A += temp_change_A
        self.b += temp_change_b

        # Update momentum change
        self.w_W = temp_change_W
        self.w_A = temp_change_A
        self.w_b = temp_change_b

        return run_loss
    
    # Train with Adam
    def train_adam(self, graphs, true_labels, t):
        delta_W, delta_A, delta_b, run_loss = self.calc_gradients(graphs, true_labels)

        # Update moments
        self.m_W = self.beta_1 * self.m_W + (1.0 - self.beta_1) * delta_W
        self.m_A = self.beta_1 * self.m_A + (1.0 - self.beta_1) * delta_A
        self.m_b = self.beta_1 * self.m_b + (1.0 - self.beta_1) * delta_b

        self.v_W = self.beta_2 * self.v_W + (1.0 - self.beta_2) * np.power(delta_W, 2)
        self.v_A = self.beta_2 * self.v_A + (1.0 - self.beta_2) * np.power(delta_A, 2)
        self.v_b = self.beta_2 * self.v_b + (1.0 - self.beta_2) * np.power(delta_b, 2)

        # Calculate corrections
        m_hat_W = self.m_W / (1.0 - np.power(self.beta_1, t))
        m_hat_A = self.m_A / (1.0 - np.power(self.beta_1, t))
        m_hat_b = self.m_b / (1.0 - np.power(self.beta_1, t))

        v_hat_W = self.v_W / (1.0 - np.power(self.beta_2, t))
        v_hat_A = self.v_A / (1.0 - np.power(self.beta_2, t))
        v_hat_b = self.v_b / (1.0 - np.power(self.beta_2, t))

        # Update learnable parameters
        temp_change_W = - self.alpha * m_hat_W / (np.sqrt(v_hat_W) + 1e-8)
        temp_change_A = - self.alpha * m_hat_A / (np.sqrt(v_hat_A) + 1e-8)
        temp_change_b = - self.alpha * m_hat_b / (np.sqrt(v_hat_b) + 1e-8)

        self.W += temp_change_W
        self.A += temp_change_A
        self.b += temp_change_b

        return run_loss
    
    # Save model
    def save_model(self, directory):
        save_list(directory, (self.W, self.A, self.b))
        
    # Load model
    def load_model(self, directory):
        W, A, b = pickle.load(open(directory, "rb"))
        
        assert W.shape == (self.dim, self.dim)
        assert A.shape == (self.dim, )
        
        self.W = W
        self.A = A
        self.b = b
        
    # MODEL FUNCTIONS
    # Aggregator-1
    def aggregator_1(self, graph):
        temp_graph = deepcopy(graph)

        # Aggregate adjacent nodes' feature to the main node
        for index, main_node in graph.nodes.items():
            main_node.feature = np.zeros(graph.dim)
            for _, adj_node in temp_graph.nodes[index].adjacent.items():
                main_node.feature += adj_node.feature

        return graph

    # Aggregator-2 (inplace)
    def aggregator_2(self, graph, W):
        # Multiply with weight
        for _, main_node in graph.nodes.items():
            main_node.feature = relu(W @ main_node.feature)
        
        return graph

    # Readout
    def readout(self, graph):
        h_g = np.zeros(graph.dim)

        for _, main_node in graph.nodes.items():
            h_g += main_node.feature

        return h_g
    
    # Weighted sum layer
    def wsum(self, h_g, A, b):
        return np.dot(A, h_g) + b

# Save list
def save_list(name, x):
    with open(name, 'wb') as fp:
        pickle.dump(x, fp)


# Run problem solution
def problem_2():
    # Initialize graph
    dim = 32
    graph = Graph(dim)
    
    # Add nodes
    for i in range(0,10):
        graph.add_node(i)
    
    # Add edges
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(0, 3)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    graph.add_edge(4, 5)
    graph.add_edge(5, 6)
    graph.add_edge(6, 7)
    graph.add_edge(6, 8)
    graph.add_edge(8, 9)
    graph.add_edge(9, 0)

    # Test model
    init_W = np.random.normal(0.0, 0.4, (dim, dim))
    init_A = np.random.normal(0.0, 0.4, (dim))
    gnn = GNN(dim, init_W=init_W, init_A=init_A)
    true_y = 0

    for epoch in range(0, 100):
        print("Epoch {}".format(epoch))
        print("Current loss: {}".format(gnn.train(graph, true_y)))

# Train and test directory
TRAIN_PATH = '../datasets/train/'
TEST_PATH = '../datasets/test/'

def problem_3_and_4(train_mb_size, epochs, alg_type, see_mb_loss, save_epoch, load_dir, predict_only):
    # Print all arguments
    print("Batch size: {}".format(train_mb_size))
    print("Epoch: {}".format(epochs))
    print("Algorithm type: {}".format(alg_type))
    print("See mini-batch loss: {}".format(see_mb_loss))
    print("Save epoch: {}".format(save_epoch))
    print("Load model: {}".format(load_dir))
    print("Predict only: {}".format(predict_only))

    # Build model
    dim = 8
    B = train_mb_size
    B_valid = 1
    init_W = np.random.normal(0.0, 0.4, (dim, dim))
    init_A = np.random.normal(0.0, 0.4, (dim))
    gnn = GNN(dim, init_W=init_W, init_A=init_A)

    # Get and partition indices into training and validation indices
    train_indices = utils.get_indices(TRAIN_PATH)
    random.shuffle(train_indices)
    split_index = int(len(train_indices) * 0.8)

    # Save data or load data
    if os.path.isfile('./train_ids.npy') :
        print("Load data")
        train_data = np.load('train_ids.npy')
        valid_data = np.load('valid_ids.npy')
    else:
        print("Save data")
        train_data = np.array(train_indices[:split_index])
        valid_data = np.array(train_indices[split_index:])

        np.save('train_ids.npy', train_data)
        np.save('valid_ids.npy', valid_data)
    
    # Records
    train_accs = []
    train_losses = []
    valid_accs = []
    valid_losses = []
    
    # Load model
    if load_dir is not None:
        gnn.load_model(load_dir)

    # Adam time
    if alg_type == 'adam':
        t = 0
        
    # Train model
    if not predict_only:
        for epoch in range(1, epochs+1):
            print("Epoch {}".format(epoch))

            # Make generators
            train_gen = utils.graph_generator(TRAIN_PATH, train_data, dim, mb_size=B, shuffle=True)
            valid_gen = utils.graph_generator(TRAIN_PATH, valid_data, dim, mb_size=B_valid)

            train_mb_size = np.ceil(len(train_data) / B)
            valid_mb_size = np.ceil(len(valid_data) / B_valid)

            # --------------------
            # Training
            # --------------------
            total_data = 0

            # Initialize running mb loss
            running_loss = []

            # Training loop
            for graphs, labels in tqdm.tqdm(train_gen, total=train_mb_size):
                len_data = len(labels)
                total_data += len_data
                
                if alg_type == 'sgd':
                    running_loss.append(gnn.train_sgd(graphs, labels))
                elif alg_type == 'msgd':
                    running_loss.append(gnn.train_momentum_sgd(graphs, labels))
                elif alg_type == 'adam':
                    t += 1
                    running_loss.append(gnn.train_adam(graphs, labels, t))
                    
                if see_mb_loss:
                    avg_mb_loss = running_loss[-1] / len_data
                    print("Average mb training loss: {}".format(avg_mb_loss))

            # Calculate running mb loss
            if see_mb_loss:
                avg_loss = np.sum(running_loss) / total_data
                print("Average whole training loss: {}".format(avg_loss))

            # --------------------
            # Training Predictions
            # --------------------
            # Reinit train generator
            train_gen = utils.graph_generator(TRAIN_PATH, train_data, dim, mb_size=1)
            train_mb_size = len(train_data)

            # Initialize running loss and number of correct predictions
            correct = 0
            pred_run_loss = 0
            for graphs, labels in tqdm.tqdm(train_gen, total=train_mb_size):
                assert len(graphs) == len(labels)
                assert len(graphs) == 1

                pred_label, pred_loss = gnn.prediction(graphs[0], labels[0])

                if pred_label == labels[0]:
                    correct += 1
                pred_run_loss += pred_loss

            # Training Accuracy
            train_acc = correct * 100.0 / len(train_data)
            print("Correct data : {}".format(correct))
            print("Train data : {}".format(len(train_data)))
            print("Accuracy (Training): {}".format(train_acc))
            train_accs.append(train_acc)

            # Training Loss
            train_loss = pred_run_loss / len(train_data)
            print("Loss (Training): {}".format(train_loss))
            train_losses.append(train_loss)

            # --------------------
            # Validation Predictions
            # --------------------
            # Initialize running loss and number of correct predictions
            correct = 0
            pred_run_loss = 0
            for graphs, labels in tqdm.tqdm(valid_gen, total=valid_mb_size):
                assert len(graphs) == len(labels)
                assert len(graphs) == 1

                pred_label, pred_loss = gnn.prediction(graphs[0], labels[0])

                if pred_label == labels[0]:
                    correct += 1
                pred_run_loss += pred_loss

            # Validation Accuracy
            valid_acc = correct * 100.0 / len(valid_data)
            print("Accuracy (Validation): {}".format(valid_acc))
            valid_accs.append(valid_acc)

            # Validation Loss
            valid_loss = pred_run_loss / len(valid_data)
            print("Loss (Training): {}".format(valid_loss))
            valid_losses.append(valid_loss)

            # Save model
            if (save_epoch != 0) and (epoch % save_epoch == 0):
                parent_msave_dir = './models/gnn_{}/'.format(alg_type)
                if not os.path.exists(parent_msave_dir):
                    os.makedirs(parent_msave_dir)
                save_dir = parent_msave_dir + 'e{}_b{}_va{:.2f}.pkl'.format(epoch, B, valid_acc)
                print("Saving model to {}".format(save_dir))
                gnn.save_model(save_dir)
                
        # Save records
        parent_save_dir = './train_' + alg_type
        if not os.path.exists(parent_save_dir):
            os.makedirs(parent_save_dir)
        save_list(parent_save_dir + '/train_losses_b{}.pkl'.format(B), train_losses)
        save_list(parent_save_dir + '/train_accs_b{}.pkl'.format(B), train_accs)
        save_list(parent_save_dir + '/valid_losses_b{}.pkl'.format(B), valid_losses)
        save_list(parent_save_dir + '/valid_accs_b{}.pkl'.format(B), valid_accs)  
        
    # Predict model only
    else:
        # Check validation accuracy
        print("Check validation accuracy")
        
        valid_gen = utils.graph_generator(TRAIN_PATH, valid_data, dim, mb_size=B_valid)
        valid_mb_size = np.ceil(len(valid_data) / B_valid)
        
        # --------------------
        # Validation Predictions
        # --------------------
        # Initialize running loss and number of correct predictions
        correct = 0
        pred_run_loss = 0
        for graphs, labels in tqdm.tqdm(valid_gen, total=valid_mb_size):
            assert len(graphs) == len(labels)
            assert len(graphs) == 1

            pred_label, pred_loss = gnn.prediction(graphs[0], labels[0])

            if pred_label == labels[0]:
                correct += 1
            pred_run_loss += pred_loss

        # Validation Accuracy
        valid_acc = correct * 100.0 / len(valid_data)
        print("Accuracy (Validation): {}".format(valid_acc))
        valid_accs.append(valid_acc)

        # Validation Loss
        valid_loss = pred_run_loss / len(valid_data)
        print("Loss (Training): {}".format(valid_loss))
        valid_losses.append(valid_loss)
        
        # --------------------
        # Test Predictions
        # --------------------
        print("Running test predictions")
        test_data = utils.get_indices(TEST_PATH)
        test_data.sort(key=int)
        
        test_gen = utils.graph_generator(TEST_PATH, test_data, dim, mb_size=1, train=False)
        test_mb_size = np.ceil(len(test_data) / 1)
        
        out_file = open("../prediction.txt", "w")
        for graphs, _ in tqdm.tqdm(test_gen, total=test_mb_size):
            assert len(graphs) == 1

            pred_label, _ = gnn.prediction(graphs[0])
            
            out_file.write(str(pred_label))
            out_file.write("\n")
        out_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PFN Machine Learning Task")
    parser.add_argument('-b', '--batch_size', default=32)
    parser.add_argument('-e', '--epoch', default=10)
    parser.add_argument('-t', '--type', default='sgd', help="Type of algorithm to run: sgd - Vanilla SGD, msgd - Momentum SGD, adam - Adam")
    parser.add_argument('-sml', '--see_mb_loss', dest='see_mb_loss', action='store_true', help="See per-minibatch average loss.")
    parser.set_defaults(see_mb_loss=False)
    parser.add_argument('-sm', '--save_model', default=0, help="Save model every n epoch in directory called 'models'. (n=0 means not saving)")
    parser.add_argument('-lm', '--load_model', default=None, help="Load model saved in a directory.")
    parser.add_argument('-po', '--predict_only', dest='predict_only', action='store_true', help="Predict test data only. (no training)")
    parser.set_defaults(predict_only=False)
    
    args = parser.parse_args()
    
    if args.type not in ['sgd', 'msgd', 'adam']:
        raise Exception("Error: Algorithm type invalid.")

    problem_3_and_4(int(args.batch_size), int(args.epoch), args.type, args.see_mb_loss, int(args.save_model), args.load_model, args.predict_only)
