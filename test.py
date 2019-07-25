import gnn
import numpy as np
import unittest
from copy import deepcopy


# Test for Node Class
class NodeTest(unittest.TestCase):
    def setUp(self):
        self.dim = 4
        self.node1 = gnn.Node.fromdim(self.dim)
        self.node2 = gnn.Node.fromdim(self.dim)
    
    def test_add_node(self):
        self.node1.add_node(0, self.node2)

        self.assertEqual(self.node1.adjacent[0], self.node2)

    def test_remove_node(self):
        self.node1.add_node(0, self.node2)
        self.node1.remove_node(0)

        self.assertEqual(0, len(self.node1.adjacent))

# Test for Graph Class
class GraphTest(unittest.TestCase):
    def setUp(self):
        self.dim = 4
        self.graph = gnn.Graph(self.dim)

    def test_add_node(self):
        # Test dimension
        self.graph.add_node(0)

        # Test feature
        self.graph.add_node(1, np.zeros(self.dim))

        self.assertEqual(2, len(self.graph.nodes))
        self.assertSequenceEqual(np.zeros(self.dim).tolist(), self.graph.nodes[1].feature.tolist())
    
    def test_remove_node(self):
        # Make test graph
        node_num = 4
        for i in range(node_num):
            self.graph.add_node(i)
        for i in range(1, node_num):
            self.graph.add_edge(0, i)

        self.graph.remove_node(0)

        # Test node existence
        self.assertFalse(0 in self.graph.nodes)
        for i in range(1, node_num):
            self.assertFalse(0 in self.graph.nodes[i].adjacent)
    
    def test_add_edge(self):
        # Make test graph
        self.graph.add_node(0)
        self.graph.add_node(1)
        self.graph.add_edge(0, 1)

        self.assertEqual(self.graph.nodes[0], self.graph.nodes[1].adjacent[0])
        self.assertEqual(self.graph.nodes[1], self.graph.nodes[0].adjacent[1])

    def test_remove_edge(self):
        # Make test graph
        self.graph.add_node(0)
        self.graph.add_node(1)
        self.graph.add_edge(0, 1)
        self.graph.remove_edge(0, 1)

        self.assertFalse(1 in self.graph.nodes[0].adjacent)
        self.assertFalse(0 in self.graph.nodes[1].adjacent)

# Test for GNN
class GNNTest(unittest.TestCase):
    def setUp(self):
        self.dim = 3
        W = np.array([[1.0, 2.0, 1.0],
            [2.0, 1.0, 2.0],
            [-1.0, -2.0, -1.0]])

        A = np.array([-1.0, 2.0, 3.0])

        b = 2

        # Build graph
        self.graph = gnn.Graph(self.dim)
        
        # Add nodes
        self.graph.add_node(0, np.array([-1.0, 0.0, 4.0]))
        self.graph.add_node(1, np.array([9.0, -5.0, 2.0]))
        self.graph.add_node(2, np.array([3.0, -10.0, 6.0]))
        self.graph.add_node(3, np.array([-8.0, 4.0, -9.0]))

        # Add edges
        self.graph.add_edge(0, 1)
        self.graph.add_edge(1, 2)
        self.graph.add_edge(2, 0)
        self.graph.add_edge(0, 3)
        
        self.network = gnn.GNN(self.dim, agg_T=2, init_W=W, init_A=A, init_b=b)

    def test_aggregate_1(self):
        temp_graph = deepcopy(self.graph)
        temp_graph = self.network.aggregator_1(temp_graph)

        self.assertSequenceEqual([4.0, -11.0, -1.0], temp_graph.nodes[0].feature.tolist())
        self.assertSequenceEqual([2.0, -10.0, 10.0], temp_graph.nodes[1].feature.tolist())
        self.assertSequenceEqual([8.0, -5.0, 6.0], temp_graph.nodes[2].feature.tolist())
        self.assertSequenceEqual([-1.0, 0.0, 4.0], temp_graph.nodes[3].feature.tolist())

    def test_aggregate_2(self):
        temp_graph = deepcopy(self.graph)
        temp_graph = self.network.aggregator_2(temp_graph, self.network.W)

        test_node_0 = [3.0, 6.0, 0.0]
        test_node_1 = [1.0, 17.0, 0.0]
        test_node_2 = [0.0, 8.0, 11.0]
        test_node_3 = [0.0, 0.0, 9.0]

        self.assertSequenceEqual(test_node_0, temp_graph.nodes[0].feature.tolist())
        self.assertSequenceEqual(test_node_1, temp_graph.nodes[1].feature.tolist())
        self.assertSequenceEqual(test_node_2, temp_graph.nodes[2].feature.tolist())
        self.assertSequenceEqual(test_node_3, temp_graph.nodes[3].feature.tolist())

    def test_readout(self):
        h_g = self.network.readout(self.graph)

        self.assertSequenceEqual([3.0, -11.0, 3.0], h_g.tolist())

    # Test the model
    def test_model(self):
        temp_graph = deepcopy(self.graph)
        result = self.network.forward(temp_graph, self.network.W, self.network.A, self.network.b)

        self.assertEqual(254, result)

        # Testing for when the prediction is right
        true_label = 1
        pred_result, pred_loss = self.network.prediction(temp_graph, true_label=true_label)
        loss = self.network.loss(temp_graph, true_label=true_label, W=self.network.W, A=self.network.A, b=self.network.b)

        self.assertAlmostEqual(0, pred_loss)
        self.assertEqual(pred_loss, loss)
        self.assertEqual(1, pred_result)

        # Testing for when the prediction is wrong
        true_label = 0
        pred_result, pred_loss = self.network.prediction(temp_graph, true_label=true_label)
        loss = self.network.loss(temp_graph, true_label=true_label, W=self.network.W, A=self.network.A, b=self.network.b)

        self.assertEqual(254.0, pred_loss)
        self.assertEqual(pred_loss, loss)
        self.assertEqual(1, pred_result)


unittest.main()