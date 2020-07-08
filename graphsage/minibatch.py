from __future__ import division
from __future__ import print_function

import numpy as np
import networkx as nx
import os
import time

class EdgeMinibatchIterator(object):
    
    """ This minibatch iterator iterates over batches of sampled edges or
    random pairs of co-occuring edges.

    G -- networkx graph
    id2idx -- dict mapping node ids to index in feature tensor
    placeholders -- tensorflow placeholders object
    context_pairs -- if not none, then a list of co-occuring node pairs (from random walks)
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    n2v_retrain -- signals that the iterator is being used to add new embeddings to a n2v model
    fixed_n2v -- signals that the iterator is being used to retrain n2v with only existing nodes as context
    """
    def __init__(self, G, id2idx, 
            placeholders, context_pairs=None, batch_size=100, max_degree=25,
            n2v_retrain=False, fixed_n2v=False,
            **kwargs):

        self.G = G
        self.nodes = G.nodes()
        self.id2idx = id2idx
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0

        self.nodes = np.random.permutation(G.nodes())
        self.adj, self.deg = self.construct_adj()
        self.test_adj = self.construct_test_adj()
        if context_pairs is None:
            edges = G.edges()
        else:
            edges = context_pairs
        self.train_edges = self.edges = np.random.permutation(edges)
        if not n2v_retrain:
            self.train_edges = self._remove_isolated(self.train_edges)
            self.val_edges = [e for e in G.edges() if G[e[0]][e[1]]['train_removed']]
        else:
            if fixed_n2v:
                self.train_edges = self.val_edges = self._n2v_prune(self.edges)
            else:
                self.train_edges = self.val_edges = self.edges

        print(len([n for n in G.nodes() if not G.node[n]['test'] and not G.node[n]['val']]), 'train nodes')
        print(len([n for n in G.nodes() if G.node[n]['test'] or G.node[n]['val']]), 'test nodes')
        self.val_set_size = len(self.val_edges)

    def _n2v_prune(self, edges):
        is_val = lambda n : self.G.node[n]["val"] or self.G.node[n]["test"]
        return [e for e in edges if not is_val(e[1])]

    def _remove_isolated(self, edge_list):
        new_edge_list = []
        missing = 0
        for n1, n2 in edge_list:
            if not n1 in self.G.node or not n2 in self.G.node:
                missing += 1
                continue
            if (self.deg[self.id2idx[n1]] == 0 or self.deg[self.id2idx[n2]] == 0) \
                    and (not self.G.node[n1]['test'] or self.G.node[n1]['val']) \
                    and (not self.G.node[n2]['test'] or self.G.node[n2]['val']):
                continue
            else:
                new_edge_list.append((n1,n2))
        print("Unexpected missing:", missing)
        return new_edge_list

    def construct_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        deg = np.zeros((len(self.id2idx),))

        for nodeid in self.G.nodes():
            if self.G.node[nodeid]['test'] or self.G.node[nodeid]['val']:
                continue
            neighbors = np.array([self.id2idx[neighbor] 
                for neighbor in self.G.neighbors(nodeid)
                if (not self.G[nodeid][neighbor]['train_removed'])])
            deg[self.id2idx[nodeid]] = len(neighbors)
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj, deg

    def construct_test_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        for nodeid in self.G.nodes():
            neighbors = np.array([self.id2idx[neighbor] 
                for neighbor in self.G.neighbors(nodeid)])
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_edges)

    def batch_feed_dict(self, batch_edges):
        batch1 = []
        batch2 = []
        for node1, node2 in batch_edges:
            batch1.append(self.id2idx[node1])
            batch2.append(self.id2idx[node2])

        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size'] : len(batch_edges)})
        feed_dict.update({self.placeholders['batch1']: batch1})
        feed_dict.update({self.placeholders['batch2']: batch2})

        return feed_dict

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_edges))
        batch_edges = self.train_edges[start_idx : end_idx]
        return self.batch_feed_dict(batch_edges)

    def num_training_batches(self):
        return len(self.train_edges) // self.batch_size + 1

    def val_feed_dict(self, size=None):
        edge_list = self.val_edges
        if size is None:
            return self.batch_feed_dict(edge_list)
        else:
            ind = np.random.permutation(len(edge_list))
            val_edges = [edge_list[i] for i in ind[:min(size, len(ind))]]
            return self.batch_feed_dict(val_edges)

    def incremental_val_feed_dict(self, size, iter_num):
        edge_list = self.val_edges
        val_edges = edge_list[iter_num*size:min((iter_num+1)*size, 
            len(edge_list))]
        return self.batch_feed_dict(val_edges), (iter_num+1)*size >= len(self.val_edges), val_edges

    def incremental_embed_feed_dict(self, size, iter_num):
        node_list = self.nodes
        val_nodes = node_list[iter_num*size:min((iter_num+1)*size, 
            len(node_list))]
        val_edges = [(n,n) for n in val_nodes]
        return self.batch_feed_dict(val_edges), (iter_num+1)*size >= len(node_list), val_edges

    def label_val(self):
        train_edges = []
        val_edges = []
        for n1, n2 in self.G.edges():
            if (self.G.node[n1]['val'] or self.G.node[n1]['test'] 
                    or self.G.node[n2]['val'] or self.G.node[n2]['test']):
                val_edges.append((n1,n2))
            else:
                train_edges.append((n1,n2))
        return train_edges, val_edges

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_edges = np.random.permutation(self.train_edges)
        self.nodes = np.random.permutation(self.nodes)
        self.batch_num = 0

class NodeMinibatchIterator(object):
    
    """ 
    This minibatch iterator iterates over nodes for supervised learning.

    G -- networkx graph
    id2idx -- dict mapping node ids to integer values indexing feature tensor
    placeholders -- standard tensorflow placeholders object for feeding
    label_map -- map from node ids to class values (integer or list)
    num_classes -- number of output classes
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    """
    def __init__(self, G, id2idx, 
            placeholders, label_map, num_classes, 
            batch_size=100, max_degree=25,
            **kwargs):

        self.G = G
        self.nodes = G.nodes()
        self.id2idx = id2idx
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0
        self.label_map = label_map
        self.num_classes = num_classes

        self.adj, self.deg = self.construct_adj()
        self.test_adj = self.construct_test_adj()

        self.val_nodes = [n for n in self.G.nodes() if self.G.node[n]['val']]
        self.test_nodes = [n for n in self.G.nodes() if self.G.node[n]['test']]

        self.no_train_nodes_set = set(self.val_nodes + self.test_nodes)
        self.train_nodes = set(G.nodes()).difference(self.no_train_nodes_set)
        # don't train on nodes that only have edges to test set
        self.train_nodes = [n for n in self.train_nodes if self.deg[id2idx[n]] > 0]

    def _make_label_vec(self, node):
        label = self.label_map[node]
        if isinstance(label, list):
            label_vec = np.array(label)
        else:
            label_vec = np.zeros((self.num_classes))
            class_ind = self.label_map[node]
            label_vec[class_ind] = 1
        return label_vec

    def construct_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        deg = np.zeros((len(self.id2idx),))

        for nodeid in self.G.nodes():
            if self.G.node[nodeid]['test'] or self.G.node[nodeid]['val']:
                continue
            neighbors = np.array([self.id2idx[neighbor] 
                for neighbor in self.G.neighbors(nodeid)
                if (not self.G[nodeid][neighbor]['train_removed'])])
            deg[self.id2idx[nodeid]] = len(neighbors)
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj, deg

    def construct_test_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        for nodeid in self.G.nodes():
            neighbors = np.array([self.id2idx[neighbor] 
                for neighbor in self.G.neighbors(nodeid)])
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_nodes)

    def batch_feed_dict(self, batch_nodes, val=False):
        batch1id = batch_nodes
        batch1 = [self.id2idx[n] for n in batch1id]
              
        labels = np.vstack([self._make_label_vec(node) for node in batch1id])
        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size'] : len(batch1)})
        feed_dict.update({self.placeholders['batch']: batch1})
        feed_dict.update({self.placeholders['labels']: labels})

        return feed_dict, labels

    def node_val_feed_dict(self, size=None, test=False):
        if test:
            val_nodes = self.test_nodes
        else:
            val_nodes = self.val_nodes
        if not size is None:
            val_nodes = np.random.choice(val_nodes, size, replace=True)
        # add a dummy neighbor
        ret_val = self.batch_feed_dict(val_nodes)
        return ret_val[0], ret_val[1]

    def incremental_node_val_feed_dict(self, size, iter_num, test=False):
        if test:
            val_nodes = self.test_nodes
        else:
            val_nodes = self.val_nodes
        val_node_subset = val_nodes[iter_num*size:min((iter_num+1)*size, 
            len(val_nodes))]

        # add a dummy neighbor
        ret_val = self.batch_feed_dict(val_node_subset)
        return ret_val[0], ret_val[1], (iter_num+1)*size >= len(val_nodes), val_node_subset

    def num_training_batches(self):
        return len(self.train_nodes) // self.batch_size + 1

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_nodes))
        batch_nodes = self.train_nodes[start_idx : end_idx]
        return self.batch_feed_dict(batch_nodes)

    def incremental_embed_feed_dict(self, size, iter_num):
        node_list = self.nodes
        val_nodes = node_list[iter_num*size:min((iter_num+1)*size, 
            len(node_list))]
        return self.batch_feed_dict(val_nodes), (iter_num+1)*size >= len(node_list), val_nodes

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_nodes = np.random.permutation(self.train_nodes)
        self.batch_num = 0


class CrossModalMinibatchIterator(object):
    """
    This minibatch iterator iterates over nodes for supervised learning.

    A -- adjacency matrix for image/text training data
    placeholders -- standard tensorflow placeholders object for feeding
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    """

    def __init__(self, A, train_num, placeholders,
                 batch_size=100, max_degree=25,
                 dataset='nuswide', percentage=1.0, unuse=False, **kwargs):

        self.A = A
        self.dataset = dataset
        self.train_num = train_num
        self.total_num = self.A.shape[0]
        self.labeled_num = int(self.train_num * percentage)
        self.unlabeled_num = self.train_num - self.labeled_num
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0
        self.test_nodes = range(self.total_num)
        self.train_nodes = self.test_nodes[:self.labeled_num]
        self.labeled_nodes = self.test_nodes[:self.labeled_num]
        self.unlabeled_nodes = self.test_nodes[self.labeled_num:self.train_num]
        self.train_A = np.zeros((self.total_num, self.total_num))
        if not unuse:
            self.train_A[:self.train_num, :train_num] = A[:self.train_num, :train_num]
        else:
            self.train_A[:self.labeled_num, :self.labeled_num] = A[:self.labeled_num, :self.labeled_num]
        '''use for comparison with only labeled data'''
        # self.train_A[:self.labeled_num, :self.labeled_num] = A[:self.labeled_num, :self.labeled_num]
        self.train_G = nx.convert_matrix.from_numpy_array(self.train_A, create_using=nx.DiGraph())
        self.adj, self.deg = self.construct_adj()
        self.test_G = nx.convert_matrix.from_numpy_array(self.A, create_using=nx.DiGraph())
        self.test_adj = self.construct_test_adj()
        # self.train_nodes = self.train_G.nodes()
        # save_dir = './data/' + self.dataset + '/adjacency'
        # if not os.path.exists(save_dir):
        #     os.mkdir(save_dir)
        #     self.train_A = np.zeros((self.total_num, self.total_num))
        #     self.train_A[:train_num, :train_num] = A[:train_num, :train_num]
        #     self.train_G = nx.convert_matrix.from_numpy_array(self.train_A, create_using=nx.DiGraph())
        #     self.adj, self.deg = self.construct_adj()
        #     import cPickle
        #     with open(save_dir+'/train_adj.pkl', 'w') as f:
        #         cPickle.dump(self.adj, f)
        #     with open(save_dir+'/train_deg.pkl', 'w') as f:
        #         cPickle.dump(self.deg, f)
        #     self.test_G = nx.convert_matrix.from_numpy_array(self.A, create_using=nx.DiGraph())
        #     self.test_adj = self.construct_test_adj()
        #     with open(save_dir+'/test_adj.pkl', 'w') as f:
        #         cPickle.dump(self.test_adj, f)
        #
        # else:
        #     import cPickle
        #     with open(save_dir + '/train_adj.pkl', 'rb') as f:
        #         self.adj = cPickle.load(f)
        #     with open(save_dir + '/test_adj.pkl', 'rb') as f:
        #         self.test_adj = cPickle.load(f)
        #     with open(save_dir + '/train_deg.pkl', 'rb') as f:
        #         self.deg = cPickle.load(f)

    def construct_adj(self):
        adj = self.total_num * np.ones((self.total_num + 1, self.max_degree))
        deg = np.zeros((self.total_num,))

        for nodeid in self.train_G.nodes():
            neighbors = np.array([neighbor for neighbor in self.train_G.neighbors(nodeid)])
            deg[nodeid] = len(neighbors)
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[nodeid, :] = neighbors
        return adj, deg

    def construct_test_adj(self):
        adj = self.total_num * np.ones((self.total_num + 1, self.max_degree))

        for nodeid in self.test_G.nodes():
            neighbors = np.array([neighbor for neighbor in self.test_G.neighbors(nodeid)])
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[nodeid, :] = neighbors
        return adj

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_nodes)

    def batch_feed_dict(self, batch_nodes, val=False):
        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size']: len(batch_nodes)})
        feed_dict.update({self.placeholders['batch']: batch_nodes})
        return feed_dict

    def num_training_batches(self):
        return len(self.train_nodes) // self.batch_size + 1

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_nodes))
        batch_nodes = self.train_nodes[start_idx: end_idx]
        return self.batch_feed_dict(batch_nodes), batch_nodes

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_nodes = np.random.permutation(self.train_nodes)
        self.batch_num = 0

    def incremental_node_val_feed_dict(self, size, iter_num):
        test_node_subset = self.test_nodes[iter_num*size:min((iter_num+1)*size, len(self.test_nodes))]
        # add a dummy neighbor
        batch_feed_dict = self.batch_feed_dict(test_node_subset)
        return batch_feed_dict, (iter_num+1)*size >= len(self.test_nodes)

    def next_minibatch_unlabeled_feed_dict(self, size, iter_num):
        unlabeled_node_subset = self.unlabeled_nodes[iter_num * size:min((iter_num + 1) * size, self.unlabeled_num)]
        # add a dummy neighbor
        batch_feed_dict = self.batch_feed_dict(unlabeled_node_subset)
        return batch_feed_dict, (iter_num + 1) * size >= self.unlabeled_num

    def next_minibatch_labeled_feed_dict(self, size, iter_num):
        labeled_node_subset = self.labeled_nodes[iter_num * size:min((iter_num + 1) * size, self.labeled_num)]
        # add a dummy neighbor
        batch_feed_dict = self.batch_feed_dict(labeled_node_subset)
        return batch_feed_dict, (iter_num + 1) * size >= self.labeled_num


class CrossModalMinibatchIteratorSimple(object):
    """
    This minibatch iterator iterates over nodes for supervised learning.

    A -- adjacency matrix for image/text training data
    placeholders -- standard tensorflow placeholders object for feeding
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    upperBound -- construct all train/test graphs according to labels
    """

    def __init__(self, A, train_num, placeholders,
                 batch_size=100, max_degree=25,
                 dataset='nuswide', adjacency_name=None, **kwargs):

        self.A = A
        self.dataset = dataset
        self.train_num = train_num
        self.total_num = self.A.shape[0]
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0
        self.test_nodes = range(self.total_num)
        self.train_nodes = self.test_nodes[:train_num]
        # self.train_nodes = self.train_G.nodes()
        save_dir = './data/' + self.dataset + '/adjacency'
        if adjacency_name:
            save_dir = './data/' + self.dataset + '/adjacency_' + adjacency_name
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            self.train_A = np.zeros((self.total_num, self.total_num))
            self.train_A[:train_num, :train_num] = A[:train_num, :train_num]
            self.test_A = A
            self.adj, self.deg = self.construct_adj()
            import cPickle
            with open(save_dir + '/train_adj.pkl', 'w') as f:
                cPickle.dump(self.adj, f)
            with open(save_dir + '/train_deg.pkl', 'w') as f:
                cPickle.dump(self.deg, f)
            self.test_adj = self.construct_test_adj()
            with open(save_dir+'/test_adj.pkl', 'w') as f:
                cPickle.dump(self.test_adj, f)
        else:
            import cPickle
            with open(save_dir + '/train_adj.pkl', 'rb') as f:
                self.adj = cPickle.load(f)
            with open(save_dir + '/test_adj.pkl', 'rb') as f:
                self.test_adj = cPickle.load(f)
            with open(save_dir + '/train_deg.pkl', 'rb') as f:
                self.deg = cPickle.load(f)
        print('train_adj:\n{}'.format(self.adj))
        print('train_deg:\n{}'.format(self.deg))
        print('test_adj:\n{}'.format(self.test_adj))

    def construct_adj(self):
        adj = self.total_num * np.ones((self.total_num + 1, self.max_degree))
        deg = np.zeros((self.total_num,))
        start_time = time.time()
        for nodeid in self.train_nodes:
            neighbors = np.where(self.train_A[nodeid] == 1)[0]
            deg[nodeid] = len(neighbors)
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[nodeid, :] = neighbors
        end_time = time.time()
        print('construct train adj in {}s!'.format(end_time-start_time))
        return adj, deg

    def construct_test_adj(self):
        adj = self.total_num * np.ones((self.total_num + 1, self.max_degree))

        for nodeid in self.test_nodes:
            neighbors = np.where(self.test_A[nodeid] == 1)[0]
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[nodeid, :] = neighbors
        return adj

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_nodes)

    def batch_feed_dict(self, batch_nodes, val=False):

        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size']: len(batch_nodes)})
        feed_dict.update({self.placeholders['batch']: batch_nodes})

        return feed_dict

    def num_training_batches(self):
        return len(self.train_nodes) // self.batch_size + 1

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_nodes))
        batch_nodes = self.train_nodes[start_idx: end_idx]
        return self.batch_feed_dict(batch_nodes), batch_nodes

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_nodes = np.random.permutation(self.train_nodes)
        self.batch_num = 0

    def incremental_node_val_feed_dict(self, size, iter_num):
        test_node_subset = self.test_nodes[iter_num*size:min((iter_num+1)*size, len(self.test_nodes))]
        # add a dummy neighbor
        batch_feed_dict = self.batch_feed_dict(test_node_subset)
        return batch_feed_dict, (iter_num+1)*size >= len(self.test_nodes)


class CrossModalMinibatchIteratorAdaptive(object):
    """
    This minibatch iterator iterates over nodes for supervised learning.

    A -- adjacency matrix for image/text training data
    placeholders -- standard tensorflow placeholders object for feeding
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    """

    def __init__(self, A, train_num, placeholders,train_labels,
                 batch_size=100, max_degree=25,
                 dataset='nuswide', percentage=1.0, **kwargs):

        self.A = A
        self.dataset = dataset
        self.train_num = train_num
        self.total_num = self.A.shape[0]
        self.labeled_num = int(self.train_num * percentage)
        self.unlabeled_num = self.train_num - self.labeled_num
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0
        self.test_nodes = range(self.total_num)
        self.train_nodes = self.test_nodes[:self.labeled_num]
        self.labeled_nodes = self.test_nodes[:self.labeled_num]
        self.unlabeled_nodes = self.test_nodes[self.labeled_num:self.train_num]
        self.test_A = A
        self.train_A = np.zeros((self.total_num, self.total_num))
        self.train_A[:self.train_num, :self.train_num] = A[:self.train_num, :self.train_num]
        self.labeled_labels = train_labels[:self.labeled_num]
        '''use for comparison with only labeled data'''
        # self.train_A[:self.labeled_num, :self.labeled_num] = A[:self.labeled_num, :self.labeled_num]
        self.adj, self.deg = self.construct_adj()
        self.test_adj = self.construct_test_adj()

    def construct_adj(self):
        adj = self.total_num * np.ones((self.total_num + 1, self.max_degree))
        deg = np.zeros((self.total_num,))
        start_time = time.time()
        for nodeid in self.train_nodes:
            neighbors = np.where(self.train_A[nodeid] == 1)[0]
            deg[nodeid] = len(neighbors)
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[nodeid, :] = neighbors
        end_time = time.time()
        print('construct train adj in {}s!'.format(end_time-start_time))
        return adj, deg

    def construct_test_adj(self):
        adj = self.total_num * np.ones((self.total_num + 1, self.max_degree))

        for nodeid in self.test_nodes:
            neighbors = np.where(self.test_A[nodeid] == 1)[0]
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[nodeid, :] = neighbors
        return adj

    def update_graph(self, confident_nodes, confident_labels):
        confident_num = len(confident_nodes)
        for index in range(confident_num):
            index_label = confident_labels[index]
            index_s = (np.dot(self.labeled_labels, index_label)>0).astype(int)
            self.train_A[confident_nodes[index]] = 0
            self.test_A[confident_nodes[index], :self.train_num] = 0
            self.train_A[confident_nodes[index], :self.labeled_num] = index_s
            self.test_A[confident_nodes[index], :self.labeled_num] = index_s
            # due to time complexity, remain labeled adj unchanged
            # self.train_A[:self.train_num, confident_nodes[index]] = index_s
            for index_ in range(confident_num):
                sij = (np.dot(confident_labels[index_], index_label)>0).astype(int)
                self.train_A[confident_nodes[index], confident_nodes[index_]] = sij
                self.test_A[confident_nodes[index], confident_nodes[index_]] = sij
        # update train_adj
        for node in confident_nodes:
            neighbors = np.where(self.train_A[node] == 1)[0]
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            self.adj[node, :] = neighbors
            self.deg[node] = len(neighbors)
        # update test adj
        for node in confident_nodes:
            neighbors = np.where(self.test_A[node] == 1)[0]
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            self.test_adj[node, :] = neighbors
        self.labeled_labels = np.vstack((self.labeled_labels, confident_labels))
        self.train_nodes = np.append(self.train_nodes, confident_nodes)
        self.unlabeled_nodes = np.delete(self.unlabeled_nodes, confident_nodes)
        self.labeled_nodes = np.append(self.train_nodes, confident_nodes)
        confident_num = len(confident_nodes)
        self.train_num = len(self.train_nodes)
        self.labeled_num += confident_num
        self.unlabeled_num -= confident_num

    def show_info(self):
        print('-----------minibatch info start------------')
        print('train_num:', len(self.train_nodes), self.train_num,
              'labeled_num:', len(self.labeled_nodes),self.labeled_num,
              'unlabeled_num:', len(self.unlabeled_nodes), self.unlabeled_num)
        print('-----------minibatch info end------------')

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_nodes)

    def batch_feed_dict(self, batch_nodes, val=False):
        feed_dict = dict()
        feed_dict.update({self.placeholders['batch_size']: len(batch_nodes)})
        feed_dict.update({self.placeholders['batch']: batch_nodes})
        return feed_dict

    def num_training_batches(self):
        return len(self.train_nodes) // self.batch_size + 1

    def next_minibatch_feed_dict(self):
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_nodes))
        batch_nodes = self.train_nodes[start_idx: end_idx]
        return self.batch_feed_dict(batch_nodes), batch_nodes

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        self.train_nodes = np.random.permutation(self.train_nodes)
        self.batch_num = 0

    def incremental_node_val_feed_dict(self, size, iter_num):
        test_node_subset = self.test_nodes[iter_num*size:min((iter_num+1)*size, len(self.test_nodes))]
        # add a dummy neighbor
        batch_feed_dict = self.batch_feed_dict(test_node_subset)
        return batch_feed_dict, (iter_num+1)*size >= len(self.test_nodes)

    def next_minibatch_unlabeled_feed_dict(self, size, iter_num):
        unlabeled_node_subset = self.unlabeled_nodes[iter_num * size:min((iter_num + 1) * size, self.unlabeled_num)]
        # add a dummy neighbor
        batch_feed_dict = self.batch_feed_dict(unlabeled_node_subset)
        return batch_feed_dict, (iter_num + 1) * size >= self.unlabeled_num

    def next_minibatch_labeled_feed_dict(self, size, iter_num):
        labeled_node_subset = self.labeled_nodes[iter_num * size:min((iter_num + 1) * size, self.labeled_num)]
        # add a dummy neighbor
        batch_feed_dict = self.batch_feed_dict(labeled_node_subset)
        return batch_feed_dict, (iter_num + 1) * size >= self.labeled_num


