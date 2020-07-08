import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from evaluation import calc_cosine
import sys
import cPickle
import random
import os

seed = 1
random.seed(seed)
np.random.seed(seed)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_wiki():
    """
    train_img_feats:2173*4096
    train_txt_vecs:2173*5000
    train_labels:2173*1
    test_txt_vecs:693*5000
    test_img_feats:693*4096
    test_labels:693*1
    """
    with open('./data/wikipedia_dataset/train_img_feats.pkl', 'rb') as f:
        train_img_feats = cPickle.load(f)
    with open('./data/wikipedia_dataset/train_txt_vecs.pkl', 'rb') as f:
        train_txt_vecs = cPickle.load(f)
    with open('./data/wikipedia_dataset/train_labels_onehot.pkl', 'rb') as f:
        train_labels = cPickle.load(f)
    with open('./data/wikipedia_dataset/test_img_feats.pkl', 'rb') as f:
        test_img_feats = cPickle.load(f)
    with open('./data/wikipedia_dataset/test_txt_vecs.pkl', 'rb') as f:
        test_txt_vecs = cPickle.load(f)
    with open('./data/wikipedia_dataset/test_labels_onehot.pkl', 'rb') as f:
        test_labels = cPickle.load(f)

    train_img_feats = np.array(train_img_feats)
    train_txt_vecs = np.array(train_txt_vecs)
    train_labels = np.array(train_labels)
    test_img_feats = np.array(test_img_feats)
    test_txt_vecs = np.array(test_txt_vecs)
    test_labels = np.array(test_labels)

    return train_img_feats, train_txt_vecs, train_labels, test_img_feats, test_txt_vecs, test_labels


def load_Flickr():
    """
    image:20015*224*224*3 train:16012 test:4003
    tags:20015*1386
    labels:20015*24
    """
    SAVE_DIR = './data/Flickr-25k-relu/'
    train_img_path = SAVE_DIR + 'train_img_feats.pkl'
    train_txt_path = SAVE_DIR + 'train_bow.pkl'
    train_labels_path = SAVE_DIR + 'train_labels.pkl'

    test_img_path = SAVE_DIR + 'test_img_feats.pkl'
    test_txt_path = SAVE_DIR + 'test_bow.pkl'
    test_labels_path = SAVE_DIR + 'test_labels.pkl'
    with open(train_img_path, 'rb') as f:
        train_img_feats = cPickle.load(f)
    with open(train_txt_path, 'rb') as f:
        train_txt_vecs = cPickle.load(f)
    with open(train_labels_path, 'rb') as f:
        train_labels = cPickle.load(f)
    with open(test_img_path, 'rb') as f:
        test_img_feats = cPickle.load(f)
    with open(test_txt_path, 'rb') as f:
        test_txt_vecs = cPickle.load(f)
    with open(test_labels_path, 'rb') as f:
        test_labels = cPickle.load(f)

    train_img_feats = np.array(train_img_feats)
    train_txt_vecs = np.array(train_txt_vecs)
    train_labels = np.array(train_labels)
    test_img_feats = np.array(test_img_feats)
    test_txt_vecs = np.array(test_txt_vecs)
    test_labels = np.array(test_labels)

    return train_img_feats, train_txt_vecs, train_labels, test_img_feats, test_txt_vecs, test_labels

def load_nuswide():
    """
    load cross modal (img,txt) feature
    :param dataset_str: dataset name
    :return:
    """
    with open('./data/nuswide/img_train_id_feats.pkl', 'rb') as f:
        train_img_feats = cPickle.load(f)
    with open('./data/nuswide/train_id_bow.pkl', 'rb') as f:
        train_txt_vecs = cPickle.load(f)
    with open('./data/nuswide/train_id_label_map.pkl', 'rb') as f:
        train_labels = cPickle.load(f)
    # load test data
    with open('./data/nuswide/img_test_id_feats.pkl', 'rb') as f:
        test_img_feats = cPickle.load(f)
    with open('./data/nuswide/test_id_bow.pkl', 'rb') as f:
        test_txt_vecs = cPickle.load(f)
    with open('./data/nuswide/test_id_label_map.pkl', 'rb') as f:
        test_labels = cPickle.load(f)
    # index of trainging set and test set
    # not equal to the index in training set,this index is shuffled
    with open('./data/nuswide/train_ids.pkl', 'rb') as f:
        train_ids = cPickle.load(f)
    with open('./data/nuswide/test_ids.pkl', 'rb') as f:
        test_ids = cPickle.load(f)

    np.random.shuffle(train_ids)
    np.random.shuffle(test_ids)

    train_img_feats = [train_img_feats[i] for i in train_ids]
    train_txt_vecs = [train_txt_vecs[i] for i in train_ids]
    train_labels = [train_labels[i] for i in train_ids]

    test_img_feats = [test_img_feats[i] for i in test_ids]
    test_txt_vecs = [test_txt_vecs[i] for i in test_ids]
    test_labels = [test_labels[i] for i in test_ids]

    # train_img_feats = sp.csr_matrix(train_img_feats)
    # train_txt_vecs = sp.csr_matrix(train_txt_vecs)
    # train_labels = sp.csr_matrix(train_labels)
    # test_img_feats = sp.csr_matrix(test_img_feats)
    # test_txt_vecs = sp.csr_matrix(test_txt_vecs)
    # test_labels = sp.csr_matrix(test_labels)

    train_img_feats = np.array(train_img_feats)
    train_txt_vecs = np.array(train_txt_vecs)
    train_labels = np.array(train_labels)
    test_img_feats = np.array(test_img_feats)
    test_txt_vecs = np.array(test_txt_vecs)
    test_labels = np.array(test_labels)

    return train_img_feats, train_txt_vecs, train_labels, test_img_feats, test_txt_vecs, test_labels


def load_semi_nuswide(percentage):
    """
    load cross modal (img,txt) feature
    :param dataset_str: dataset name
    :return:
    """
    with open('./data/semi-nuswide/img_train_id_feats.pkl', 'rb') as f:
        train_img_feats = cPickle.load(f)
    with open('./data/semi-nuswide/train_id_bow.pkl', 'rb') as f:
        train_txt_vecs = cPickle.load(f)
    with open('./data/semi-nuswide/train_id_label_map.pkl', 'rb') as f:
        train_labels = cPickle.load(f)
    # load test data
    with open('./data/semi-nuswide/img_test_id_feats.pkl', 'rb') as f:
        test_img_feats = cPickle.load(f)
    with open('./data/semi-nuswide/test_id_bow.pkl', 'rb') as f:
        test_txt_vecs = cPickle.load(f)
    with open('./data/semi-nuswide/test_id_label_map.pkl', 'rb') as f:
        test_labels = cPickle.load(f)
    # index of trainging set and test set
    # not equal to the index in training set,this index is shuffled
    with open('./data/semi-nuswide/train_ids.pkl', 'rb') as f:
        train_ids = cPickle.load(f)
    with open('./data/semi-nuswide/test_ids.pkl', 'rb') as f:
        test_ids = cPickle.load(f)

    np.random.shuffle(train_ids)
    np.random.shuffle(test_ids)

    train_img_feats = [train_img_feats[i] for i in train_ids]
    train_txt_vecs = [train_txt_vecs[i] for i in train_ids]
    train_labels = [train_labels[i] for i in train_ids]

    test_img_feats = [test_img_feats[i] for i in test_ids]
    test_txt_vecs = [test_txt_vecs[i] for i in test_ids]
    test_labels = [test_labels[i] for i in test_ids]

    train_img_feats = np.array(train_img_feats)
    train_txt_vecs = np.array(train_txt_vecs)
    train_labels = np.array(train_labels)
    test_img_feats = np.array(test_img_feats)
    test_txt_vecs = np.array(test_txt_vecs)
    test_labels = np.array(test_labels)

    num_train = train_img_feats.shape[0]
    used_num = int(num_train * percentage)
    train_img_feats = train_img_feats[:used_num, :]
    train_txt_vecs = train_txt_vecs[:used_num, :]
    train_labels = train_labels[:used_num, :]

    return train_img_feats, train_txt_vecs, train_labels, test_img_feats, test_txt_vecs, test_labels


'''
    use all labels to construct the graph but only use the train subgraph when trainning 
'''
def compute_adj(dataset_str, train_labels, test_labels):
    """
    :param dataset_str: dataset name
    :param train_labels: train labels
    :param test_labels: test labels
    :return: csr_matrix
    """
    data_path = 'data/' + dataset_str + '/cross_modal_adj' + '.pkl'
    if os.path.exists(data_path):
        with open(data_path, 'r') as f:
            adj = cPickle.load(f)
            return adj
    else:
        labels = np.vstack((train_labels, test_labels))
        adj = np.dot(labels, np.transpose(labels))
        sp_adj = sp.csr_matrix(adj, dtype=np.float32)
        with open(data_path, 'w') as f:
            cPickle.dump(sp_adj, f)
        return adj


def compute_img_adj_semi(dataset_str, train, test, train_labels, label_percentage, threshold, semi_threshold=None):
    """
    :param dataset_str: dataset name
    :param train: train feature
    :param test: test feature
    :param train_labels: train labels
    :param threshold: similarity threshold
    :param semi_threshold: threshold for unlabeled data in training set
    :param label_percentage: percentage of labeled data
    :return: csr_matrix
    """
    if semi_threshold is None:
        data_path = 'data/' + dataset_str + '/img_adj_' + str(threshold) + '_' + str(label_percentage) + '.pkl'
    else:
        data_path = 'data/' + dataset_str + '/img_adj_' + str(threshold) + '_' + str(semi_threshold) + '_' + str(
            label_percentage) + '.pkl'

    if os.path.exists(data_path):
        print('load img adj from : {}'.format(data_path))
        with open(data_path, 'r') as f:
            img_adj = cPickle.load(f)
            return img_adj
    else:
        print('create img adj to :{}'.format(data_path))
        len_train = len(train)
        len_test = len(test)
        labeled_num = int(len_train * label_percentage)
        unlabeled_num = len_train - labeled_num
        labeled_train_data = train[0:labeled_num, :]
        used_label = train_labels[0:labeled_num]
        unlabeled_train_data = train[labeled_num:, :]

        shape = (len_test + len_train, len_test + len_train)
        S_img = (np.dot(used_label, np.transpose(used_label)) > 0).astype(int)

        img_adj = np.zeros(shape, dtype=np.float32)
        img_adj[:labeled_num, :labeled_num] = S_img

        for index in range(unlabeled_num):
            temp = unlabeled_train_data[index]
            dists = calc_cosine(labeled_train_data, temp)
            sorted_idx = np.argsort(dists)
            for k in range(semi_threshold):
                img_adj[labeled_num + index][sorted_idx[k]] = 1
                img_adj[sorted_idx[k]][labeled_num + index] = 1
            print("process unlabeld: [{0}/{1}]".format(index, unlabeled_num))

        for index in range(len_test):
            temp = test[index]
            dists = calc_cosine(labeled_train_data, temp)
            sorted_idx = np.argsort(dists)
            for k in range(threshold):
                img_adj[len_train + index][sorted_idx[k]] = 1
                img_adj[sorted_idx[k]][len_train + index] = 1
            print("process test: [{0}/{1}]".format(index, len_test))
        img_adj = sp.csr_matrix(img_adj, dtype=np.float32)
        with open(data_path, 'w') as f:
            cPickle.dump(img_adj, f)
        return img_adj


def compute_txt_adj_semi(dataset_str, train, test, train_labels, label_percentage, threshold, semi_threshold=None):
    """
    :param dataset_str: dataset name
    :param train: train feature
    :param test: test feature
    :param train_labels: train labels
    :param threshold: similarity threshold
    :param semi_threshold: threshold for unlabeled data in training set
    :param label_percentage: percentage of labeled data
    :return: csr_matrix
    """
    if semi_threshold is None:
        data_path = 'data/' + dataset_str + '/txt_adj_' + str(threshold) + '_' + str(label_percentage) + '.pkl'
    else:
        data_path = 'data/' + dataset_str + '/txt_adj_' + str(threshold) + '_' + str(semi_threshold) + '_' + str(
            label_percentage) + '.pkl'

    if os.path.exists(data_path):
        print('load txt adj from : {}'.format(data_path))
        with open(data_path, 'r') as f:
            txt_adj = cPickle.load(f)
            return txt_adj
    else:
        print('create txt adj to :{}'.format(data_path))
        len_train = len(train)
        len_test = len(test)
        labeled_num = int(len_train * label_percentage)
        unlabeled_num = len_train - labeled_num
        labeled_train_data = train[0:labeled_num, :]
        used_label = train_labels[0:labeled_num]
        unlabeled_train_data = train[labeled_num:, :]

        shape = (len_test + len_train, len_test + len_train)
        S_txt = (np.dot(used_label, np.transpose(used_label)) > 0).astype(int)

        img_adj = np.zeros(shape, dtype=np.float32)
        img_adj[:labeled_num, :labeled_num] = S_txt

        for index in range(unlabeled_num):
            temp = unlabeled_train_data[index]
            dists = calc_cosine(labeled_train_data, temp)
            sorted_idx = np.argsort(dists)
            for k in range(semi_threshold):
                img_adj[labeled_num + index][sorted_idx[k]] = 1
                img_adj[sorted_idx[k]][labeled_num + index] = 1
            print("process unlabeld: [{0}/{1}]".format(index, unlabeled_num))

        for index in range(len_test):
            temp = test[index]
            dists = calc_cosine(labeled_train_data, temp)
            sorted_idx = np.argsort(dists)
            for k in range(threshold):
                img_adj[len_train + index][sorted_idx[k]] = 1
                img_adj[sorted_idx[k]][len_train + index] = 1
            print("process test: [{0}/{1}]".format(index, len_test))
        txt_adj = sp.csr_matrix(img_adj, dtype=np.float32)
        with open(data_path, 'w') as f:
            cPickle.dump(txt_adj, f)
        return txt_adj


def compute_img_adj(dataset_str, train, test, train_labels, density):
    """
    :param dataset_str: dataset name
    :param train: train feature
    :param test: test feature
    :param train_labels: train labels
    :param density: k most similar train data to test data
    :return: csr_matrix
    """
    data_path = 'data/'+dataset_str+'/img_adj_'+str(density)+'.pkl'
    if os.path.exists(data_path):
        with open(data_path,'r') as f:
            img_adj = cPickle.load(f)
            return img_adj
    else:
        # train = train.toarray()
        # test = test.toarray()
        # train_labels = train_labels.toarray()
        len_train = len(train)
        len_test = len(test)
        shape = (len_test+len_train, len_test+len_train)
        S_img = (np.dot(train_labels, np.transpose(train_labels)) > 0).astype(int)

        img_adj = np.zeros(shape)
        img_adj[:len_train, :len_train] = S_img
        for index in range(len_test):
            temp = test[index]
            diffs = train - temp
            dists = np.linalg.norm(diffs, axis=1)
            sorted_idx = np.argsort(dists)
            for k in range(density):
                img_adj[sorted_idx[k]][len_train+index] = 1
                img_adj[len_train + index][sorted_idx[k]] = 1
            print("process: [{0}/{1}]".format(index, len_test))
        img_adj = sp.csr_matrix(img_adj, dtype=np.int32)
        with open(data_path,'w') as f:
            cPickle.dump(img_adj,f)
        return img_adj

def compute_txt_adj(dataset_str, train, test, train_labels, density):
    """
    :param dataset_str: dataset name
    :param train: train feature
    :param test: test feature
    :param train_labels: train labels
    :param density: k most similar train data to test data
    :return: csr_matrix
    """
    data_path = 'data/'+dataset_str+'/txt_adj_'+str(density)+'.pkl'
    if os.path.exists(data_path):
        with open(data_path,'r') as f:
            txt_adj = cPickle.load(f)
            return txt_adj
    else:

        # train = train.toarray()
        # test = test.toarray()
        # train_labels = train_labels.toarray()
        len_train = len(train)
        len_test = len(test)
        shape = (len_test+len_train, len_test+len_train)
        S_txt = (np.dot(train_labels, np.transpose(train_labels)) > 0).astype(int)
        txt_adj = np.zeros(shape)
        txt_adj[:len_train, :len_train] = S_txt
        train = np.array(train)
        for index in range(len_test):
            temp = test[index]
            dists = -1 * np.dot(temp, train.transpose())
            sorted_idx = np.argsort(dists)
            for k in range(density):
                txt_adj[sorted_idx[k]][len_train + index] = 1
                txt_adj[len_train + index][sorted_idx[k]] = 1
            print("process: [{0}/{1}]".format(index, len_test))
        txt_adj = sp.csr_matrix(txt_adj, dtype=np.int32)
        with open(data_path,'w') as f:
            cPickle.dump(txt_adj,f)
        return txt_adj

def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def preprocess_adj_dense(adj):
    """Preprocessing of adjacency matrix for simple GCN model and adj is not sparse"""
    adj_normalized = normalize_adj(adj + np.eye(adj.shape[0]))
    return adj_normalized.toarray()


def construct_feed_dict(img_feature, txt_feature, img_support, txt_support, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['img_feature']: img_feature})
    feed_dict.update({placeholders['txt_feature']: txt_feature})
    feed_dict.update({placeholders['img_support']: img_support})
    feed_dict.update({placeholders['txt_support']: txt_support})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
