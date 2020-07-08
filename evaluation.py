import numpy as np
import time
import scipy
import matplotlib.pyplot as plt


def incremental_evaluate(sess, op, minibatch_iter, size, hash_bit, placeholders):
    iter_num = 0
    finished = False
    results = np.zeros((1, hash_bit))
    while not finished:
        feed_dict_val, finished = minibatch_iter.incremental_node_val_feed_dict(size, iter_num)
        feed_dict_val.update({placeholders['dropout']: 0.})
        node_outs_val = sess.run(op, feed_dict=feed_dict_val)
        iter_num += 1
        results = np.vstack((results, node_outs_val))
    return results[1:]


def incremental_evaluate_label_labeled(sess, op, minibatch_iter, size, labels, placeholders):
    iter_num = 0
    finished = False
    label_dim = labels.shape[1]
    results = np.zeros((1, label_dim))
    while not finished:
        feed_dict_val, finished = minibatch_iter.next_minibatch_labeled_feed_dict(size, iter_num)
        feed_dict_val.update({placeholders['dropout']: 0.})
        node_outs_val = sess.run(op, feed_dict=feed_dict_val)
        iter_num += 1
        results = np.vstack((results, node_outs_val))
    results = results[1:]
    labels_single = np.argmax(labels, axis=1)
    predict_single = np.argmax(results, axis=1)
    correct = np.where(predict_single == labels_single)[0]
    accuracy = (1.0 * len(correct)) / len(labels_single)
    return accuracy


def incremental_evaluate_label_unlabeled(sess, op, minibatch_iter, size, labels, placeholders):
    iter_num = 0
    finished = False
    label_dim = labels.shape[1]
    results = np.zeros((1, label_dim))
    while not finished:
        feed_dict_val, finished = minibatch_iter.next_minibatch_unlabeled_feed_dict(size, iter_num)
        feed_dict_val.update({placeholders['dropout']: 0.})
        node_outs_val = sess.run(op, feed_dict=feed_dict_val)
        iter_num += 1
        results = np.vstack((results, node_outs_val))
    results = results[1:]
    labels_single = np.argmax(labels, axis=1)
    predict_single = np.argmax(results, axis=1)
    confident_count = 0.001
    confident_true_count = 0
    for i in range(predict_single.shape[0]):
        prob = results[i, predict_single[i]]
        if prob > 0.98:
            confident_count += 1
            if predict_single[i] == labels_single[i]:
                confident_true_count += 1
    print('confident acc: {}, confident_count:{}, confident_true_count:{}'.format(confident_true_count*1.0/confident_count, confident_count, confident_true_count))
    correct = np.where(predict_single == labels_single)[0]
    accuracy = (1.0 * len(correct)) / (len(labels_single) + 0.001)
    return accuracy


def predict_unlabeled(sess, op, minibatch_iter, size, label_dim, placeholders):
    iter_num = 0
    finished = False
    results = np.zeros((1, label_dim))
    while not finished:
        feed_dict_val, finished = minibatch_iter.next_minibatch_unlabeled_feed_dict(size, iter_num)
        feed_dict_val.update({placeholders['dropout']: 0.})
        node_outs_val = sess.run(op, feed_dict=feed_dict_val)
        iter_num += 1
        results = np.vstack((results, node_outs_val))
    results = results[1:]
    predict_single = np.argmax(results, axis=1)
    confident_nodes = []
    confident_labels = np.zeros(shape=results.shape)
    for i in range(predict_single.shape[0]):
        prob = results[i, predict_single[i]]
        confident_labels[i, predict_single[i]] = 1
        # default 0.98
        if prob > 0.98:
            confident_nodes.append(i)
    return confident_nodes, confident_labels


def findStartIndex(precision, recall):
    for i in range(len(precision)):
        if (precision[i] > 0.001) or (recall[i] > 0.001):
            return i
    return len(precision)-1


def plotRPcurve(precision=None, recall=None, style=None, label=None, start=-1):
    if precision is not None:
        if not start == -1:
            plt.plot(recall[start:], precision[start:], style, label=label, linewidth=1.3, markersize=4)
        else:
            start = findStartIndex(precision, recall)
            plt.plot(recall[start:], precision[start:], style, label=label, linewidth=1.3, markersize=4)


def calc_precision_recall(qB, rB, query_L, retrieval_L, eps=2.2204e-16):
    """
    calculate precision recall
    Input:
        query_L: 0-1 label matrix (numQuery * numLabel) for query set.
        retrieval_L: 0-1 label matrix (numQuery * numLabel) for retrieval set.
        qB: compressed binary code for query set.
        rB: compressed binary code for retrieval set.
    Output:
        Pre: maxR-dims vector. Precision within different hamming radius.
        Rec: maxR-dims vector. Recall within different hamming radius.
    """
    Wtrue = (np.dot(query_L, np.transpose(retrieval_L)) > 0).astype(int)
    Dhamm = calc_hammingDist(qB, rB)

    maxHamm = int(np.max(Dhamm))
    totalGoodPairs = np.sum(Wtrue)

    precision = np.zeros((maxHamm+1, 1))
    recall = np.zeros((maxHamm+1, 1))
    for i in range(maxHamm+1):
        j = (Dhamm <= (i + 0.001)).astype(int)
        retrievalPairs = np.sum(j)
        retrievalGoodPairs = np.sum(np.multiply(Wtrue, j))
        print(retrievalGoodPairs, retrievalPairs)
        precision[i] = retrievalGoodPairs * 1.0 / (retrievalPairs + eps)
        recall[i] = retrievalGoodPairs * 1.0 / totalGoodPairs

    return precision, recall


def calc_map(qB, rB, query_L, retrieval_L):
    """from deep cross modal hashing"""
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    num_query = query_L.shape[0]
    map = 0
    for iter in xrange(num_query):
        gnd = (np.dot(query_L[iter, :], retrieval_L.transpose()) > 0).astype(np.float32)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map = map + np.mean(count / (tindex))
    map = map / num_query
    return map


def calculate_map(test_img_feats_trans, test_txt_vecs_trans, test_labels):
    """Calculate top-50 mAP"""
    start = time.time()
    avg_precs = []
    all_precs = []
    all_k = [50]
    for k in all_k:
        for i in range(len(test_txt_vecs_trans)):
            query_label = test_labels[i]

            # distances and sort by distances
            wv = test_txt_vecs_trans[i]
            #dists = calc_l2_norm(wv, test_img_feats_trans)
            dists = calc_hammingDist(wv, test_img_feats_trans)
            sorted_idx = np.argsort(dists)

            # for each k do top-k
            precs = []
            for topk in range(1, k + 1):
                hits = 0
                top_k = sorted_idx[0: topk]
                # if query_label != test_labels[top_k[-1]]:
                #     continue
                if np.any(query_label != test_labels[top_k[-1]]):
                    continue
                for ii in top_k:
                    retrieved_label = test_labels[ii]
                    if np.all(retrieved_label == query_label):
                        hits += 1
                precs.append(float(hits) / float(topk))
            if len(precs) == 0:
                precs.append(0)
            avg_precs.append(np.average(precs))
        mean_avg_prec = np.mean(avg_precs)
        all_precs.append(mean_avg_prec)
    print('[Eval - txt2img] mAP: %f in %4.4fs' % (all_precs[0], (time.time() - start)))

    avg_precs = []
    all_precs = []
    all_k = [50]
    for k in all_k:
        for i in range(len(test_img_feats_trans)):
            query_img_feat = test_img_feats_trans[i]
            ground_truth_label = test_labels[i]
            # calculate distance and sort
            #dists = calc_l2_norm(query_img_feat, test_txt_vecs_trans)
            dists = calc_hammingDist(query_img_feat, test_txt_vecs_trans)
            sorted_idx = np.argsort(dists)

            # for each k in top-k
            precs = []
            for topk in range(1, k + 1):
                hits = 0
                top_k = sorted_idx[0: topk]
                if np.any(ground_truth_label != test_labels[top_k[-1]]):
                    continue
                for ii in top_k:
                    retrieved_label = test_labels[ii]
                    if np.all(ground_truth_label == retrieved_label):
                        hits += 1
                precs.append(float(hits) / float(topk))
            if len(precs) == 0:
                precs.append(0)
            avg_precs.append(np.average(precs))
        mean_avg_prec = np.mean(avg_precs)
        all_precs.append(mean_avg_prec)
    print('[Eval - img2txt] mAP: %f in %4.4fs' % (all_precs[0], (time.time() - start)))


def calc_map_category(qB, rB, query_L, retrieval_L):
    """calculate map score for each category"""
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    num_query = query_L.shape[0]
    num_category = query_L.shape[1]
    map_category = np.zeros(num_category)

    for i in range(num_category):
        q_index = np.where(query_L[:, i] == 1)[0]
        qB_i = qB[q_index, :]
        query_L_i = query_L[q_index, :]
        map_i = calc_map(qB_i, rB, query_L_i, retrieval_L)
        map_category[i] = map_i
    return map_category


def calc_hammingDist(request, retrieval_all):
    K = retrieval_all.shape[1]
    distH = 0.5 * (K - np.dot(request, retrieval_all.transpose()))
    return distH


def calc_l2_norm(request, retrieval_all):
    diffs = retrieval_all - request
    dists = np.linalg.norm(diffs, axis=1)
    return dists


def calc_cosine(matrix, vector):
    """
    Compute the cosine distances between each row of matrix and vector.
    """
    v = vector.reshape(1, -1)
    return scipy.spatial.distance.cdist(matrix, v, 'cosine').reshape(-1)


def calc_inner_product(matrix, vector):
    """
    Compute the inner-product between each row of matrix and vector.
    """
    return np.dot(matrix, np.transpose(vector))


def calc_gaussian(matrix, vector, gamma=None):
    """
    Compute the rbf (gaussian) kernel between X and Y::
    K(x, y) = exp(-gamma ||x-y||^2)
    from sklearn rbf-kernel
    :param matrix:
    :param vector:
    :param sigma:
    :return: gaussian kernel
    """
    if gamma is None:
        gamma = 1.0 / matrix.shape[1]

    K = np.power(calc_l2_norm(matrix, vector), 2)
    K *= -gamma
    K = np.exp(K)
    return 1-K
