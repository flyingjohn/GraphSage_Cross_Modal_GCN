from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import cPickle
from graphsage.minibatch import CrossModalMinibatchIteratorAdaptive
from graphsage.neigh_samplers import UniformNeighborSampler
from graphsage.models import SAGEInfo
from utils import *
from evaluation import *
from networks import ImgModel, TxtModel, label_classifier, siamese_net, domain_adversarial
import os
from datetime import datetime
import matplotlib.pyplot as plt
import random

import numpy as np

# Set random seed
seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)
# random.seed(seed)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,6'

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'is training or test')
flags.DEFINE_float('threshold', 0.65, 'density for construct graph')
flags.DEFINE_integer('kNum', 20, 'density for construct graph')
flags.DEFINE_integer('semi_kNum', 20, 'density for construct graph')
flags.DEFINE_float('percentage', 1.0, 'density for construct graph')
flags.DEFINE_string('dataset', 'Flickr-25k-relu', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_integer('max_degree', 128, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of samples in layer 2')
#img-model setting
flags.DEFINE_integer('img_gc1', 1000, 'Initial learning rate.')
#txt-model setting
flags.DEFINE_integer('txt_gc1', 300, 'Initial learning rate.')
#out-put dim
flags.DEFINE_integer('hash_bit', 24, 'Initial learning rate.')
flags.DEFINE_integer('siamese_bit', 128, 'Initial learning rate.')
flags.DEFINE_integer('siamese_bit_1', 64, 'Initial learning rate.')
#learning rate
flags.DEFINE_float('lr_img', 0.0001, 'Initial learning rate.')
flags.DEFINE_float('lr_txt', 0.0001, 'Initial learning rate.')
flags.DEFINE_float('lr_domain', 0.0001, 'Initial learning rate.')
flags.DEFINE_float('lr_label', 0.00005, 'Initial learning rate.')
flags.DEFINE_integer('batch_size', 256, 'batch size.')
#other setting
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
#flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
# all img_feats, labels, txt_vecs, adj are csr_matrix
load_time = time.time()
train_img_feats, train_txt_vecs, train_labels, test_img_feats, test_txt_vecs, test_labels = load_Flickr()
img_dim = train_img_feats.shape[1]
label_dim = train_labels.shape[1]
txt_dim = train_txt_vecs.shape[1]
train_num = train_txt_vecs.shape[0]
test_num = test_txt_vecs.shape[0]
total_num = train_num + test_num
labeled_num = int(train_num * FLAGS.percentage)
unlabeled_num = train_num - labeled_num
unlabeled_labels = train_labels[labeled_num:]
labeled_labels = train_labels[:labeled_num]

# get img/txt adjacent matrix
img_adj = compute_img_adj_semi(FLAGS.dataset, train_img_feats, test_img_feats, train_labels, FLAGS.percentage, FLAGS.kNum, FLAGS.semi_kNum)
txt_adj = compute_txt_adj_semi(FLAGS.dataset, train_txt_vecs, test_txt_vecs, train_labels, FLAGS.percentage, FLAGS.kNum, FLAGS.semi_kNum)
img_adj = img_adj.toarray()
txt_adj = txt_adj.toarray()
adj = txt_adj
# adj = np.logical_or(img_adj, txt_adj).astype(np.int32)
# adj = adj - np.eye(total_num)
print("load adjacent matrix finished in {}s!".format(time.time()-load_time))

# concat training feature and test feature
img_feature = np.vstack((train_img_feats, test_img_feats))
print('img_feature:\n{}'.format(img_feature))
txt_feature = np.vstack((train_txt_vecs, test_txt_vecs))
print('txt_feature:\n{}'.format(txt_feature))
labels = np.vstack((train_labels, test_labels))
print('labels:\n{}'.format(labels))

# Define placeholders
placeholders = {
    'dropout': tf.placeholder_with_default(0., shape=(),name='dropout'),
    'batch_size': tf.placeholder(tf.int32, name='batch_size'),
    'batch': tf.placeholder(tf.int32, shape=[None], name='batch_nodes'),
    'S': tf.placeholder(tf.int32, shape=[None, None]),
    'batch_labels': tf.placeholder(tf.int32, shape=[None, label_dim])
}

# construct batch generator
minibatch = CrossModalMinibatchIteratorAdaptive(A=adj,
                                        train_num=train_num,
                                        placeholders=placeholders,
                                        train_labels=train_labels,
                                        batch_size=FLAGS.batch_size,
                                        max_degree=FLAGS.max_degree,
                                        percentage=FLAGS.percentage,
                                        dataset=FLAGS.dataset)
adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

sampler = UniformNeighborSampler(adj_info)
layer_infos_img = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.img_gc1),
                   SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.siamese_bit)]
layer_infos_txt = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.txt_gc1),
                   SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.siamese_bit)]

# Create model
img_model = ImgModel(placeholders=placeholders,
                     features=img_feature,
                     adj=adj_info,
                     degrees=minibatch.deg,
                     layer_infos=layer_infos_img,
                     aggregator_type="maxpool",
                     logging=True)
txt_model = TxtModel(placeholders=placeholders,
                     features=txt_feature,
                     adj=adj_info,
                     degrees=minibatch.deg,
                     layer_infos=layer_infos_txt,
                     aggregator_type="maxpool",
                     logging=True)
emb_v = img_model.outputs_img
emb_w = txt_model.outputs_txt

# label classifier network
predict_v = label_classifier(emb_v, label_dim)
predict_w = label_classifier(emb_w, label_dim, reuse=True)
predict_v_softmax = tf.nn.softmax(predict_v)
predict_w_softmax = tf.nn.softmax(predict_w)

#siamese network
emb_v, hash_emb_v = siamese_net(emb_v)
emb_w, hash_emb_w = siamese_net(emb_w, reuse=True)

'''
    --------------------loss------------------
'''

'''
    label predict loss
'''
gamma = pow(10, -14)
label_loss = tf.nn.softmax_cross_entropy_with_logits(labels=placeholders['batch_labels'], logits=predict_v) + \
            tf.nn.softmax_cross_entropy_with_logits(labels=placeholders['batch_labels'], logits=predict_w)
label_loss = tf.reduce_sum(label_loss)
label_loss = label_loss * gamma

'''
    domain adversarial
'''
emb_v_class = domain_adversarial(emb_v)
emb_w_class = domain_adversarial(emb_w, reuse=True)
all_emb_v = tf.placeholder(tf.float32, shape=(None, 2))
all_emb_w = tf.placeholder(tf.float32, shape=(None, 2))
# discriminator loss
domain_class_loss = tf.nn.softmax_cross_entropy_with_logits(logits=emb_v_class, labels=all_emb_v) + \
            tf.nn.softmax_cross_entropy_with_logits(logits=emb_w_class, labels=all_emb_w)
domain_class_loss = tf.reduce_mean(domain_class_loss)
# # generator loss
domain_class_loss_fake = tf.nn.softmax_cross_entropy_with_logits(logits=emb_v_class, labels=all_emb_w) + \
            tf.nn.softmax_cross_entropy_with_logits(logits=emb_w_class, labels=all_emb_v)
domain_class_loss_fake = tf.reduce_mean(domain_class_loss_fake)
# alpha = 1000000
alpha = 1
# domain_class_loss = domain_class_loss * alpha # do not matter
domain_class_loss_fake = domain_class_loss_fake * alpha

# discriminator accuracy
domain_img_class_acc = tf.equal(tf.greater(emb_v_class, 0.5), tf.greater(all_emb_v, 0.5))
domain_txt_class_acc = tf.equal(tf.greater(0.5, emb_w_class), tf.greater(0.5, all_emb_w))
domain_class_acc = tf.reduce_mean(tf.cast(tf.concat([domain_img_class_acc, domain_txt_class_acc], axis=0), tf.float32))

'''
    similarity preserving loss
'''
# img loss
theta_v = 1.0 / 2 * tf.matmul(emb_v, tf.transpose(emb_w))
likely_v = tf.multiply(tf.cast(placeholders['S'], dtype=tf.float32), theta_v) - tf.log(1.0 + tf.exp(theta_v))
neg_likely_v = -1 * tf.reduce_sum(likely_v)

#loss using hash code
hash_theta_v = 1.0 / 2 * tf.matmul(hash_emb_v, tf.transpose(hash_emb_w))
hash_likely_v = tf.multiply(tf.cast(placeholders['S'], dtype=tf.float32), hash_theta_v) - tf.log(1.0 + tf.exp(hash_theta_v))
hash_neg_likely_v = -1 * tf.reduce_sum(hash_likely_v)

# # txt loss
theta_w = 1.0 / 2 * tf.matmul(emb_w, tf.transpose(emb_v))
likely_w = tf.multiply(tf.cast(placeholders['S'], dtype=tf.float32), theta_w) - tf.log(1.0 + tf.exp(theta_w))
neg_likely_w = -1 * tf.reduce_sum(likely_w)

# loss using hash code
hash_theta_w = 1.0 / 2 * tf.matmul(hash_emb_w, tf.transpose(hash_emb_v))
hash_likely_w = tf.multiply(tf.cast(placeholders['S'], dtype=tf.float32), hash_theta_w) - tf.log(1.0 + tf.exp(hash_theta_w))
hash_neg_likely_w = -1 * tf.reduce_sum(hash_likely_w)

'''
    quantization loss
'''
quantization_img = 2 * tf.nn.l2_loss(emb_v - hash_emb_v)
quantization_txt = 2 * tf.nn.l2_loss(emb_w - hash_emb_w)

'''
    balance loss
'''
ones = tf.placeholder(shape=(None, 1), dtype=tf.float32)
alpha = 0.001
img_balance = alpha * tf.reduce_sum(tf.pow(tf.matmul(tf.transpose(emb_w), ones), 2))
txt_balance = alpha * tf.reduce_sum(tf.pow(tf.matmul(tf.transpose(emb_v), ones), 2))
balance_loss = img_balance + txt_balance

'''
    equality loss
'''
beta = 1.0
equality_loss = tf.nn.l2_loss(emb_v - emb_w)
equality_loss = beta * equality_loss

'''
    total loss
'''
img_loss = neg_likely_v
txt_loss = neg_likely_w
total_loss = img_loss + txt_loss + equality_loss + balance_loss + label_loss

'''
    optimizer
'''
# label classifier optimizer
# lc_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="label_classifier")
# classifier_vars = [var for var in lc_variables]

t_vars = tf.trainable_variables()
dc_vars = [v for v in t_vars if 'dc_' in v.name]
siamese_vars = [v for v in t_vars if 'siamese_fc_' in v.name]
classifier_vars = [v for v in t_vars if 'lc_fc_' in v.name]
# img optimizer
img_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_img)
img_vars = img_model.vars
img_op = img_optimizer.minimize(img_loss, var_list=img_vars)
#
# # txt optimizer
txt_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_txt)
txt_vars = txt_model.vars
txt_op = txt_optimizer.minimize(txt_loss, var_list=txt_vars)

# # label classifier optimizer
lc_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_label)
lc_op = lc_optimizer.minimize(label_loss, var_list=classifier_vars)

# # domain optimizer
dc_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_domain)
dc_op = dc_optimizer.minimize(domain_class_loss, var_list=dc_vars)

# combine optimize
total_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr_img)
total_op = total_optimizer.minimize(total_loss, var_list=img_vars+txt_vars+siamese_vars)


print('optimization variables:')
print('img parameters:')
print(img_vars)
print('txt parameters:')
print(txt_vars)
print('siamese parameters:')
print(siamese_vars)
print('classifier parameters:')
print(classifier_vars)

saver = tf.train.Saver()
train_adj_info = tf.assign(adj_info, minibatch.adj)
val_adj_info = tf.assign(adj_info, minibatch.test_adj)
if FLAGS.train:
    # Initialize session
    sess = tf.Session()

    # Init variables
    sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})

    cost_val = []
    # plt loss curve
    sim_loss_img = []
    hash_sim_img = []
    q_loss_img = []
    sim_loss_txt = []
    q_loss_txt = []
    hash_sim_txt = []

    max_map = 0
    # Train model
    for epoch in range(FLAGS.epochs):
        t = time.time()
        minibatch.shuffle()
        iter = 0
        while not minibatch.end():
            iter += 1
            feed_dict, batch_nodes = minibatch.next_minibatch_feed_dict()
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            size = len(batch_nodes)
            S = np.zeros((size, size))
            batch_labels = np.zeros((size, label_dim))
            for rowid in range(size):
                for colid in range(size):
                    S[rowid, colid] = adj[batch_nodes[rowid], batch_nodes[colid]]
            for i in range(size):
                batch_labels[i] = labels[batch_nodes[i]]
            feed_dict.update({placeholders['batch_labels']: batch_labels})
            temp_one = np.ones((size, 1))
            temp_zero = np.zeros((size, 1))
            img_domain = np.concatenate((temp_one, temp_zero), axis=1)
            txt_domain = np.concatenate((temp_zero, temp_one), axis=1)
            feed_dict.update({placeholders['S']: S})
            feed_dict.update({ones: temp_one})
            feed_dict.update({all_emb_v: img_domain})
            feed_dict.update({all_emb_w: txt_domain})

            '''
                combine optimization without domain classifier, but with a label classifier
            '''
            op_, lc_ = sess.run([total_op, lc_op], feed_dict=feed_dict)
            likely_loss_v, hash_likely_loss_v, q_img, temp_emb_v, temp_balance_img = sess.run([neg_likely_v, hash_neg_likely_v, quantization_img, emb_v, img_balance], feed_dict=feed_dict)
            likely_loss_w, hash_likely_loss_w, q_txt, temp_emb_w, temp_balance_txt = sess.run([neg_likely_w, hash_neg_likely_w, quantization_txt, emb_w, txt_balance], feed_dict=feed_dict)
            equality_loss_, label_loss_ = sess.run([equality_loss, label_loss], feed_dict=feed_dict)

            sim_loss_img.append(likely_loss_v)
            hash_sim_img.append(hash_likely_loss_v)
            q_loss_img.append(q_img)
            sim_loss_txt.append(likely_loss_w)
            q_loss_txt.append(q_txt)
            hash_sim_txt.append(hash_likely_loss_w)

            if iter % 10 == 0:
                '''
                        print result without domain classifier
                '''
                print("Epoch:", '%04d' % (epoch + 1),
                      "neg_v:", "{:.5f}".format(likely_loss_v),
                      "quantization_img:", "{:.5f}".format(q_img),
                      "balance_img:", "{:.5f}".format(temp_balance_img),
                      "neg_w:", "{:.5f}".format(likely_loss_w),
                      "quantization_txt:", "{:.5f}".format(q_txt),
                      "balance_txt:", "{:.5f}".format(temp_balance_txt),
                      "equality_loss", '{:.5f}'.format(equality_loss_),
                      "label_loss", '{:.5f}'.format(label_loss_),
                      "time=", "{:.5f}".format(time.time() - t))

            '''
            evaluation per epoch
            '''

        '''
            adaptively change the graph according to confident label prediction
        '''
        if epoch % 10 == 0:
            sess.run(val_adj_info.op)
            confident_nodes_v, confident_labels_v = predict_unlabeled(sess, predict_v_softmax, minibatch, FLAGS.batch_size, label_dim, placeholders)
            confident_nodes_w, confident_labels_w = predict_unlabeled(sess, predict_w_softmax, minibatch, FLAGS.batch_size, label_dim, placeholders)
            confident_nodes = [node for node in confident_nodes_v if node in confident_nodes_w]
            if len(confident_nodes) > 0:
                confident_labels = np.zeros(shape=(1, label_dim))
                for node in confident_nodes:
                    confident_labels = np.vstack((confident_labels, confident_labels_v[node]))
                confident_labels = confident_labels[1:]
                minibatch.update_graph(confident_nodes, confident_labels)
            print('-----------update {} nodes in epoch {}!-----------'.format(len(confident_nodes), epoch))
            minibatch.show_info()
            sess.run(train_adj_info.op)

        if epoch % 20 == 0:
            sess.run(val_adj_info.op)
            img_feature_trans = incremental_evaluate(sess, emb_v, minibatch, FLAGS.batch_size, FLAGS.hash_bit, placeholders)
            txt_feature_trans = incremental_evaluate(sess, emb_w, minibatch, FLAGS.batch_size, FLAGS.hash_bit, placeholders)
            img_class_acc = incremental_evaluate_label_unlabeled(sess, predict_v_softmax, minibatch, FLAGS.batch_size, unlabeled_labels, placeholders)
            txt_class_acc = incremental_evaluate_label_unlabeled(sess, predict_w_softmax, minibatch, FLAGS.batch_size, unlabeled_labels, placeholders)
            img_class_acc_labeled = incremental_evaluate_label_labeled(sess, predict_v, minibatch, FLAGS.batch_size, labeled_labels, placeholders)
            txt_class_acc_labeled = incremental_evaluate_label_labeled(sess, predict_w, minibatch, FLAGS.batch_size, labeled_labels, placeholders)
            print('unlabeled: img label acc :{}, txt label acc:{}'.format(img_class_acc, txt_class_acc))
            print('labeled: img label acc :{}, txt label acc:{}'.format(img_class_acc_labeled, txt_class_acc_labeled))
            test_img_feats_trans = img_feature_trans[train_num:]
            test_txt_vecs_trans = txt_feature_trans[train_num:]
            train_img_feats_trans = img_feature_trans[:train_num]
            train_txt_vecs_trans = txt_feature_trans[:train_num]
        #     # binary code
            test_img_feats_trans_binary = np.sign(test_img_feats_trans)
            test_txt_vecs_trans_binary = np.sign(test_txt_vecs_trans)
            train_img_feats_trans_binary = np.sign(train_img_feats_trans)
            train_txt_vecs_trans_binary = np.sign(train_txt_vecs_trans)
            print("--------------binary map:---------------")
            calculate_map(test_img_feats_trans_binary, test_txt_vecs_trans_binary, test_labels)
            mapi2t = calc_map(test_img_feats_trans_binary, train_txt_vecs_trans_binary, test_labels, train_labels)
            mapt2i = calc_map(test_txt_vecs_trans_binary, train_img_feats_trans_binary, test_labels, train_labels)
            print('...test map: map(i->t): %3.3f, map(t->i): %3.3f' % (mapi2t, mapt2i))
            print("--------------continuous map------------")
            calculate_map(test_img_feats_trans, test_txt_vecs_trans, test_labels)
            c_mapi2t = calc_map(test_img_feats_trans, train_txt_vecs_trans, test_labels, train_labels)
            c_mapt2i = calc_map(test_txt_vecs_trans, train_img_feats_trans, test_labels, train_labels)
            print('...test map: map(i->t): %3.3f, map(t->i): %3.3f' % (c_mapi2t, c_mapt2i))
            sess.run(train_adj_info.op)

        if epoch == FLAGS.epochs-1:
            print("img_emb:{}\n".format(temp_emb_v))
            print("txt_emb:{}\n".format(temp_emb_w))
    print("Optimization Finished!")


    # # Testing
    sess.run(val_adj_info.op)
    img_feature_trans = incremental_evaluate(sess, emb_v, minibatch, FLAGS.batch_size, FLAGS.hash_bit, placeholders)
    txt_feature_trans = incremental_evaluate(sess, emb_w, minibatch, FLAGS.batch_size, FLAGS.hash_bit, placeholders)
    test_img_feats_trans = img_feature_trans[train_num:]
    test_txt_vecs_trans = txt_feature_trans[train_num:]
    train_img_feats_trans = img_feature_trans[:train_num]
    train_txt_vecs_trans = txt_feature_trans[:train_num]
    # binary code
    test_img_feats_trans_binary = np.sign(test_img_feats_trans)
    test_txt_vecs_trans_binary = np.sign(test_txt_vecs_trans)
    train_img_feats_trans_binary = np.sign(train_img_feats_trans)
    train_txt_vecs_trans_binary = np.sign(train_txt_vecs_trans)

    '''
        save model parameters
    '''
    check_point_dir = './data/' + FLAGS.dataset + '/model/' + 'temp.ckpt'
    saver.save(sess, check_point_dir)
else:
    save_path = './data/' + FLAGS.dataset + '/model/temp.ckpt'
    with tf.Session() as sess:
        saver.restore(sess, save_path)
        sess.run(val_adj_info.op)
        img_feature_trans = incremental_evaluate(sess, emb_v, minibatch, FLAGS.batch_size, FLAGS.hash_bit, placeholders)
        txt_feature_trans = incremental_evaluate(sess, emb_w, minibatch, FLAGS.batch_size, FLAGS.hash_bit, placeholders)
        test_img_feats_trans = img_feature_trans[train_num:]
        test_txt_vecs_trans = txt_feature_trans[train_num:]
        train_img_feats_trans = img_feature_trans[:train_num]
        train_txt_vecs_trans = txt_feature_trans[:train_num]
        # binary code
        test_img_feats_trans_binary = np.sign(test_img_feats_trans)
        test_txt_vecs_trans_binary = np.sign(test_txt_vecs_trans)
        train_img_feats_trans_binary = np.sign(train_img_feats_trans)
        train_txt_vecs_trans_binary = np.sign(train_txt_vecs_trans)

'''
    save embedding
'''
# with open('./data/'+FLAGS.dataset+'/embedding/'+str(FLAGS.hash_bit)+'_img_binary.pkl', 'w') as f:
#     cPickle.dump(train_img_feats_trans_binary, f)
# with open('./data/'+FLAGS.dataset+'/embedding/'+str(FLAGS.hash_bit)+'_txt_binary.pkl', 'w') as f:
#     cPickle.dump(train_txt_vecs_trans_binary, f)
# with open('./data/'+FLAGS.dataset+'/embedding/'+str(FLAGS.hash_bit)+'_img_float.pkl', 'w') as f:
#     cPickle.dump(train_img_feats_trans, f)
# with open('./data/'+FLAGS.dataset+'/embedding/'+str(FLAGS.hash_bit)+'_txt_float.pkl', 'w') as f:
#     cPickle.dump(train_txt_vecs_trans, f)
#

print("--------------binary map:---------------")
calculate_map(test_img_feats_trans_binary, test_txt_vecs_trans_binary, test_labels)
mapi2t = calc_map(test_img_feats_trans_binary, train_txt_vecs_trans_binary, test_labels, train_labels)
mapt2i = calc_map(test_txt_vecs_trans_binary, train_img_feats_trans_binary, test_labels, train_labels)
print ('...test map: map(i->t): %3.3f, map(t->i): %3.3f' % (mapi2t, mapt2i))
print("--------------continuous map------------")
calculate_map(test_img_feats_trans, test_txt_vecs_trans, test_labels)
c_mapi2t = calc_map(test_img_feats_trans, train_txt_vecs_trans, test_labels, train_labels)
c_mapt2i = calc_map(test_txt_vecs_trans, train_img_feats_trans, test_labels, train_labels)
print ('...test map: map(i->t): %3.3f, map(t->i): %3.3f' % (c_mapi2t, c_mapt2i))

precision_i2t, recall_i2t = calc_precision_recall(test_img_feats_trans_binary, train_txt_vecs_trans_binary, test_labels, train_labels)
precision_t2i, recall_t2i = calc_precision_recall(test_txt_vecs_trans_binary, train_img_feats_trans_binary, test_labels, train_labels)
print('precision_t2i')
print(precision_t2i)
print('recall_t2i')
print(recall_t2i)
print('precision_i2t')
print(precision_i2t)
print('recall_i2t')
print(recall_i2t)


# result_save_path = './data/'+FLAGS.dataset+'/'+'result/'+datetime.now().strftime("%d-%h-%m-%s")+'_'+str(FLAGS.hash_bit)+'_bit_result.pkl'
result_save_path = './data/'+FLAGS.dataset+'/'+'result/'+'semi_'+str(FLAGS.hash_bit)+'_'+str(FLAGS.percentage)+'_bit_adaptive.pkl'
result = dict()
result['desc'] = 'no quantization'
result['bit'] = FLAGS.hash_bit
result['epoch'] = FLAGS.epochs
result['lr'] = FLAGS.lr_img
result['mapi2t'] = mapi2t
result['mapt2i'] = mapt2i
result['c_mapi2t'] = c_mapi2t
result['c_mapt2i'] = c_mapt2i
result['precision_i2t'] = precision_i2t
result['precision_t2i'] = precision_t2i
result['recall_i2t'] = recall_i2t
result['recall_t2i'] = recall_t2i
with open(result_save_path, 'w') as f:
    cPickle.dump(result, f)

'''
    plot PR-curve
'''
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title('text2img@' + str(FLAGS.hash_bit) + '-bit')
plt.xlabel('recall')
plt.ylabel('precision')
plt.axis([0, 1, 0.2, 1])
plt.grid(True)
plotRPcurve(precision_t2i, recall_t2i, 'r-s', 'GCN')
plt.subplot(122)
plt.title('img2text@' + str(FLAGS.hash_bit) + '-bit')
plt.xlabel('recall')
plt.ylabel('precision')
plt.axis([0, 1, 0.2, 1])
plt.grid(True)
plotRPcurve(precision_i2t, recall_i2t, 'r-s', 'GCN')
# plt.savefig('fig-'+str(FLAGS.hash_bit)+'.png')
plt.show()

# '''
#     save model parameters
# '''
# check_point_dir = './data/' + FLAGS.dataset + '/model/' + 'temp.ckpt'
# saver.save(sess, check_point_dir)

'''
    plot loss curve
'''
#img_plot_save_path = './figure/img_'+str(FLAGS.epochs)+'_'+str(FLAGS.lr_img)+'_'+str(FLAGS.hash_bit)+'.png'
#txt_plot_save_path = './figure/txt_'+str(FLAGS.epochs)+'_'+str(FLAGS.lr_txt)+'_'+str(FLAGS.hash_bit)+'.png'
#x_index = np.linspace(1, FLAGS.epochs, FLAGS.epochs)
#plt.figure(num='img')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.plot(x_index, sim_loss_img, 'r-s', label='sim_loss')
#plt.plot(x_index, q_loss_img, 'g-s', label='quantization')
#plt.plot(x_index, hash_sim_img, 'b-s', label='hash_sim_loss')
#plt.legend(loc='upper right')
#plt.savefig(img_plot_save_path)
#plt.show()

#plt.figure(num='txt')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.plot(x_index, sim_loss_txt, 'r-s', label='sim_loss')
#plt.plot(x_index, q_loss_txt, 'g-s', label='quantization')
#plt.plot(x_index, hash_sim_txt, 'b-s', label='hash_sim_loss')
#plt.legend(loc='upper right')
#plt.savefig(txt_plot_save_path)
#plt.show()
