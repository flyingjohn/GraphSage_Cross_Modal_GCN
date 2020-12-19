import graphsage.layers as layers
import graphsage.models as models
from graphsage.inits import *
from graphsage.aggregators import MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator, GCNAggregator
import tensorflow.contrib.slim as slim
import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS


# def label_classifier(inputs, label_num):
#     with tf.variable_scope("label_classifier", reuse=True):
#         weights = glorot([FLAGS.hash_bit, label_num], name='lc_weights')
#         bias = zeros([label_num], name='lc_bias')
#         predict = tf.matmul(inputs, weights) + bias
#     return predict


def label_classifier(inputs, label_num, reuse=False):
    with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
        net = slim.fully_connected(inputs, FLAGS.siamese_bit_1, scope='lc_fc_0')
        net = slim.fully_connected(net, label_num, scope='lc_fc_1')
    return net


def img_label_classifier(inputs, label_num, reuse=False):
    with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
        net = slim.fully_connected(inputs, FLAGS.siamese_bit_1, scope='ilc_fc_0')
        net = slim.fully_connected(net, label_num, scope='ilc_fc_1')
    return net


def txt_label_classifier(inputs, label_num, reuse=False):
    with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
        net = slim.fully_connected(inputs, FLAGS.siamese_bit_1, scope='tlc_fc_0')
        net = slim.fully_connected(net, label_num, scope='tlc_fc_1')
    return net


def siamese_net_single(inputs, reuse=False):
    with tf.variable_scope("siamese_net_single", reuse=reuse):
        weights = glorot([FLAGS.siamese_bit, FLAGS.hash_bit], name='sia_weights')
        bias = zeros([FLAGS.hash_bit], name='sia_bias')
        representation = tf.nn.tanh(tf.matmul(inputs, weights) + bias)
        hash_represetation = tf.sign(representation)
    return representation, hash_represetation


def siamese_net(inputs, is_training=True, reuse=False):
    with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.tanh, reuse=reuse):
        net = slim.fully_connected(inputs, FLAGS.siamese_bit_1, scope='siamese_fc_0')
        net = slim.fully_connected(net, FLAGS.hash_bit, scope='siamese_fc_1')
        hash_code = tf.sign(net)
    return net, hash_code


def domain_adversarial(inputs, is_training=True, reuse=False):
    with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
        net = slim.fully_connected(inputs, FLAGS.hash_bit/2, scope='dc_fc_0')
        net = slim.fully_connected(net, FLAGS.hash_bit/4, scope='dc_fc_1')
        net = slim.fully_connected(net, 2, scope='dc_fc_2')
    return net


class ImgModel(models.SampleAndAggregate):
    """
        image model
        modified from SupervisedGraphsage model.
    """

    def __init__(self, placeholders, features, adj, degrees,
            layer_infos, concat=True, aggregator_type="mean",
            model_size="small", identity_dim=0,
            **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features.
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees.
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all
                   the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - sigmoid_loss: Set to true if nodes can belong to multiple classes
        '''

        models.GeneralizedModel.__init__(self, **kwargs)

        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)

        # get info from placeholders...
        self.inputs_img = placeholders["batch"]
        self.model_size = model_size
        self.adj_info = adj
        if identity_dim > 0:
           self.embeds = tf.get_variable("node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
        else:
           self.embeds = None
        if features is None:
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.features = self.embeds
        else:
            self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
            if not self.embeds is None:
                self.features = tf.concat([self.embeds, self.features], axis=1)
        self.degrees = degrees
        self.concat = concat
        self.dims = [(0 if features is None else features.shape[1]) + identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.batch_size = placeholders["batch_size"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos

        # self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        with tf.variable_scope(self.name):
            self.build()
        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = [var for var in variables]

    def build(self):
        samples_img, support_sizes = self.sample(self.inputs_img, self.layer_infos)
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
        self.outputs_img, self.aggregators = self.aggregate(samples_img, [self.features],
                                                            self.dims, num_samples,
                                                            support_sizes, concat=self.concat,
                                                            model_size=self.model_size)
        # self.outputs_img = tf.nn.l2_normalize(self.outputs_img, 1)


    def _regular_loss(self):
        # Weight decay loss
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.regular_loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.node_pred.vars.values():
            self.regular_loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        tf.summary.scalar('loss', self.loss)


class TxtModel(models.SampleAndAggregate):
    """
        txt model
        modified from SupervisedGraphsage model.
    """

    def __init__(self, placeholders, features, adj, degrees,
            layer_infos, concat=True, aggregator_type="mean",
            model_size="small", identity_dim=0,
            **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features.
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees.
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all
                   the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - sigmoid_loss: Set to true if nodes can belong to multiple classes
        '''

        models.GeneralizedModel.__init__(self, **kwargs)

        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)

        # get info from placeholders...
        self.inputs_txt = placeholders["batch"]
        self.model_size = model_size
        self.adj_info = adj
        if identity_dim > 0:
           self.embeds = tf.get_variable("node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
        else:
           self.embeds = None
        if features is None:
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.features = self.embeds
        else:
            self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
            if not self.embeds is None:
                self.features = tf.concat([self.embeds, self.features], axis=1)
        self.degrees = degrees
        self.concat = concat
        self.dims = [(0 if features is None else features.shape[1]) + identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.batch_size = placeholders["batch_size"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos

        # self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        with tf.variable_scope(self.name):
            self.build()
        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = [var for var in variables]


    def build(self):
        samples_txt, support_sizes = self.sample(self.inputs_txt, self.layer_infos)
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
        self.outputs_txt, self.aggregators = self.aggregate(samples_txt, [self.features],
                                                            self.dims, num_samples,
                                                            support_sizes, concat=self.concat,
                                                            model_size=self.model_size)
        # self.outputs_img = tf.nn.l2_normalize(self.outputs_img, 1)


    def _regular_loss(self):
        # Weight decay loss
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.regular_loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.node_pred.vars.values():
            self.regular_loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        tf.summary.scalar('loss', self.loss)


