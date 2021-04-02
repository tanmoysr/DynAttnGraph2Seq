import tensorflow as tf
from tensorflow.python.layers.core import Dense
import tensorflow.contrib.seq2seq as seq2seq
from neigh_samplers import UniformNeighborSampler
from aggregators import MeanAggregator, MaxPoolingAggregator, GatedMeanAggregator
import numpy as np
import match_utils
import configure as conf_file  # Do not use name 'conf'. It will create confusion. Used for Post Encoder
if conf_file.post_encoder:
    import post_encoder_model as model_enhancement # Post Encoder
import time  # edited by T
import tensorflow_hub as hub

class Graph2SeqNN(object):

    PAD = 0
    GO = 1
    EOS = 2

    def __init__(self, mode, conf, path_embed_method):

        self.mode = mode
        self.word_vocab_size = conf.word_vocab_size
        self.l2_lambda = conf.l2_lambda
        self.path_embed_method = path_embed_method
        self.word_embedding_dim = conf.word_embedding_dim
        self.timestamp = conf.timestamp
        self.node_feature_format = conf.node_feature_format
        self.test_batch_size = conf.test_batch_size

        # the setting for the GCN
        self.graph_encode_direction = conf.graph_encode_direction
        self.sample_layer_size = conf.sample_layer_size
        self.hidden_layer_dim = conf.hidden_layer_dim
        self.concat = conf.concat

        # the setting for node level attention
        self.node_attention_layer = {
            'weights': tf.Variable(tf.random_normal([4 * self.hidden_layer_dim, 1])),
            'biases': tf.Variable(tf.random_normal([1]))}

        # the setting for the dynGraph encoder
        self.dyn_num_layers = conf.dyn_num_layers
        self.dyn_graph_emb_dim = conf.dyn_graph_emb_dim

        # the setting for the decoder
        self.beam_width = conf.beam_width
        self.decoder_type = conf.decoder_type
        self.seq_max_len = conf.seq_max_len
        self.feature_max_len = conf.feature_max_len

        # setting for multi-task learning
        self.beta = conf.beta
        self.window_size = conf.window_size
        self.stride = conf.stride

        self._text = tf.placeholder(tf.int32, [None, None])
        self.decoder_seq_length = tf.placeholder(tf.int32, [None])
        self.loss_weights = tf.placeholder(tf.float32, [None, None])

        # the following place holders are for the gcn
        self.fw_adj_info = tf.placeholder(tf.int32, [None, None, None])               # the fw adj info for each node
        self.bw_adj_info = tf.placeholder(tf.int32, [None, None, None])               # the bw adj info for each node
        if conf_file.um_specific:
            self.feature_info = tf.placeholder(tf.int32, [None, None, None])
        else:
            self.feature_info = tf.placeholder(tf.int32, [None, None])              # the feature info for each node
        self.batch_nodes = tf.placeholder(tf.int32, [None, None])               # the nodes for each batch

        self.sample_size_per_layer = tf.shape(self.fw_adj_info)[2]

        self.single_graph_nodes_size = tf.shape(self.batch_nodes)[1]
        self.attention = conf.attention
        self.hierarchical_attention = conf.hierarchical_attention
        self.attention_method = conf.attention_method
        self.dropout = conf.dropout
        self.fw_aggregators = []
        self.bw_aggregators = []

        self.if_pred_on_dev = False

        self.learning_rate = conf.learning_rate

        if conf_file.post_encoder:
            ##-------------- For Post Encoder (start)---------------- ##
            # parameter for second encoder
            self.post_encoder_output = tf.placeholder(tf.float32, [None, None, None])  # Edited by TC
            self.post_encoder_states_c = tf.placeholder(tf.float32, [None, None])  # Edited by TC
            self.post_encoder_states_h = tf.placeholder(tf.float32, [None, None])  # Edited by TC
            self.month_vect_inputs = tf.placeholder(tf.float32, [None, None, None], name='input_data1')
            #         encoding_embedding_size = tf.placeholder(tf.int32, name='encoding_embedding_size')

            # Network parameter
            self.rnn_size = tf.constant(conf_file.post_encoder_batch_dim, name='rnn_size')
            self.num_layers = tf.constant(3, name='num_layers')
            self.sequence_length = None  # will work on it later
            self.keep_prob = tf.constant(0.5, name='keep_prob')
            self.reshape_axis11 = tf.placeholder(tf.int32, name='reshape_axis11')
            self.reshape_axis12 = tf.placeholder(tf.int32, name='reshape_axis12')

            self.post_vect_inputs = tf.placeholder(tf.float32, [None, None, None], name='input_data')
            self.rnn_size2 = tf.constant(conf_file.post_encoder_batch_dim, name='rnn_size2')
            self.num_layers2 = tf.constant(3, name='num_layers2')
            self.sequence_length2 = None  # will work on it later
            self.keep_prob2 = tf.constant(conf_file.post_encoder_keep_prob, name='keep_prob2')
            self.reshape_axis2 = tf.placeholder(tf.int32, name='reshape_axis2')

            self.encoder1_output = model_enhancement.month_encoder(self.month_vect_inputs, self.num_layers, self.rnn_size,
                                                                   self.sequence_length, self.keep_prob,
                                                                   self.reshape_axis11, self.reshape_axis12)
            self.encoder2_output = model_enhancement.user_encoder(self.post_vect_inputs, self.num_layers2, self.rnn_size2,
                                                                  self.sequence_length2, self.keep_prob2,
                                                                  self.reshape_axis2)
            # Bert Model
            self.input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None])
            self.input_mask = tf.placeholder(dtype=tf.int32, shape=[None, None])
            self.segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None])

            bert_inputs = dict(
                input_ids=self.input_ids,
                input_mask=self.input_mask,
                segment_ids=self.segment_ids)
            BERT_URL = 'https://tfhub.dev/google/bert_cased_L-12_H-768_A-12/1'
            bert_module = hub.Module(BERT_URL)

            self.bert_outputs = bert_module(bert_inputs, signature="tokens", as_dict=True)

            ##-------------- For Post Encoder (end)---------------- ##

    def _init_decoder_train_connectors(self):
        batch_size, sequence_size = tf.unstack(tf.shape(self._text))
        self.batch_size = batch_size
        GO_SLICE = tf.ones([batch_size, 1], dtype=tf.int32) * self.GO
        EOS_SLICE = tf.ones([batch_size, 1], dtype=tf.int32) * self.PAD
        self.decoder_train_inputs = tf.concat([GO_SLICE, self._text], axis=1)
        self.decoder_train_length = self.decoder_seq_length + 1
        decoder_train_targets = tf.concat([self._text, EOS_SLICE], axis=1)
        _, decoder_train_targets_seq_len = tf.unstack(tf.shape(decoder_train_targets))
        decoder_train_targets_eos_mask = tf.one_hot(self.decoder_train_length - 1, decoder_train_targets_seq_len,
                                                    on_value=self.EOS, off_value=self.PAD, dtype=tf.int32)
        self.decoder_train_targets = tf.add(decoder_train_targets, decoder_train_targets_eos_mask)
        self.decoder_train_inputs_embedded = tf.nn.embedding_lookup(self.word_embeddings, self.decoder_train_inputs)


    def encode(self):
        with tf.variable_scope("embedding_layer"):
            pad_word_embedding = tf.zeros([1, self.word_embedding_dim])  # this is for the PAD symbol
            self.word_embeddings = tf.concat([pad_word_embedding,
                                              tf.get_variable('W_train', shape=[self.word_vocab_size,self.word_embedding_dim],
                                                                            initializer=tf.contrib.layers.xavier_initializer(), trainable=True)], 0)

        with tf.variable_scope("graph_encoding_layer"):

            # self.encoder_outputs, self.encoder_state = self.gcn_encode()

            # this is for optimizing gcn
            self.node_attention=[]

            if conf_file.g2s:
                # print('g2s is working')
                node_embedding, graph_embedding = self.optimized_gcn_encode(self.timestamp - 1)
                node_embedding_seq = tf.expand_dims(node_embedding, 0)
                graph_embedding_seq = tf.expand_dims(graph_embedding, 0)
                dyn_cell = self._build_encoder_cell(self.dyn_num_layers, self.dyn_graph_emb_dim)
                # print(tf.shape(node_embedding)[0])
                if conf_file.post_encoder:
                    source_sequence_length = tf.reshape(
                        tf.ones([tf.shape(node_embedding)[0], 1], dtype=tf.int32) * 1,
                        (tf.shape(node_embedding)[0],))
                    # source_sequence_length = tf.reshape(tf.ones([tf.shape(node_embedding)[0], 1], dtype=tf.int32) * (1+conf_file.timestamp),
                    #                                     (tf.shape(node_embedding)[0],))
                else:
                    source_sequence_length = tf.reshape(tf.ones([tf.shape(node_embedding)[0], 1], dtype=tf.int32) * 1,
                    (tf.shape(node_embedding)[0],))
            else:
                for t in range(0, self.timestamp):
                    node_embedding, graph_embedding = self.optimized_gcn_encode(t)

                    if t == 0:
                        node_embedding_seq = tf.expand_dims(node_embedding, 0)
                        graph_embedding_seq = tf.expand_dims(graph_embedding, 0)
                    else:
                        node_embedding_seq = tf.concat([node_embedding_seq, tf.expand_dims(node_embedding, 0)], 0)
                        graph_embedding_seq = tf.concat([graph_embedding_seq, tf.expand_dims(graph_embedding, 0)], 0)
                    node_embedding_seq = tf.add(node_embedding_seq, node_embedding)
                    graph_embedding_seq = tf.add(graph_embedding_seq, graph_embedding)

                graph_embedding_seq = tf.math.scalar_mul(1.0/t, graph_embedding_seq)
                node_embedding_seq = tf.math.scalar_mul(1.0/t, node_embedding_seq)

                # dynamic graph embedding encoder
                dyn_cell = self._build_encoder_cell(self.dyn_num_layers, self.dyn_graph_emb_dim)
                if conf_file.post_encoder:
                    source_sequence_length = tf.reshape(
                        tf.ones([tf.shape(node_embedding)[0], 1], dtype=tf.int32) * self.timestamp,
                        (tf.shape(node_embedding)[0],))
                else:
                    source_sequence_length = tf.reshape(
                    tf.ones([tf.shape(node_embedding)[0], 1], dtype=tf.int32) * self.timestamp,
                    (tf.shape(node_embedding)[0],))


            # if conf_file.post_encoder:
            #     ##-----------Post encoder (start)--------------
            #     source_sequence_length = tf.reshape(
            #         tf.ones([tf.shape(node_embedding)[0], 1], dtype=tf.int32) * self.timestamp * 2,
            #         (tf.shape(node_embedding)[0],))
            #     ##------------Post encoder (end)-----------------
            # else:
            #     source_sequence_length = tf.reshape(
            #         tf.ones([tf.shape(node_embedding)[0], 1], dtype=tf.int32) * self.timestamp,
            #         (tf.shape(node_embedding)[0],))



            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                dyn_cell, graph_embedding_seq, dtype=tf.float32,
                sequence_length=source_sequence_length, time_major=True)


            if conf_file.post_encoder:
                encoder_outputs2 = encoder_outputs
                # encoder_outputs2 = tf.concat([encoder_outputs, self.post_encoder_output], 0)
                # encoder_state_c = tf.concat([encoder_state.c, self.post_encoder_states_c], 0)
                # encoder_state_h = tf.concat([encoder_state.h, self.post_encoder_states_h], 0)
                # encoder_state = tf.nn.rnn_cell.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
                # ncoder_state_c = tf.concat([encoder_state.c, self.post_encoder_states_c], 1)

                # -----------------------Concatenate with neural layers starts----------------------------
                encoder_state_c = tf.concat([encoder_state.c, self.post_encoder_states_c], 1)
                encoder_state_h = tf.concat([encoder_state.h, self.post_encoder_states_h], 1)
                encoder_state_c.set_shape([None, conf_file.dyn_graph_emb_dim+conf_file.post_encoder_batch_dim])
                encoder_state_c2 = tf.contrib.layers.fully_connected(encoder_state_c, num_outputs=conf_file.dyn_graph_emb_dim,
                                                                     activation_fn=tf.nn.relu)
                encoder_state_h.set_shape([None, conf_file.dyn_graph_emb_dim+conf_file.post_encoder_batch_dim])
                encoder_state_h2 = tf.contrib.layers.fully_connected(encoder_state_h, num_outputs=conf_file.dyn_graph_emb_dim,
                                                                     activation_fn=tf.nn.relu)
                encoder_state = tf.nn.rnn_cell.LSTMStateTuple(c=encoder_state_c2, h=encoder_state_h2)
                # -----------------------Concatenate with neural layers ends----------------------------

                '''
                ##-----------Post encoder (start)--------------
                # -------------concatenate output start ----------------------------
                # encoder_outputs2=encoder_outputs
                if conf_file.g2s:
                    post_encoder_output_mnth = tf.reduce_sum(self.post_encoder_output, 0, keep_dims=True)
                    post_encoder_states_c_mnth = tf.reduce_sum(self.post_encoder_states_c, 0, keep_dims=True)
                    post_encoder_states_h_mnth = tf.reduce_sum(self.post_encoder_states_h, 0, keep_dims=True)
                    encoder_outputs = tf.concat([encoder_outputs, post_encoder_output_mnth], 2)
                    encoder_state_c = tf.concat([encoder_state.c, post_encoder_states_c_mnth], 1)
                    encoder_state_h = tf.concat([encoder_state.h, post_encoder_states_h_mnth], 1)

                else:
                    encoder_outputs = tf.concat([encoder_outputs, self.post_encoder_output], 2)
                    encoder_state_c = tf.concat([encoder_state.c, self.post_encoder_states_c], 1)
                    encoder_state_h = tf.concat([encoder_state.h, self.post_encoder_states_h], 1)

                encoder_outputs.set_shape([None, None, 256])
                # encoder_outputs2 = tf.contrib.layers.fully_connected(encoder_outputs, num_outputs=128,
                #                                                      activation_fn=tf.nn.relu)
                encoder_outputs2 = tf.contrib.layers.fully_connected(encoder_outputs, num_outputs=128,
                                                                     activation_fn=None)
                # -------------concatenate output end ----------------------------
                # -----------------------Concatenate with neural layers starts----------------------------

                encoder_state_c.set_shape([None, 256])
                # encoder_state_c2 = tf.contrib.layers.fully_connected(encoder_state_c, num_outputs=128,
                #                                                      activation_fn=tf.nn.relu)
                encoder_state_c2 = tf.contrib.layers.fully_connected(encoder_state_c, num_outputs=128,
                                                                     activation_fn=None)
                encoder_state_h.set_shape([None, 256])
                # encoder_state_h2 = tf.contrib.layers.fully_connected(encoder_state_h, num_outputs=128,
                #                                                      activation_fn=tf.nn.relu)
                encoder_state_h2 = tf.contrib.layers.fully_connected(encoder_state_h, num_outputs=128,
                                                                     activation_fn=None)
                encoder_state = tf.nn.rnn_cell.LSTMStateTuple(c=encoder_state_c2, h=encoder_state_h2)
                # -----------------------Concatenate with neural layers ends----------------------------
                ##------------Post encoder (end)-----------------

                ##### OFF GRAPH-------------
                # encoder_state = tf.nn.rnn_cell.LSTMStateTuple(c=self.post_encoder_states_c, h=self.post_encoder_states_h)
                ##### OFF GRAPH END---------'''


                return encoder_outputs2, encoder_state, source_sequence_length # post encoder

            else:
                return encoder_outputs, encoder_state, source_sequence_length
            # return self.post_encoder_output, encoder_state, source_sequence_length
            # return encoder_outputs2, encoder_state, source_sequence_length

    def encode_node_feature(self, word_embeddings, feature_info): #------------> focus here. Output should be = [batch_size * node_size*timestamp, node_embedding
        # in some cases, we can use LSTM to produce the node feature representation
        # cell = self._build_encoder_cell(conf.num_layers, conf.dim)

        if self.node_feature_format == "bow":
            node_repres = tf.cast(feature_info, tf.float32)
        elif self.node_feature_format == "word_emb":
            # feature_info [batch_size*node_size, word_max_len]
            feature_embedded_chars = tf.nn.embedding_lookup(word_embeddings, feature_info)
            batch_size = tf.shape(feature_embedded_chars)[0]
            # concat way
            # node_repres = tf.reshape(feature_embedded_chars, [batch_size, -1])
            # average way
            node_repres = tf.reduce_mean(feature_embedded_chars, 1)

        # output dim: [batch_size * node_size, node_embedding] # batch=user, node=graph
        return node_repres

    def optimized_gcn_encode(self, t): #t = timestamp -- focus here, may be change here

        fw_aggregators_t = []
        bw_aggregators_t = []

        if conf_file.um_specific:
            feature_info_time_slice = tf.reshape(tf.slice(self.feature_info, [0, t, 0],
                                                          [tf.shape(self.feature_info)[0], 1,
                                                           tf.shape(self.feature_info)[2]]),
                                                 [tf.shape(self.feature_info)[0], tf.shape(self.feature_info)[2]])
            embedded_node_rep = self.encode_node_feature(self.word_embeddings, feature_info_time_slice)  # edited by TC
        else:
            # [node_size, word_max_len * hidden_layer_dim]
            embedded_node_rep = self.encode_node_feature(self.word_embeddings, self.feature_info) # ----> change the feature_info. here node size = user*subforum = example 10*80 = 800

        # change to user*subforum*timestamp
        fw_sampler = UniformNeighborSampler(tf.gather(self.fw_adj_info, t))
        bw_sampler = UniformNeighborSampler(tf.gather(self.bw_adj_info, t))
        nodes = tf.reshape(self.batch_nodes, [-1, ])

        # batch_size = tf.shape(nodes)[0]

        # the fw_hidden and bw_hidden is the initial node embedding
        # [node_size, word_max_len * hidden_layer_dim]
        fw_hidden = tf.nn.embedding_lookup(embedded_node_rep, nodes)
        bw_hidden = tf.nn.embedding_lookup(embedded_node_rep, nodes)

        # [node_size, adj_size]
        fw_sampled_neighbors = fw_sampler((nodes, self.sample_size_per_layer))
        bw_sampled_neighbors = bw_sampler((nodes, self.sample_size_per_layer))

        fw_sampled_neighbors_len = tf.constant(0)
        bw_sampled_neighbors_len = tf.constant(0)

        # sample
        for layer in range(self.sample_layer_size):
            if layer == 0:
                # concat
                # input_dim = self.feature_max_len * self.word_embedding_dim
                # avg
                if self.node_feature_format == "bow":
                    input_dim = self.word_vocab_size
                elif self.node_feature_format == "word_emb":
                    input_dim = self.word_embedding_dim
            else:
                input_dim = self.hidden_layer_dim * 2

            if layer > 6:
                fw_aggregator = fw_aggregators_t[6]
            else:
                fw_aggregator = MeanAggregator(input_dim, self.hidden_layer_dim, concat=self.concat, mode=self.mode)
                fw_aggregators_t.append(fw_aggregator)

            # [node_size, adj_size, word_embedding_dim]
            if layer == 0:
                neigh_vec_hidden = tf.nn.embedding_lookup(embedded_node_rep, fw_sampled_neighbors)

                # compute the neighbor size
                tmp_sum = tf.reduce_sum(tf.nn.relu(neigh_vec_hidden), axis=2)
                tmp_mask = tf.sign(tmp_sum)
                fw_sampled_neighbors_len = tf.reduce_sum(tmp_mask, axis=1)
            else:
                neigh_vec_hidden = tf.nn.embedding_lookup(
                    tf.concat([fw_hidden, tf.zeros([1, input_dim])], 0), fw_sampled_neighbors)

            fw_hidden = fw_aggregator((fw_hidden, neigh_vec_hidden, fw_sampled_neighbors_len))

            if self.graph_encode_direction == "bi":
                if layer > 6:
                    bw_aggregator = bw_aggregators_t[6]
                else:
                    bw_aggregator = MeanAggregator(input_dim, self.hidden_layer_dim, concat=self.concat, mode=self.mode)
                    bw_aggregators_t.append(bw_aggregator)

                if layer == 0:
                    neigh_vec_hidden = tf.nn.embedding_lookup(embedded_node_rep, bw_sampled_neighbors)
                    # compute the neighbor size
                    tmp_sum = tf.reduce_sum(tf.nn.relu(neigh_vec_hidden), axis=2)
                    tmp_mask = tf.sign(tmp_sum)
                    bw_sampled_neighbors_len = tf.reduce_sum(tmp_mask, axis=1)
                else:
                    neigh_vec_hidden = tf.nn.embedding_lookup(
                        tf.concat([bw_hidden, tf.zeros([1, input_dim])], 0), bw_sampled_neighbors)

                bw_hidden = bw_aggregator((bw_hidden, neigh_vec_hidden, bw_sampled_neighbors_len))

        self.fw_aggregators.append(fw_aggregators_t)
        self.bw_aggregators.append(bw_aggregators_t)

        # hidden stores the representation for all nodes
        fw_hidden = tf.reshape(fw_hidden, [-1, self.single_graph_nodes_size, 2 * self.hidden_layer_dim])
        if self.graph_encode_direction == "bi":
            bw_hidden = tf.reshape(bw_hidden, [-1, self.single_graph_nodes_size, 2 * self.hidden_layer_dim])
            hidden = tf.concat([fw_hidden, bw_hidden], axis=2)
        else:
            hidden = fw_hidden

        if self.hierarchical_attention: # use separate boolean variable
            # Apply feed forward attention mechanism for aggregating node rep to graph rep
            attention_logits = tf.matmul(tf.reshape(hidden, [-1, 4 * self.hidden_layer_dim]),
                                         self.node_attention_layer['weights']) + self.node_attention_layer['biases']
            # logits [batch_size, node_num, 1]
            attention_logits = tf.reshape(attention_logits, [-1, self.single_graph_nodes_size, 1])
            # weights [batch_size, node_num, 1]
            attention_weights = tf.nn.softmax(attention_logits, axis=1)
            # store the weights for output
            # [time, batch_size, node_num]
            self.node_attention.append(attention_weights)

            graph_embedding = tf.reduce_sum(tf.multiply(attention_weights, hidden), 1)
        else:
            # use max pooling suggested by graph2seq model
            hidden = tf.nn.relu(hidden)
            pooled = tf.reduce_max(hidden, 1)
            if self.graph_encode_direction == "bi":
                graph_embedding = tf.reshape(pooled, [-1, 4 * self.hidden_layer_dim])
            else:
                graph_embedding = tf.reshape(pooled, [-1, 2 * self.hidden_layer_dim])

        # shape of hidden: [batch_size, single_graph_nodes_size, 4 * hidden_layer_dim]
        # shape of graph_embedding: [batch_size, 4 * hidden_layer_dim]
        return hidden, graph_embedding

    def decode(self, encoder_outputs, encoder_state, source_sequence_length):
        with tf.variable_scope("Decoder") as scope:
            encoder_outputs = tf.transpose(encoder_outputs, [1, 0, 2])
            beam_width = self.beam_width
            decoder_type = self.decoder_type
            seq_max_len = self.seq_max_len
            batch_size = tf.shape(encoder_outputs)[0]

            if self.path_embed_method == "lstm":
                self.decoder_cell = self._build_decode_cell()
                if self.mode == "test" and beam_width > 0:
                    memory = seq2seq.tile_batch(encoder_outputs, multiplier=beam_width)
                    source_sequence_length = seq2seq.tile_batch(source_sequence_length, multiplier=beam_width)
                    encoder_state = seq2seq.tile_batch(encoder_state, multiplier=beam_width)
                    batch_size = batch_size * beam_width
                else:
                    memory = encoder_outputs
                    source_sequence_length = source_sequence_length
                    encoder_state = encoder_state

                # attention mechanism options
                if self.attention:
                    if self.attention_method == "luong":
                        attention_mechanism = seq2seq.LuongAttention(
                            self.dyn_graph_emb_dim, memory, memory_sequence_length=source_sequence_length, scale=True)
                    elif self.attention_method == "mono_luong":
                        # mode can be 'recursive', 'parallel', or 'hard'
                        # recommend score_bias_init in [-1, -2, -3, -4] for long src dyngraphs
                        attention_mechanism = seq2seq.LuongMonotonicAttention(
                            self.dyn_graph_emb_dim, memory, memory_sequence_length=source_sequence_length, scale=True,
                            sigmoid_noise=1.0, score_bias_init=-1.0)
                    elif self.attention_method == "bahdanau":
                        attention_mechanism = seq2seq.BahdanauAttention(
                            self.dyn_graph_emb_dim, memory, memory_sequence_length=source_sequence_length, normalize=True)
                    elif self.attention_method == "mono_bahdanau":
                        # mode can be 'recursive', 'parallel', or 'hard'
                        # recommend score_bias_init in [-1, -2, -3, -4] for long src dyngraphs
                        attention_mechanism = seq2seq.BahdanauMonotonicAttention(
                            self.dyn_graph_emb_dim, memory, memory_sequence_length=source_sequence_length, normalize=True,
                            sigmoid_noise=1.0, score_bias_init=-1.0)
                    else:
                        raise ValueError("Unknown attention method %s" % self.attention_method)

                    # only show alignment when greedy infer mode is used
                    alignment_history = (decoder_type == "greedy")

                    self.decoder_cell = seq2seq.AttentionWrapper(self.decoder_cell, attention_mechanism,
                                                                 attention_layer_size=self.dyn_graph_emb_dim,
                                                                 alignment_history=alignment_history)
                    self.decoder_initial_state = self.decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)
                else:
                    self.decoder_initial_state = encoder_state

            projection_layer = Dense(self.word_vocab_size, use_bias=False)

            """For training the model"""
            if self.mode == "train":
                decoder_train_helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_train_inputs_embedded,
                                                                         self.decoder_train_length)
                decoder_train = seq2seq.BasicDecoder(self.decoder_cell, decoder_train_helper,
                                                     self.decoder_initial_state,
                                                     projection_layer)
                decoder_outputs_train, decoder_states_train, decoder_seq_len_train = seq2seq.dynamic_decode(decoder_train)
                decoder_logits_train = decoder_outputs_train.rnn_output
                self.decoder_logits_train = tf.reshape(decoder_logits_train, [batch_size, -1, self.word_vocab_size])

            """For test the model"""
            # if self.mode == "infer" or self.if_pred_on_dev:
            if decoder_type == "greedy":
                decoder_infer_helper = seq2seq.GreedyEmbeddingHelper(self.word_embeddings,
                                                                     tf.ones([batch_size], dtype=tf.int32),
                                                                     self.EOS)
                decoder_infer = seq2seq.BasicDecoder(self.decoder_cell, decoder_infer_helper,
                                                     self.decoder_initial_state, projection_layer)
            elif decoder_type == "beam":
                decoder_infer = seq2seq.BeamSearchDecoder(cell=self.decoder_cell, embedding=self.word_embeddings,
                                                          start_tokens=tf.ones([self.batch_size], dtype=tf.int32),
                                                          end_token=self.EOS,
                                                          initial_state=self.decoder_initial_state,
                                                          beam_width=beam_width,
                                                          output_layer=projection_layer)

            decoder_outputs_infer, decoder_states_infer, decoder_seq_len_infer = seq2seq.dynamic_decode(decoder_infer,
                                                                                                        maximum_iterations=seq_max_len)
            if decoder_type == "beam":
                self.decoder_logits_infer = tf.no_op()
                self.sample_id = decoder_outputs_infer.predicted_ids

            elif decoder_type == "greedy":
                self.decoder_logits_infer = decoder_outputs_infer.rnn_output
                self.sample_id = decoder_outputs_infer.sample_id

            self.infer_summary = self._get_attention_summary(decoder_states_infer)

    def _build_decode_cell(self):
        if self.dyn_num_layers == 1:
            cell = tf.contrib.rnn.BasicLSTMCell(self.dyn_graph_emb_dim)
            if self.mode == "train":
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, 1 - self.dropout)
            return cell
        else:
            cell_list = []
            for i in range(self.dyn_num_layers):
                single_cell = tf.contrib.rnn.BasicLSTMCell(self.dyn_graph_emb_dim)
                if self.mode == "train":
                    single_cell = tf.nn.rnn_cell.DropoutWrapper(single_cell, 1 - self.dropout)
                cell_list.append(single_cell)
            return tf.contrib.rnn.MultiRNNCell(cell_list)

    def _build_encoder_cell(self, num_layers, hidden_layer_dim):
        if num_layers == 1:
            cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_layer_dim)
            if self.mode == "train":
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, 1 - self.dropout)
            return cell
        else:
            cell_list = []
            for i in range(num_layers):
                single_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layer_dim)
                if self.mode == "train":
                    single_cell = tf.nn.rnn_cell.DropoutWrapper(single_cell, 1 - self.dropout)
                cell_list.append(single_cell)
            return tf.contrib.rnn.MultiRNNCell(cell_list)

    def _init_optimizer(self):
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.decoder_train_targets, logits=self.decoder_logits_train)
        decode_loss = (tf.reduce_sum(crossent * self.loss_weights) / tf.cast(self.batch_size, tf.float32))

        train_loss = decode_loss
        l21_loss = tf.constant(0, dtype=tf.float32)

        if self.beta > 0.0:
            # concat feature weights into one vector for each timestamp
            # dim = [timestamp, total dimension]
            fw_aggregators_weight_matrix = []
            for aggregators_t in self.fw_aggregators:
                # dim = [total dimension]
                weights_at_t = tf.concat([aggregators_t[0].vars['self_weights'], aggregators_t[0].vars['neigh_weights']], 1)

                fw_aggregators_weight_matrix.append(weights_at_t)
            # [time, feature, ?]
            fw_aggregators_weight_matrix = tf.stack(fw_aggregators_weight_matrix)

            bw_aggregators_weight_matrix = []
            for aggregators_t in self.bw_aggregators:
                # dim = [total dimension]
                weights_at_t = tf.concat([aggregators_t[0].vars['self_weights'], aggregators_t[0].vars['neigh_weights']], 1)

                bw_aggregators_weight_matrix.append(weights_at_t)
            # [time, feature, ?]
            bw_aggregators_weight_matrix = tf.stack(bw_aggregators_weight_matrix)

            # reshape to [feature,time,?]
            fw_aggregators_weight_matrix = tf.transpose(fw_aggregators_weight_matrix, [1, 0, 2])
            bw_aggregators_weight_matrix = tf.transpose(bw_aggregators_weight_matrix, [1, 0, 2])

            feature_dim = tf.shape(fw_aggregators_weight_matrix)[0]

            for t in range(0, self.timestamp - self.window_size + 1, self.stride):
                # old way
                # l21_loss += self.beta * tf.reduce_sum(tf.norm(fw_aggregators_weight_matrix[t:t + self.window_size, :], axis=0))
                # l21_loss += self.beta * tf.reduce_sum(tf.norm(bw_aggregators_weight_matrix[t:t + self.window_size, :], axis=0))
                # group way
                l21_loss += self.beta * tf.reduce_sum(
                    tf.norm(tf.reshape(fw_aggregators_weight_matrix[:, t:t + self.window_size, :], [feature_dim, -1]),
                            axis=1))
                l21_loss += self.beta * tf.reduce_sum(
                    tf.norm(tf.reshape(bw_aggregators_weight_matrix[:, t:t + self.window_size, :], [feature_dim, -1]),
                            axis=1))

        train_loss += l21_loss

        # L2 loss of vars in aggregators
        for fw_aggregators_t in self.fw_aggregators:
            for aggregator in fw_aggregators_t:
                for var in aggregator.vars.values():
                    train_loss += self.l2_lambda * tf.nn.l2_loss(var)
        for bw_aggregators_t in self.bw_aggregators:
            for aggregator in bw_aggregators_t:
                for var in aggregator.vars.values():
                    train_loss += self.l2_lambda * tf.nn.l2_loss(var)

        self.loss_op = train_loss
        self.l21_loss = l21_loss
        params = tf.trainable_variables()
        gradients = tf.gradients(train_loss, params)

        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(clipped_gradients, params))



    def _build_graph(self):
        encoder_outputs, encoder_state, source_sequence_length = self.encode()

        if self.mode == "train":
            self._init_decoder_train_connectors()

        self.decode(encoder_outputs=encoder_outputs, encoder_state=encoder_state,
                                                 source_sequence_length=source_sequence_length)

        if self.mode == "train":
            self._init_optimizer()

        self.feature_weights = []
        if self.mode == "test" and self.beta > 0.0:
            self._formulate_feature_weight_matrix()

    def _get_attention_summary(self, final_context_state):
        if not self.attention or self.decoder_type == "beam":
            return tf.no_op()

        """create attention image and attention summary."""
        attention_images = (final_context_state.alignment_history.stack())
        # Reshape to (batch, src_seq_len, tgt_seq_len,1)
        attention_images = tf.expand_dims(
            tf.transpose(attention_images, [1, 2, 0]), -1)
        # Scale to range [0, 255]
        attention_images *= 255
        attention_summary = tf.summary.image("attention_images", attention_images,
                                             max_outputs=self.test_batch_size)
        # print(attention_images.get_shape().as_list())
        return attention_summary

    def _formulate_feature_weight_matrix(self):
        fw_aggregators_weight_matrix = []
        for aggregators_t in self.fw_aggregators:
            # dim = [total dimension]
            weights_at_t = tf.concat([aggregators_t[0].vars['self_weights'], aggregators_t[0].vars['neigh_weights']], 1)

            fw_aggregators_weight_matrix.append(weights_at_t)
        # [time, feature, ?]
        fw_aggregators_weight_matrix = tf.stack(fw_aggregators_weight_matrix)

        bw_aggregators_weight_matrix = []
        for aggregators_t in self.bw_aggregators:
            # dim = [total dimension]
            weights_at_t = tf.concat([aggregators_t[0].vars['self_weights'], aggregators_t[0].vars['neigh_weights']], 1)

            bw_aggregators_weight_matrix.append(weights_at_t)
        # [time, feature, ?]
        bw_aggregators_weight_matrix = tf.stack(bw_aggregators_weight_matrix)

        # reshape to [feature,time,?]
        fw_aggregators_weight_matrix = tf.transpose(fw_aggregators_weight_matrix, [1, 0, 2])
        bw_aggregators_weight_matrix = tf.transpose(bw_aggregators_weight_matrix, [1, 0, 2])

        # [2,feature,time,?]
        self.feature_weights = tf.stack([fw_aggregators_weight_matrix, bw_aggregators_weight_matrix])

    def act(self, sess, mode, dict, if_pred_on_dev):
        text = np.array(dict['seq'])
        decoder_seq_length = np.array(dict['decoder_seq_length'])
        loss_weights = np.array(dict['loss_weights'])
        batch_graph = dict['batch_graph']
        ## [t, ...]
        fw_adj_info = batch_graph['g_fw_adj_seq']
        bw_adj_info = batch_graph['g_bw_adj_seq']
        feature_info = batch_graph['g_ids_features'] # focus here
        batch_nodes = batch_graph['g_nodes']
        self.if_pred_on_dev = if_pred_on_dev

        if conf_file.post_encoder:
            # -------Post Encoder (Start)--------------
            batch_posts = batch_graph['g_posts']  # Post Encoder
            # newTic = time.perf_counter()
            reshape_axis_2 = batch_posts.shape[0]
            # Need to change the next line based on dataset
            encoded_user_output, encoded_user_state = (
                sess.run(self.encoder2_output, {self.post_vect_inputs: batch_posts, self.reshape_axis2: reshape_axis_2}))
            # encoded_user_output (batchx96x128)
            # print(type(encoded_user_output))
            encoded_user_output_reshaped = np.reshape(encoded_user_output, (encoded_user_output.shape[1], encoded_user_output.shape[0], encoded_user_output.shape[2]))  # 6/16/20

            post_encoder_states_c = encoded_user_state[0]
            # print(encoded_user_state.shape)  # (2xuserx128)
            post_encoder_states_h = encoded_user_state[1]

            # newToc = time.perf_counter()
            # print("second encoder required {} seconds".format((newToc - newTic)))

            feed_dict = {
                self._text: text,
                self.decoder_seq_length: decoder_seq_length,
                self.loss_weights: loss_weights,
                self.fw_adj_info: fw_adj_info,
                self.bw_adj_info: bw_adj_info,
                self.feature_info: feature_info,
                self.batch_nodes: batch_nodes,
                self.post_encoder_states_c: post_encoder_states_c,
                self.post_encoder_states_h: post_encoder_states_h,
                self.post_encoder_output: encoded_user_output_reshaped
            }
            # --------Post Encoder (End)----------------
        else:
            feed_dict = {
                self._text: text,
                self.decoder_seq_length: decoder_seq_length,
                self.loss_weights: loss_weights,
                self.fw_adj_info: fw_adj_info,
                self.bw_adj_info: bw_adj_info,
                self.feature_info: feature_info,
                self.batch_nodes: batch_nodes
            }
        if mode == "train" and not if_pred_on_dev:
            output_feeds = [self.train_op, self.loss_op, self.l21_loss]
        elif mode == "test" or if_pred_on_dev:
            output_feeds = [self.sample_id, self.infer_summary, self.node_attention, self.feature_weights]

        results = sess.run(output_feeds, feed_dict)
        return results

