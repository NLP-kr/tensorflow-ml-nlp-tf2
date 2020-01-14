# -*- coding: utf-8 -*-
import tensorflow as tf
import sys

from configs import DEFINES
import numpy as np


def layer_norm(inputs, eps=1e-6):
    # LayerNorm(x + Sublayer(x))
    feature_shape = inputs.get_shape()[-1:]
    #  평균과 표준편차을 넘겨 준다.
    mean = tf.keras.backend.mean(inputs, [-1], keepdims=True)
    std = tf.keras.backend.std(inputs, [-1], keepdims=True)
    beta = tf.get_variable("beta", initializer=tf.zeros(feature_shape))
    gamma = tf.get_variable("gamma", initializer=tf.ones(feature_shape))

    return gamma * (inputs - mean) / (std + eps) + beta


def sublayer_connection(inputs, sublayer, dropout=0.2):
    outputs = layer_norm(inputs + tf.keras.layers.Dropout(dropout)(sublayer))
    return outputs


def positional_encoding(dim, sentence_length):
    encoded_vec = np.array([pos/np.power(10000, 2*i/dim)
                            for pos in range(sentence_length) for i in range(dim)])

    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])

    return tf.constant(encoded_vec.reshape([sentence_length, dim]), dtype=tf.float32)


class MultiHeadAttention(tf.keras.Model):
    def __init__(self, num_units, heads, masked=False):
        super(MultiHeadAttention, self).__init__()

        self.heads = heads
        self.masked = masked
        self.num_units = num_units #cwjun
        self.depth = num_units // self.heads #cwjun
        
        self.query_dense = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)
        self.key_dense = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)
        self.value_dense = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)
        
        self.dense = tf.keras.layers.Dense(num_units) #cwjun

    def scaled_dot_product_attention(self, query, key, value, masked=False):
        #key_seq_length = float(key.get_shape().as_list()[-1])
        #key = tf.transpose(key, perm=[0, 2, 1])
        #outputs = tf.matmul(query, key) / tf.sqrt(key_seq_length)

        matmul_qk = tf.matmul(query, key, transpose_b=True)#cwjun
        dk = tf.cast(tf.shape(key)[-1], tf.float32)#cwjun
        outputs = matmul_qk / tf.math.sqrt(dk)#cwjun
        
        if masked:
            diag_vals = tf.ones_like(outputs[0, :, :])
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1, 1])

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

        attention_map = tf.nn.softmax(outputs)

        return tf.matmul(attention_map, value)

    def call(self, query, key, value):
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        query = tf.reshape(query, (tf.shape(query)[0], -1, self.heads, self.depth))#cwjun
        query = tf.transpose(query, perm=[0, 2, 1, 3]) # (batch_size, num_heads, seq_len, depth)
        
        key = tf.reshape(key, (tf.shape(key)[0], -1, self.heads, self.depth))#cwjun
        key = tf.transpose(key, perm=[0, 2, 1, 3])
        
        value = tf.reshape(value, (tf.shape(value)[0], -1, self.heads, self.depth))#cwjun
        value = tf.transpose(value, perm=[0, 2, 1, 3])      
        
        #query = tf.concat(tf.split(query, self.heads, axis=-1), axis=0)
        #key = tf.concat(tf.split(key, self.heads, axis=-1), axis=0)
        #value = tf.concat(tf.split(value, self.heads, axis=-1), axis=0)

        attention_map = self.scaled_dot_product_attention(query, key, value, self.masked)

        attention_map = tf.transpose(attention_map, perm=[0, 2, 1, 3]) # (batch_size, seq_len_q, num_heads, depth)#cwjun
        
        #attn_outputs = tf.concat(tf.split(attention_map, self.heads, axis=0), axis=-1)#cwjun
        
        attention_map = tf.reshape(attention_map, (tf.shape(value)[0], -1, self.num_units))#cwjun
        
        attn_outputs = self.dense(attention_map) #cwjun 

        return attn_outputs


class Encoder(tf.keras.Model):
    def __init__(self, model_dims, ffn_dims, attn_heads, num_layers=1):
        super(Encoder, self).__init__()

        self.self_attention = [MultiHeadAttention(model_dims, attn_heads) for _ in range(num_layers)]
        self.position_feedforward = [PositionWiseFeedForward(ffn_dims, model_dims) for _ in range(num_layers)]

    def call(self, inputs):
        output_layer = None

        for i, (s_a, p_f) in enumerate(zip(self.self_attention, self.position_feedforward)):
            with tf.variable_scope('encoder_layer_' + str(i + 1)):
                attention_layer = sublayer_connection(inputs, s_a(inputs, inputs, inputs))
                output_layer = sublayer_connection(attention_layer, p_f(attention_layer))

                inputs = output_layer

        return output_layer


class Decoder(tf.keras.Model):
    def __init__(self, model_dims, ffn_dims, attn_heads, num_layers=1):
        super(Decoder, self).__init__()

        self.self_attention = [MultiHeadAttention(model_dims, attn_heads, masked=True) for _ in range(num_layers)]
        self.encoder_decoder_attention = [MultiHeadAttention(model_dims, attn_heads) for _ in range(num_layers)]
        self.position_feedforward = [PositionWiseFeedForward(ffn_dims, model_dims) for _ in range(num_layers)]

    def call(self, inputs, encoder_outputs):
        output_layer = None

        for i, (s_a, ed_a, p_f) in enumerate(zip(self.self_attention, self.encoder_decoder_attention, self.position_feedforward)):
            with tf.variable_scope('decoder_layer_' + str(i + 1)):
                masked_attention_layer = sublayer_connection(inputs, s_a(inputs, inputs, inputs))
                attention_layer = sublayer_connection(masked_attention_layer, ed_a(masked_attention_layer,
                                                                                           encoder_outputs,
                                                                                           encoder_outputs))
                output_layer = sublayer_connection(attention_layer, p_f(attention_layer))
                inputs = output_layer

        return output_layer


class PositionWiseFeedForward(tf.keras.Model):
    def __init__(self, num_units, feature_shape):
        super(PositionWiseFeedForward, self).__init__()

        self.inner_dense = tf.keras.layers.Dense(num_units, activation=tf.nn.relu)
        self.output_dense = tf.keras.layers.Dense(feature_shape)

    def call(self, inputs):
        inner_layer = self.inner_dense(inputs)
        outputs = self.output_dense(inner_layer)

        return outputs


def Model(features, labels, mode, params):
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT

    position_encode = positional_encoding(params['embedding_size'], params['max_sequence_length'])

    embedding = tf.keras.layers.Embedding(params['vocabulary_length'],
                                          params['embedding_size'])

    encoder_layers = Encoder(params['model_hidden_size'], params['ffn_hidden_size'],
                      params['attention_head_size'], params['layer_size'])

    decoder_layers = Decoder(params['model_hidden_size'], params['ffn_hidden_size'],
                      params['attention_head_size'], params['layer_size'])

    logit_layer = tf.keras.layers.Dense(params['vocabulary_length'])

    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        x_embedded_matrix = embedding(features['input']) + position_encode
        encoder_outputs = encoder_layers(x_embedded_matrix)

    loop_count = params['max_sequence_length'] if PREDICT else 1

    predict, output, logits = None, None, None

    for i in range(loop_count):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            if i > 0:
                output = tf.concat([tf.ones((output.shape[0], 1), dtype=tf.int64), predict[:, :-1]], axis=-1)
            else:
                output = features['output']

            y_embedded_matrix = embedding(output) + position_encode
            decoder_outputs = decoder_layers(y_embedded_matrix, encoder_outputs)

            logits = logit_layer(decoder_outputs)
            predict = tf.argmax(logits, 2)

    if PREDICT:
        predictions = {
            'indexs': predict,
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predict, name='accOp')

    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    assert TRAIN

    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
