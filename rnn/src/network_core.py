#!/usr/bin/env python

import tensorflow as tf
import numpy as np


def cnn_item(model_inputs, model_inputs_dim, keep_prob, layer_conf, initializer, layer_tag='0'):
    model_inputs = tf.expand_dims(model_inputs, 1)
    filter_width = layer_conf["filter_width"]
    output_channel = layer_conf["output_channel"]

    with tf.variable_scope("cnn_layer"):
        with tf.variable_scope(layer_tag+"_input_layer"):
            filter_shape = [1, filter_width, model_inputs_dim, output_channel]
            filter_weight = tf.get_variable(
                layer_tag+"_filter_weight",
                shape=filter_shape,
                initializer=initializer
            )
            filter_bias = tf.get_variable(
                layer_tag+"_filter_bias",
                shape=[output_channel]
            )

            layer = tf.nn.conv2d(model_inputs, filter_weight, strides=[1, 1, 1, 1],
                                 padding="SAME", name=layer_tag+"_input_layer")
            layer = tf.nn.bias_add(layer, filter_bias)
            layer = tf.nn.relu(layer)

        
        output = tf.squeeze(layer, [1])
        pooling_output = tf.reduce_max(output, axis=-2, name=layer_tag+"_pooling_layer")

        return pooling_output, output_channel


def gru_item(model_inputs, model_inputs_dim, keep_prob, layer_conf, initializer, layer_tag):
    gru_size = layer_conf["gru_size"]
    attention_size = layer_conf["attention_size"]

    with tf.variable_scope(layer_tag+"_layer"):
        gru_cell_forward = tf.contrib.rnn.GRUCell(gru_size)
        gru_cell_backward = tf.contrib.rnn.GRUCell(gru_size)

        gru_cell_forward = tf.contrib.rnn.DropoutWrapper(gru_cell_forward, output_keep_prob=keep_prob)
        gru_cell_backward = tf.contrib.rnn.DropoutWrapper(gru_cell_backward, output_keep_prob=keep_prob)

        model_inputs_forward = model_inputs
        model_inputs_backward = tf.reverse(model_inputs, [1])
        forward_outputs, forward_final_state = tf.nn.dynamic_rnn(gru_cell_forward, model_inputs_forward, dtype=tf.float32)
        backward_outputs, backward_final_state = tf.nn.dynamic_rnn(gru_cell_forward, model_inputs_backward, dtype=tf.float32)
        backward_outputs = tf.reverse(backward_outputs, [1])
        
        # attention layer
        def attention_layer(inputs, layer_name):
            with tf.variable_scope(layer_name):
                w_omega = tf.get_variable("w_omega", initializer=tf.random_normal([gru_size, attention_size], stddev=0.1))
                b_omega = tf.get_variable("b_omega", initializer=tf.random_normal([attention_size], stddev=0.1))
                u_omega = tf.get_variable("u_omega", initializer=tf.random_normal([attention_size], stddev=0.1))

                v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, gru_size]), w_omega) + tf.reshape(b_omega, [1, -1]))  
                vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
                alphas = tf.nn.softmax(vu)

                output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

                return output

        forward_outputs = attention_layer(forward_outputs, "forword_attention")
        backward_outputs = attention_layer(backward_outputs, "backward_attention")

        gru_outputs = tf.add(forward_final_state, backward_final_state)
        gru_outputs = tf.nn.relu(gru_outputs)

    return gru_outputs, gru_size


def network(inputs, keep_prob, params, initializer):
    gru_layer_conf = params["gru_layers"]
    cnn_layer_conf = params["cnn_layers"]
    outputs = list()
    outputs_dim = list()
    # ["prefix", "title", "texts", "segments", "tag", "scores"]:
    # inputs["segments"]["embedding"].shape = batch * text_num(10) * text_length * embedding_dim
    
    with tf.variable_scope("basic_info_layer"):
        for tag in ["prefix", "title"]:
            model_inputs = inputs[tag]["embedding"]
            model_inputs_dim = inputs[tag]["embedding_dim"]
            output, output_dim = gru_item(model_inputs, model_inputs_dim, keep_prob, gru_layer_conf, initializer, tag)
            outputs.append(output)
            outputs_dim.append(output_dim)

    with tf.variable_scope("segments_layer"):
        tag = "segments"
        segments_outputs = list()
        for i in range(10):
            model_inputs = inputs[tag]["embedding"][:, i, :, :]
            model_inputs_dim = inputs[tag]["embedding_dim"]
            output, output_dim = gru_item(model_inputs, model_inputs_dim, keep_prob, gru_layer_conf, initializer, tag+"_"+str(i+1))
            outputs.append(output)
            outputs_dim.append(output_dim)
            segments_outputs.append(tf.expand_dims(output, 1))
        segments_output_dim = output_dim

    with tf.variable_scope("segments_sentence_layer"):
        tag = "segments_sentence"
        segments_outputs = tf.concat(segments_outputs, axis=1)
        model_inputs = segments_outputs
        model_inputs_dim = segments_output_dim
        output, output_dim = cnn_item(model_inputs, model_inputs_dim, keep_prob, cnn_layer_conf, initializer, tag)
        outputs.append(output)
        outputs_dim.append(output_dim)
    
    with tf.variable_scope("tag_layer"):
        tag = "tag"
        output = tf.squeeze(inputs[tag]["embedding"], axis=1)
        output_dim = inputs[tag]["embedding_dim"]
        outputs.append(output)
        outputs_dim.append(output_dim)
    
    with tf.variable_scope("scores_layer"):
        tag = "scores"
        output = inputs[tag]
        output_dim = 10
        outputs.append(output)
        outputs_dim.append(output_dim)

    #with tf.variable_scope("segments_sentence_layer"):
    #    tag = "segments_sentence"
    #    segments_outputs = tf.concat(segments_outputs, axis=1)
    #    scores_outputs = inputs["scores"]
    #    output = tf.squeeze(tf.matmul(tf.expand_dims(scores_outputs, 1), segments_outputs), [1])
    #    outputs.append(output)
    #    outputs_dim.append(segments_output_dim)

    #with tf.variable_scope("texts_sentence_layer"):
    #    tag = "texts_sentence"
    #    texts_outputs = inputs["texts"]["embedding"]
    #    scores_outputs = inputs["scores"]
    #    texts_output_dim = inputs["texts"]["embedding"].shape.as_list()[2]
    #    output = tf.squeeze(tf.matmul(tf.expand_dims(scores_outputs, 1), texts_outputs), [1])
    #    outputs.append(output)
    #    outputs_dim.append(texts_output_dim)


    final_output = tf.concat(outputs, axis=-1)
    final_output_dim = sum(outputs_dim)

    return final_output, final_output_dim

