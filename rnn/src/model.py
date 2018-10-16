#!/usr/bin/env python
# coding: utf-8

import sys
import tensorflow as tf

sys.path.append("./src")
import embedding_core
import network_core
import loss_core
import optimizer_core
from ClassificationModel import ClassificationModel


class Model(ClassificationModel):
    def __new__(cls, params, vocab_size, dictionary):
        model = ClassificationModel.__new__(cls, params, vocab_size)
        model.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        model.embedding, model.word_embedding_matrix = embedding_core.nlp_embedding(
            model.inputs, model.vocab_size, model.params, dictionary, 
            initializer=model.initializer,
        )

        model.output, model.output_width = model.model_layer(params)
        model.logits = model.project_layer()
        model.loss, model.preds = loss_core.classification_loss(model.logits, model.targets, model.num_targets, model.params)
        
        model.train_op = optimizer_core.get_train_op(model.loss, model.global_step, model.params)

        # saver of the model
        model.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        return model


    def model_layer(self, params):
        return network_core.network(self.embedding, self.keep_prob, params, self.initializer)


    def project_layer(self):
        with tf.variable_scope("project"):
            # dense layer
            layer = self.output
            with tf.variable_scope("dense"):
                layers = self.params["dense_layers"]
                for i in range(len(layers)):
                    hidden_units = layers[i]["hidden_units"]
                    with tf.variable_scope("dense_%d" % i):
                        W = tf.get_variable("W", shape=[layer.shape[-1], hidden_units],
                                            dtype=tf.float32, initializer=self.initializer)
                        b = tf.get_variable("b", initializer=tf.constant(0.001, shape=[hidden_units]))
                        layer = tf.nn.xw_plus_b(layer, W, b)
                        layer = tf.nn.relu(layer)
            # project to score of tags
            with tf.variable_scope("logits"):
                W = tf.get_variable("W", shape=[layer.shape[-1], self.num_targets],
                                    dtype=tf.float32, initializer=self.initializer)
                b = tf.get_variable("b", initializer=tf.constant(0.001, shape=[self.num_targets]))
                pred = tf.nn.xw_plus_b(layer, W, b)
            return pred

