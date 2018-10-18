#!/usr/bin/env python


import tensorflow as tf

from tensorflow.contrib import crf

def classification_loss(logits, targets, num_targets, params):
    labels = tf.one_hot(tf.squeeze(targets, axis=1), num_targets)
    loss = tf.losses.softmax_cross_entropy(labels, logits, reduction=tf.losses.Reduction.MEAN)
    pred = tf.nn.softmax(logits, name="prediction")
    tf.summary.scalar("loss", loss)
    tf.summary.histogram("pred", tf.argmax(pred, axis=-1))

    return loss, pred
