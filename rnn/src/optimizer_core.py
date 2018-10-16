#!/usr/bin/env python

import tensorflow as tf


def get_train_op(loss, global_step, params):
    optimizer_type = params["optimizer_type"]
    learning_rate = params["learning_rate"]
    lr_decay = params["lr_decay"]
    lr_decay_steps = params["lr_decay_steps"]
    clip = params["clip"]
    lr = tf.train.exponential_decay(learning_rate, global_step, lr_decay_steps, lr_decay, staircase=True)
    tf.summary.scalar("learning_rate", lr)

    with tf.variable_scope("optimizer"):
        if optimizer_type == "sgd":
            opt = tf.train.GradientDescentOptimizer(lr)
        elif optimizer_type == "adam":
            opt = tf.train.AdamOptimizer(lr)
        elif optimizer_type == "adgrad":
            opt = tf.train.AdagradOptimizer(lr)
        else:
            raise KeyError
        train_op = opt.minimize(loss, global_step=global_step)

#        # apply grad clip to avoid gradient explosion
#        grads_vars = opt.compute_gradients(loss)
##       for grad, v in grads_vars:
##           tf.summary.histogram("grads_%s" % v.name, grad)
#        grads_vars = [[tf.clip_by_value(g, -clip, clip), v]
#                      for g, v in grads_vars]
#        for grad, v in grads_vars:
#            tf.summary.histogram("clipped_grads_%s" % v.name, grad)
#        train_op = opt.apply_gradients(grads_vars, global_step=global_step)
        return train_op
