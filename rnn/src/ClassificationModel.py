#!/usr/bin/env python


import sys

import numpy as np
import tensorflow as tf


class ClassificationModel(object):
    def __new__(cls, params, vocab_size):
        model = object.__new__(cls)
        model.params = params

        model.vocab_size = vocab_size
        model.num_targets = params["num_targets"]
        model.word_embedding_dim = params["word_embedding_dim"]
        model.tag_embedding_dim = params["tag_embedding_dim"]

        model.type_of_tags = ["prefix", "title", "segments", "tag"]
        model.inputs = {
            tt: tf.placeholder(tf.int32, shape=[None, None], name=tt) if tt != "segments" 
            else tf.placeholder(tf.int32, shape=[None, 10, None], name=tt)
            for tt in model.type_of_tags
        }
        model.inputs["scores"] = tf.placeholder(tf.float32, shape=[None, 10], name="scores")
        model.inputs["texts"] = tf.placeholder(tf.int32, shape=[None, 10], name="texts")
        model.targets = tf.placeholder(dtype=tf.int32, shape=[None, 1], name="targets")

        model.batch_size = tf.shape(model.inputs["prefix"])[0]

        model.initializer = tf.glorot_uniform_initializer()
        model.global_step = tf.train.create_global_step()

        model.conf_keep_prob = params["keep_prob"] if "keep_prob" in params else 1.

        model.embedding = None
        model.word_embedding_matrix = None

        model.keep_prob = None
        model.logits = None
        model.loss = None
        model.preds = None
        model.train_op = None

        model.save_path = params["model_path"]
        model.saver = None

        return model


    def set_pretrained_word_embedding(self, sess, embedding_matrix):
        if self.word_embedding_matrix is None:
            return
        assign_op = tf.assign(self.word_embedding_matrix, embedding_matrix)
        sess.run(assign_op)


    def create_feed_dict(self, is_train, batch):
        feed_dict = {
            self.inputs[tt]: batch[tt]
            for tt in self.inputs
        }

        if is_train:
            feed_dict[self.targets] = batch["targets"]
        if self.keep_prob is not None:
            feed_dict[self.targets] = batch["targets"]
            feed_dict[self.keep_prob] = self.conf_keep_prob
        return feed_dict


    def run_step(self, sess, is_train, batch, merge_summary=None, train_writer=None):
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            fetchers = [self.global_step, self.loss, self.preds, self.train_op]
            if merge_summary is not None:
                fetchers.append(merge_summary)
            ret = sess.run(
                fetchers,
                feed_dict)

            global_step, loss, pred = ret[:3]
            if merge_summary is not None:
                train_writer.add_summary(ret[4], global_step)
            return global_step, loss, pred
        else:
            loss, pred = sess.run([self.loss, self.preds], feed_dict)
            return loss, pred


    def predict_line(self, sess, sample):
        preds = self.run_step(sess, False, [sample])[1]
        return preds


    def train(self, sess, batch_manager, steps, metrics, merge_summary=None, train_writer=None):
        global_step = 0
        total_loss = 0
        n_steps = 0
        infinity = steps < 0
        targets = []
        preds = []
        while True:
            try:
                batch, batch_size = batch_manager.batch()
            except EOFError:
                break
            global_step, loss, pred = self.run_step(sess, True, batch, merge_summary=merge_summary, train_writer=train_writer)
            total_loss += loss
            n_steps += 1
            if not infinity:
                steps -= 1
                if steps <= 0:
                    break
            targets.append(batch["targets"]) 
            preds.append(pred)

        targets = np.concatenate(targets)
        preds = np.concatenate(preds)
        return global_step, total_loss / n_steps, n_steps, metrics(targets, preds)


    def eval(self, sess, batch_manager, metrics):
        total_loss = 0
        targets = []
        preds = []
        while True:
            try:
                batch, batch_size = batch_manager.batch()
            except EOFError:
                break
            targets.append(batch["targets"])
            loss, pred = self.run_step(sess, False, batch)
            total_loss += loss
            preds.append(pred)
        targets = np.concatenate(targets)
        preds = np.concatenate(preds)
        return total_loss, metrics(targets, preds)


    def predict(self, sess, batch_manager, steps):
        preds = []
        n_steps = 0
        for _ in range(steps):
            try:
                batch, batch_size = batch_manager.batch()
            except EOFError:
                break
            preds.append(self.run_step(sess, False, batch)[1])
            n_steps += 1
        return np.stack(preds), n_steps


    def save(self, sess, save_path=None):
        if save_path is None:
            save_path = self.save_path
        self.saver.save(sess, save_path)


