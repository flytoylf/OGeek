#!/usr/bin/env python

import tensorflow as tf


def nlp_embedding(inputs, vocab_size, params, dictionary, initializer=tf.glorot_normal_initializer()):
    word_embedding_dim = params["word_embedding_dim"]
    tag_embedding_dim = params["tag_embedding_dim"]
    is_use_pretrained_word_embedding = params["is_use_pretrained_word_embedding"]

    
    all_embedding = dict()
    with tf.variable_scope("embedding"):
        embedding = []
        embedding_dim = 0

        word_vocab_size = vocab_size["words"]
        if is_use_pretrained_word_embedding:
            word_embedding_weight = tf.get_variable(
                name="words_embedding_weight",
                initializer=dictionary.pretrained_eord_emnedding,
            )
        else:
            word_embedding_weight = tf.get_variable(
                name="words_embedding_weight",
                shape=[word_vocab_size, word_embedding_dim],
                initializer=initializer,
            )
        
        tag_vocab_size = vocab_size["tag"]
        tag_embedding_weight = tf.get_variable(
            name="tag_embedding_weight",
            shape=[tag_vocab_size, tag_embedding_dim],
            initializer=initializer,
        )
        
        for tt in ["prefix", "title", "texts", "segments"]:
            embedding = tf.nn.embedding_lookup(word_embedding_weight, inputs[tt])
            embedding_dim = word_embedding_dim
            all_embedding[tt] = {
                    "embedding": embedding, 
                    "embedding_dim": embedding_dim,
                }
        tt = "tag"
        embedding = tf.nn.embedding_lookup(tag_embedding_weight, inputs[tt])
        embedding_dim = tag_embedding_dim
        all_embedding[tt] = {
                "embedding": embedding, 
                "embedding_dim": embedding_dim,
            }

        tt = "scores"
        all_embedding[tt] = inputs[tt]


    return all_embedding, word_embedding_weight

