# coding: utf-8


import os
import six
import json
import numpy as np
from collections import defaultdict


class Dictionary(object):
    def __new__(cls, params, data=None, path=None, cutoff=None):
        dictionary = object.__new__(cls)
        dictionary.is_use_pretrained_word_embedding = params["is_use_pretrained_word_embedding"]
        dictionary.pretrained_emnedding_dir = params["pretrained_emnedding_dir"]
        dictionary.word_embedding_dim = params["word_embedding_dim"]
        dictionary.cutoff = None
        dictionary.token_to_id = dict()
        dictionary.id_to_token = dict()
        dictionary.token_count = dict()
        dictionary.pretrained_eord_emnedding = None
        if data is not None:
            dictionary._generate_token_to_id(data)
        else:
            dictionary.load(path)

        if dictionary.is_use_pretrained_word_embedding:
            dictionary._get_pretrained_embedding()

        return dictionary

    def _get_pretrained_embedding(self):
        word_vector = dict()
        with open(self.pretrained_emnedding_dir, 'r', encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tokens = line.split()
                word = tokens[0]
                vec = tokens[1:]
                word_vector[word] = [float(item) for item in vec]

        vecs = list()
        for i in range(len(self.id_to_token["words"])):
            token = self.id_to_token["words"][i]
            if token in word_vector:
                vecs.append(word_vector[token])
            else:
                normal_list = np.random.normal(size=self.word_embedding_dim).tolist()
                vecs.append(normal_list)
        self.pretrained_eord_emnedding = np.array(vecs, dtype=np.float32)


    def _stat_for(self, tag, data):
        if tag == "tag":
            tc = defaultdict(int)
            for sample in data:
                tc[sample["tag"]] += 1

            t2i = {"UNK": 0, "PAD": 1}
            i2t = {0: "UNK", 1: "PAD"}
            for token in tc:
                token_id = len(t2i)
                t2i[token] = token_id
                i2t[token_id] = token
            return tc, t2i, i2t
        else:
            tc = defaultdict(int)
            tags = tag.copy()
            for sample in data:
                for tag in ["prefix", "title", "texts"]:
                    for token in sample[tag]:
                        if token == "PAD":
                            continue
                        tc[token] += 1
                for segment in sample["segments"]:
                    for token in segment:
                        if token == "PAD":
                            continue
                        tc[token] += 1
            t2i = {"UNK": 0, "PAD": 1}
            i2t = {0: "UNK", 1: "PAD"}
            for token in tc:
                token_id = len(t2i)
                t2i[token] = token_id
                i2t[token_id] = token
            return tc, t2i, i2t

    def _generate_token_to_id(self, data):
        tc, t2i, i2t = self._stat_for("tag", data)
        self.token_count["tag"] = tc
        self.token_to_id["tag"] = t2i
        self.id_to_token["tag"] = i2t

        tc, t2i, i2t = self._stat_for(["prefix", "title", "texts", "segments"], data)
        self.token_count["words"] = tc
        self.token_to_id["words"] = t2i
        self.id_to_token["words"] = i2t

    def save(self, path):
        for tag in ["tag", "words"]:
            with open(os.path.join(path, tag), 'w', encoding="utf-8") as of:
                for _token, _id in six.iteritems(self.token_to_id[tag]):
                    of.write("%s\t%d\t%d\n" % (_token, _id, self.token_count[tag][_token]))
        with open(os.path.join(path, "meta.json"), 'w', encoding="utf-8") as of:
            info = {}
            if self.cutoff is not None:
                info["cutoff"] = self.cutoff
            json.dump(info, of)
    
    def load(self, path):
        tags = os.listdir(path)
        for tag in tags:
            if tag == "meta.json":
                with open(os.path.join(path, tag), 'r', encoding="utf-8") as f:
                    info = json.load(f)
                    self.cutoff = info["cutoff"] if "cutoff" in info else None
            else:
                with open(os.path.join(path, tag), 'r', encoding="utf-8") as f:
                    for line in f:
                        line = line.rstrip()
                        if not line or line.startswith("#"):
                            continue
                        _token, _id, _count = line.split("\t")
                        _id, _count = int(_id), int(_count)
                        if tag not in self.token_to_id:
                            self.token_to_id[tag] = dict()
                            self.id_to_token[tag] = dict()
                            self.token_count[tag] = dict()
                        self.token_to_id[tag][_token] = _id
                        self.id_to_token[tag][_id] = _token
                        self.token_count[_token] = _count

    def to_id(self, samples_or_features, tag=None):
        if tag is None:
            ret = [
                {tag: sample[tag] if tag not in ["prefix", "title", "texts", "segments", "tag", "targets"] else self.to_id(sample[tag], tag) for tag in sample}
                for sample in samples_or_features
            ]
            return ret
        else:
            if tag == "tag":
                return [
                    self.token_to_id[tag][samples_or_features] if samples_or_features in self.token_to_id[tag] else self.token_to_id[tag]["UNK"]
                ]
            if tag in ["prefix", "title", "texts"]:
                return [
                    self.token_to_id["words"][token] if token in self.token_to_id["words"] else self.token_to_id["words"]["UNK"] 
                    for token in samples_or_features
                ]
            if tag == "segments":
                return [
                    [self.token_to_id["words"][token] if token in self.token_to_id["words"] else self.token_to_id["words"]["UNK"] for token in segment]
                    for segment in samples_or_features
                ]
            if tag == "targets":
                return [samples_or_features]

    def to_token(self, samples_or_features, tag=None):
        if tag is None:
            return [
                {tag: sample[tag] if tag == "sentence" else self.to_token(sample[tag], tag) for tag in sample}
                for sample in samples_or_features
            ]
        else:
            if tag == "targets":
                return [self.id_to_token[tag][samples_or_features]]
            else:
                return [
                    self.id_to_token[tag][_id] for _id in samples_or_features
                ]

    def pad_id(self):
        return {
            tag: self.token_to_id[tag]['PAD']
            for tag in self.token_to_id if tag != "targets"
        }

    def vocab_size(self):
        return {
            tag: len(self.token_to_id[tag])
            for tag in self.token_to_id
        }
