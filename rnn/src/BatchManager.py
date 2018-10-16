# coding: utf-8


import math
import random

import numpy as np


class BatchManager(object):
    def __new__(cls, data, epoch, params, pad_id):
        manager = object.__new__(cls)
        manager.data = data
        manager.len_data = len(manager.data)
        manager.batch_size = params["batch_size"]
        manager.num_batch = int(math.ceil(manager.len_data / manager.batch_size))
        manager.cur_batch = 0
        manager.shuffle = params["shuffle"]
        manager.epoch = epoch
        manager._epoch = epoch
        manager.index = list(range(manager.len_data))
        manager.pad_id = pad_id
        # no shuffle padding before training
        manager.batch_data = [
            manager._construct_batch(manager._padding_batch(
                manager.data[ibatch * manager.batch_size:min((ibatch + 1) * manager.batch_size, manager.len_data)]
            ))
            for ibatch in range(manager.num_batch)
        ]
        return manager

    def _batch_noshuffle(self):
        batch = self.batch_data[self.cur_batch]
        self.cur_batch += 1
        if self.cur_batch >= self.num_batch:
            self.cur_batch = 0
            self.epoch -= 1
        return batch, len(batch)

    def init(self):
        self.epoch = self._epoch
        self.cur_batch = 0
        random.shuffle(self.index)

    @property
    def is_finished(self):
        return self.epoch <= 0

    def batch(self):
        if self.epoch <= 0:
            raise EOFError("epoch exhausted.")

        batch = self._batch_noshuffle()
        return batch

    def _padding_batch(self, batch):
        padding_size = dict()
        for tag in ["prefix", "title", "texts", "segments"]:
            if tag ==  "segments":
                padding_size[tag] = max([max([len(segment) for segment in sample[tag]]) for sample in batch])
            else:
                padding_size[tag] = max([len(sample[tag]) for sample in batch])
        result = list()
        for sample in batch:
            padding_result = dict()
            for tag in sample:
                if tag in ["prefix", "title", "texts"]:
                    padding_result[tag] = np.asarray(sample[tag] + [self.pad_id["words"]] * (padding_size[tag] - len(sample[tag])), np.int32)
                elif tag == "segments":
                    padding_result[tag] = [
                                np.asarray(segment + [self.pad_id["words"]] * (padding_size[tag] - len(segment)), np.int32)
                                for segment in sample[tag]
                         ]
                else:
                    padding_result[tag] = sample[tag]
                
            result.append(padding_result)
        return result

    def _construct_batch(self, batch):
        batch = {
            tag: np.stack([sample[tag] for sample in batch])
            for tag in batch[0]
        }
        return batch
