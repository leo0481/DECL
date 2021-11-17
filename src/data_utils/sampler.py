#!/usr/local/anaconda3/envs/torch-1.0-py3/bin/python
# coding=utf-8
# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=import-error

import numpy as np
import scipy.sparse as sp


class Sampler:

    def __init__(self, flags_obj, lil_record, dok_record, neg_sample_rate):
        self.name = flags_obj.name + '_sampler'
        self.lil_record = lil_record
        self.record = list(dok_record.keys())
        self.neg_sample_rate = neg_sample_rate
        self.n_user = lil_record.shape[0]
        self.n_item = lil_record.shape[1]

    def sample(self, index, **kwargs):
        raise NotImplementedError

    def get_pos_user_item(self, index):
        user = self.record[index][0]
        pos_item = self.record[index][1]
        return user, pos_item

    def generate_negative_samples(self, user, **kwargs):
        negative_samples = np.full(self.neg_sample_rate, -1, dtype=np.int64)

        user_pos = self.lil_record.rows[user]
        for count in range(self.neg_sample_rate):

            item = np.random.randint(self.n_item)
            while item in user_pos or item in negative_samples:
                item = np.random.randint(self.n_item)
            negative_samples[count] = item
        return negative_samples


class PointSampler(Sampler):

    def sample(self, index):
        user, pos_item = self.get_pos_user_item(index)

        users = np.full(1 + self.neg_sample_rate, user, dtype=np.int64)
        items = np.full(1 + self.neg_sample_rate, pos_item, dtype=np.int64)
        labels = np.zeros(1 + self.neg_sample_rate, dtype=np.float32)

        labels[0] = 1.0

        negative_samples = self.generate_negative_samples(user)
        items[1:] = negative_samples[:]

        return users, items, labels


class PairSampler(Sampler):

    def sample(self, index):
        user, pos_item = self.get_pos_user_item(index)

        users = np.full(self.neg_sample_rate, user, dtype=np.int64)
        items_pos = np.full(self.neg_sample_rate, pos_item, dtype=np.int64)
        items_neg = self.generate_negative_samples(user)

        return users, items_pos, items_neg


class DICESampler(Sampler):

    def __init__(self, flags_obj, lil_record, dok_record, neg_sample_rate, popularity, margin=10, pool=10):
        super(DICESampler, self).__init__(flags_obj, lil_record, dok_record, neg_sample_rate)
        self.popularity = popularity
        self.margin = margin
        self.pool = pool

    def adapt(self, epoch, decay):
        self.margin = self.margin * decay

    def generate_negative_samples(self, user, pos_item):
        negative_samples = np.full(self.neg_sample_rate, -1, dtype=np.int64)
        mask_type = np.full(self.neg_sample_rate, False, dtype=np.bool)

        user_pos = self.lil_record.rows[user]

        item_pos_pop = self.popularity[pos_item]

        pop_items = np.nonzero(self.popularity > item_pos_pop + self.margin)[0]
        pop_items = pop_items[np.logical_not(np.isin(pop_items, user_pos))]
        num_pop_items = len(pop_items)

        unpop_items = np.nonzero(self.popularity < item_pos_pop - 10)[0]
        unpop_items = np.nonzero(self.popularity < item_pos_pop / 2)[0]
        unpop_items = unpop_items[np.logical_not(np.isin(unpop_items, user_pos))]
        num_unpop_items = len(unpop_items)

        if num_pop_items < self.pool:
            for count in range(self.neg_sample_rate):
                index = np.random.randint(num_unpop_items)
                item = unpop_items[index]
                while item in negative_samples:
                    index = np.random.randint(num_unpop_items)
                    item = unpop_items[index]
                negative_samples[count] = item
                mask_type[count] = False
        elif num_unpop_items < self.pool:
            for count in range(self.neg_sample_rate):
                index = np.random.randint(num_pop_items)
                item = pop_items[index]
                while item in negative_samples:
                    index = np.random.randint(num_pop_items)
                    item = pop_items[index]

                negative_samples[count] = item
                mask_type[count] = True
        else:
            for count in range(self.neg_sample_rate):
                if np.random.random() < 0.5:
                    index = np.random.randint(num_pop_items)
                    item = pop_items[index]
                    while item in negative_samples:
                        index = np.random.randint(num_pop_items)
                        item = pop_items[index]
                    negative_samples[count] = item
                    mask_type[count] = True
                else:
                    index = np.random.randint(num_unpop_items)
                    item = unpop_items[index]
                    while item in negative_samples:
                        index = np.random.randint(num_unpop_items)
                        item = unpop_items[index]

                    negative_samples[count] = item
                    mask_type[count] = False

        return negative_samples, mask_type

    def sample(self, index):
        user, pos_item = self.get_pos_user_item(index)
        users = np.full(self.neg_sample_rate, user, dtype=np.int64)
        items_pos = np.full(self.neg_sample_rate, pos_item, dtype=np.int64)
        items_neg, mask_type = self.generate_negative_samples(user, pos_item=pos_item)

        return users, items_pos, items_neg, mask_type
