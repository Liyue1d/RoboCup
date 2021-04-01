#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from contextlib import ContextDecorator

import h5py
import pickle
import os.path

import numpy as np
from tqdm import tqdm

# from sklearn.preprocessing import normalize

# Dataset interface ------------------------------------------------------------
class Dataset(ContextDecorator):

    def __enter__(self):
        return self

    def close(self):
        raise NotImplementedError("Datasets should implement close method")

    def __exit__(self, *exc):
        self.close()
        return False

class SplitDataset:
    def __init__(self):
        pass

    def next_batch(self, size):
        raise NotImplementedError("SplitDatasets should implement next_batch method")

    def all_dataset(self):
        raise NotImplementedError("SplitDatasets should implement all_dataset method")


# H5pyDatasetInMemory ----------------------------------------------------------
class H5pyDatasetInMemory(Dataset):
    def __init__(self, filename):
        self.f = h5py.File(filename, "r")


        data = self.f['field_evaluator'][:]# TODO don't load everything in RAM

        # Preprocessing ********************
        labels = np.matrix(data[:, 1])
        labels[labels > 160] = 160
        labels[labels < -160] = -60
        data = data[:, 2:]
        print("data.shape", data.shape)
        print("labels.shape", labels.shape)
        # data = normalize(data)
        data = np.concatenate((data, labels.T), axis=1)
        print("data.shape", data.shape)
        # TODO preprocessing? (or do before in another program?): outliers, normalize

        self.n_inputs = data.shape[1]-1
        # TODO flags partitions sizes
        size = data.shape[0]
        train_partition = (0, int(0.6*size))
        test_partition = (train_partition[1], train_partition[1]+int(0.3*size))
        val_partition = (test_partition[1], test_partition[1]+int(0.1*size))
        self.train = SplitDatasetInMemory(data, train_partition)
        self.test = SplitDatasetInMemory(data, test_partition)
        self.val = SplitDatasetInMemory(data, val_partition)

    def close(self):
        self.f.close()

class SplitDatasetInMemory:
    def __init__(self, data, partition):
        self.data = data
        self.partition = partition
        self.num_examples = self.partition[1]-self.partition[0]
        self.last_end = 0

        # TODO Queue loading

    def next_batch(self, size):
        if self.last_end == self.partition[0]:
            np.random.shuffle(self.data[self.partition[0]:self.partition[1]])

        start = self.last_end
        end = self.last_end + size
        self.last_end = end
        if end > self.partition[1]:
            end = self.partition[1]
            self.last_end = self.partition[0]#Return to beginning of split dataset

        return self.data[start:end][:, 0:-1], np.array(self.data[start:end][:, -1])

    def all_dataset(self):
        return self.data[self.partition[0]:self.partition[1], 0:-1], np.array(self.data[self.partition[0]:self.partition[1], -1])

# H5pyDataset ----------------------------------------------------------
class H5pyDataset(Dataset):
    def __init__(self, filename, n_sample_normalization, queue_size):
        self.f = h5py.File(filename, "r")

        # TODO save normalization in h5py? (don't break stuff)

        d = self.f['field_evaluator']

        if not os.path.isfile('save_normalization'):
            print("Computing normalization")
            s = np.random.randint(0, d.shape[0], n_sample_normalization)
            s = np.sort(s)
            l = []
            for i in tqdm(range(0, s.shape[0])):
                v = d[s[i]]
                if not abs(v[1]) > 10000:
                    l.append(v)
            l = np.array(l)

            m = np.mean(l, axis=0)
            v = np.std(l, axis=0)
            v[v==0] = 1.0
            min_ = np.min(l[:, 1])
            max_ = np.max(l[:, 1])
            with open('save_normalization', 'wb') as fn:
                pickle.dump((m, v, min_, max_), fn)

            del l
        else:
            print("Loading normalization")
            with open('save_normalization', 'rb') as fn:
                (m, v, min_, max_) = pickle.load(fn)
            print(min_, max_)


        def normalizer(data):
            data = np.copy(data)
            # Remove outliers
            data[:, 1][data[:, 1] < min_] = min_
            data[:, 1][data[:, 1] > max_] = max_

            # print("v", v[2:])
            # Normalize
            data[:, 2:] = (data[:, 2:]-m[2:])/v[2:]

            return data

        self.n_inputs = d.shape[1]-2

        # TODO flags partitions sizes
        size = d.shape[0]
        train_partition = (0, int(0.6*size))
        test_partition = (train_partition[1], train_partition[1]+int(0.3*size))
        val_partition = (test_partition[1], test_partition[1]+int(0.1*size))
        self.train = SplitDatasetQueue(d, train_partition, normalizer, queue_size)
        self.test = SplitDatasetQueue(d, test_partition, normalizer, queue_size)
        self.val = SplitDatasetQueue(d, val_partition, normalizer, queue_size)

    def close(self):
        self.f.close()

class SplitDatasetQueue:
    def __init__(self, data, partition, normalizer, queue_size):
        self.data = data
        self.partition = partition
        self.num_examples = self.partition[1]-self.partition[0]

        self.normalizer = normalizer
        self.queue_size = queue_size

        # TODO Queue loading
        if self.num_examples < self.queue_size:
            raise Exception('Queue size must be smaller than split dataset size')

        self.queue = self.data[self.partition[0]:self.partition[0]+queue_size]
        self.queue = self.normalizer(self.queue)
        self.last_end = self.partition[0]+queue_size

    def next_batch(self, size):
        # In next batch:
        # TODO sample batch_size samples in queue
        # TODO get next batch_size samples from file and normalize

        pass
        s = np.random.randint(0, self.queue.shape[0], size)
        next_batch_samples = np.copy(self.queue[s])
        self.queue[s] = 100000000000000000#TODO check

        # Get samples from file and put in queue
        start = self.last_end
        end = self.last_end+size
        if end > self.partition[1]:
            l1 = self.data[start:self.partition[1]]
            l2 = self.data[self.partition[0]:self.partition[0]+(end-self.partition[1])]
            new_data = np.concatenate([l1, l2])
            self.last_end = self.partition[0]+(end-self.partition[1])
        else:
            new_data = self.data[start:end]
            self.last_end = end

        # print(new_data[0:1])
        self.queue[s] = self.normalizer(new_data)#TODO check, no need flor clone in normalizer here
        # print(self.queue[0:1])
        # exit(0)

        return next_batch_samples[:, 2:], np.matrix(next_batch_samples[:, 1]).T

    def all_dataset(self):
        return self.queue[:, 2:], np.matrix(self.queue[:, 1]).T
        # return self.data[self.partition[0]:self.partition[1], 0:-1], np.array(self.data[self.partition[0]:self.partition[1], -1])
#
