#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

def has_NaN(line):
    for e in line:
        if math.isnan(e):
            return True
    return False

def get_bin_id(k, N_bins, size):
    return int(k//(size/N_bins))

def make_hist(d, N, N_bins):
    hist = []

    count = 0
    for i in tqdm(range(0, N)):
        k = np.random.randint(0, d.shape[0])
        if has_NaN(d[k]):
            count += 1
            hist.append(get_bin_id(k, N_bins, d.shape[0]))
    return hist, count

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, default=100,
                    help="Number of samples")
    parser.add_argument("-b", type=int, default=20,
                    help="Number of bins")
    parser.add_argument("-f", type=str, default='hist_NaN.png',
                    help="Hist plot filename")
    parser.add_argument("-d", type=str, default='data.hdf5',
                    help="Data filename")
    args = parser.parse_args()

    f = h5py.File(args.d, "r")
    d = f['field_evaluator']

    N = args.N
    N_bins = args.b
    print("Estimating proportion and hist of NaNs in {} lines:".format(d.shape[0]))
    hist, count = make_hist(d, N, N_bins)
    print("NaN proportion: {}/{}={}".format(count, N, count/N))

    plt.hist(hist, bins=N_bins, range=(0, int((d.shape[0]-1)//(d.shape[0]/N_bins)) ))

    plt.savefig(args.f)
    plt.show()
    f.close()

if __name__ == '__main__':
  main()
