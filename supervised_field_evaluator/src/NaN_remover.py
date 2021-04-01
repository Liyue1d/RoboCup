#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import h5py
from tqdm import tqdm
import time
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("block_size")
    parser.add_argument("-i", type=str, default='data.hdf5',
                    help="Data filename")
    parser.add_argument("-o", type=str, default='data_no_NaN.hdf5',
                    help="Data filename output")
    args = parser.parse_args()

    f = h5py.File(args.i, "r")
    f_new = h5py.File(args.o, "w")

    d = f['field_evaluator']
    d_new = f_new.create_dataset("field_evaluator", d.shape, maxshape=d.shape, dtype='d')

    block_size = int(args.block_size)
    print("block_size: ", block_size, "d.shape: ", d.shape)
    i = 0
    j = 0
    count_NaN = 0
    with tqdm(total=d.shape[0]) as pbar:
        while True:
            if i >= d.shape[0]:
                break

            end_slice = i+block_size

            if end_slice > d.shape[0]:
                end_slice = d.shape[0]

            nan_mat = np.isnan(d[i:end_slice])
            nan_mat = np.any(nan_mat, axis=1)
            count_NaN  += np.sum(nan_mat)
            nan_mat = np.logical_not(nan_mat)
            count = np.sum(nan_mat)

            d_new[j:j+count] = d[i:end_slice][nan_mat, :]
            j += count

            pbar.update(end_slice - i)
            i += block_size

    d_new.resize((j, d.shape[1]))
    print("d_new.shape", d_new.shape)
    print("count_NaN: ", count_NaN)

    f.close()
    f_new.close()

if __name__ == '__main__':
  main()
