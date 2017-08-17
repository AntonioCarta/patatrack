"""
Convert dataset to LMDB format.
LMDB is a memory-mapped database so it should be faster.
"""
import lmdb
import h5py
import pandas as pd
import argparse
import os
import dataset
from dataset import Dataset
import numpy as np

import sys
s = r'C:\Users\Antonio\Downloads\caffe\python'
sys.path.append(s)
import caffe
from caffe.io import array_to_datum

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()


if __name__ == '__main__':
    files = os.listdir(args.dir)
    new_db = lmdb.open(args.output, map_size=3 * 10 ** 9)

    for f in files:
        print("processing: " + f)
        data = Dataset([args.dir + '/' + f])
        X_hit, X_info, y = data.get_data(crop=True)
        with new_db.begin(write=True) as txn:
            # txn is a Transaction object
            N = X_hit.shape[0]
            for i in range(N):
                img_i = X_hit[i, :, :, :]
                img_i = img_i.transpose((2, 0, 1))
                el = np.hstack([img_i.flatten(), X_info[i, :].flatten()])
                el = np.reshape(el, (1, 1, -1))
                datum = array_to_datum(el, int(y[i, 1]))
                str_id = '{}_ID{:08}'.format(f, i)

                # The encode is only essential in Python 3
                txn.put(str_id.encode('ascii'), datum.SerializeToString())
    new_db.close()
