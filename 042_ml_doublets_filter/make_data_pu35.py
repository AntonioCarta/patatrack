"""
Load all files in a directory (.gz), merge them in a single file
and split them in TRAIN / VAL / TEST data
"""
import gzip
import os
import numpy as np
import dataset
import pandas as pd

allfiles = ["1_" + str(i) + "_Dataset.npy" for i in range(1, 390)]

trainfiles = allfiles[:150]
valfiles = allfiles[150:250]
testfiles = allfiles[250:350]


COMPRESSED = False
dir = "data/TTBar_35_PU"

def create_numpy_data(files):
    all_data = []
    for fname in files:
        print("processsing: " + fname)
        #with gzip.open(dir + "/" + fname, 'rb') as f:
        data = np.load(dir + "/" + fname)
        all_data.append(data)
    data = np.vstack(all_data)
    return data


data_train = create_numpy_data(trainfiles)#create_numpy_data(fnames[:i_train])
data_val = create_numpy_data(valfiles)#create_numpy_data(fnames[i_train:i_val])
data_test = create_numpy_data(testfiles)#create_numpy_data(fnames[i_val:])
data_debug = data_train[:500, :]


def balance_data(data, max_ratio=0.5, verbose=True):
    """ Balance the data. """ 
    df = pd.DataFrame(data, columns=dataset.datalabs)
    data_neg = df[df[dataset.target_lab] == 0.0]
    data_pos = df[df[dataset.target_lab] != 0.0]

    n_pos = data_pos.shape[0]
    n_neg = data_neg.shape[0]
    if verbose:
        print("Number of negatives: " + str(n_neg))
        print("Number of positive: " + str(n_pos))
        print("ratio: " + str(n_neg / n_pos))
    
    data_neg = data_neg.sample(n_pos)
    balanced_data = pd.concat([data_neg, data_pos])
    balanced_data = balanced_data.sample(frac=1)  # Shuffle the dataset
    return balanced_data

np.save('data/train_pu35_balanced.npy', balance_data(data_train))
np.save('data/val_pu35_balanced.npy', balance_data(data_val))
np.save('data/test_pu35.npy', data_test)
np.save('data/debug_pu35.npy', data_debug)

    