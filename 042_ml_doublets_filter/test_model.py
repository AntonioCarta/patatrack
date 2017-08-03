# flake8: noqa: E402, F401
import socket
if socket.gethostname() == 'cmg-gpu1080':
    print('locking only one GPU.')
    import setGPU

import os
import numpy as np
import keras
import json
import argparse
from keras.models import model_from_json
from sklearn.metrics import roc_auc_score
from dataset import Dataset


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default=None)
parser.add_argument('--pred_folder', type=str, default='data/pred/')
parser.add_argument('--target_folder', type=str, default='data/target/')
args = parser.parse_args()

if args.model_name == None:
    args.model_name == input()

def streaming_prediction(model, files, output_folder, filters=[]):
    labs, preds = [], []
    for f in files:
        data_chunk = Dataset([f])
        for fil in filters:
            data_chunk.filter(fil[0], fil[1])
        X_chunk_hit, X_chunk_info, y_chunk = data_chunk.get_layer_map_data()
        y_pred = model.predict([X_chunk_hit, X_chunk_info])
        print("Saving data...")

        fname = os.path.basename(f)
        ftarget = args.target_folder + fname + "_target.npy"
        if not os.path.isfile(ftarget):
            np.save(ftarget, y_chunk)
        fpred = args.pred_folder + fname + "_pred.npy"
        if not os.path.isfile():
            np.save(fpred, y_pred)

if __name__ == '__main__':
    filename, file_extension = os.path.splitext(args.model_name)
    
    print("Loading model: " + filename)
    with open(filename + '.json') as f:
        model = model_from_json(json.load(f))
        model.load_weights(filename + '.h5')
    
    print("Computing model predictions...")
    remote_data = "data/bal_data/"
    remote_test_dir = "/eos/cms/store/cmst3/group/dehep/convPixels/TTBar_13TeV_PU35/"
    test_files = [remote_test_dir + el for el in os.listdir(remote_data + 'test/')]

    streaming_prediction(model, test_files, [('isFlippedIn', 1.0), ('isFlippedOut', 1.0)])

    print("Computing AUC...")
    labs, preds = [], []
    for f in os.listdir(args.target_folder):
        a = np.load(args.target_folder + f)
        labs.append(a)
    for f in os.listdir(args.pred_folder):
        a = np.load(args.pred_folder + f)
        preds.append(a)
    y_true = np.vstack(labs)
    y_pred = np.vstack(preds)
    print("AUC score: " + str(roc_auc_score(y_true, y_pred)))
