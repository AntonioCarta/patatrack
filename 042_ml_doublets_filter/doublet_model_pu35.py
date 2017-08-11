"""
Doublet model with hit shapes and info features.
"""
import socket

if socket.gethostname() == 'cmg-gpu1080':
    print('locking only one GPU.')
    import setGPU

import argparse
import keras
import dataset
import datetime
import json
import tempfile
import os
import numpy as np
from dataset import Dataset
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Reshape, Dropout, BatchNormalization
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.constraints import max_norm
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from model_architectures import *


DEBUG = os.name == 'nt'  # DEBUG on laptop
PU35 = True
if PU35:
    print("training with PU35 dataset.")
else:
    print("training with NOPU dataset.")

if DEBUG:
    print("DEBUG mode")

t_now = '{0:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())
# Model configuration
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=500 if not DEBUG else 3,
                    help='number of epochs')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.5)
parser.add_argument('--patience', type=int, default=15)
parser.add_argument('--log_dir', type=str, default="models/cnn_doublet")
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--maxnorm', type=float, default=10.)
parser.add_argument('--verbose', type=int, default=1)
args = parser.parse_args()

if args.name is None:
    args.name = input('model name: ')

log_dir_tf = args.log_dir + '/' + args.name
remote_data = '/eos/cms/store/cmst3/group/dehep/convPixels/clean/' 

print("Loading data...")
data_dir = 'data/'
train_fname = data_dir + 'train_balanced.npy' if not DEBUG else 'data/NOPU/debug.npy'
val_fname = data_dir + 'val.npy' if not DEBUG else 'data/NOPU/debug.npy'
test_fname = data_dir + 'test.npy' if not DEBUG else 'data/NOPU/debug.npy'

train_data = Dataset(dataset.trainfiles[:100]) if not (DEBUG or not PU35) else Dataset(train_fname)
val_data = Dataset(dataset.trainfiles[100:150]) if not (DEBUG or not PU35) else Dataset(val_fname)
test_data = Dataset(dataset.trainfiles[150:200]) if not (DEBUG or not PU35) else Dataset(test_fname)

train_data = train_data.filter('isFlippedIn', 1.0).filter('isFlippedOut', 1.0) \
    .filter('detSeqIn', 0.0).filter('detSeqOut', 1.0).balance_data()
val_data = val_data.filter('isFlippedIn', 1.0).filter('isFlippedOut', 1.0) \
    .filter('detSeqIn', 0.0).filter('detSeqOut', 1.0).balance_data()
test_data = test_data.filter('isFlippedIn', 1.0).filter('isFlippedOut', 1.0) \
    .filter('detSeqIn', 0.0).filter('detSeqOut', 1.0)

X_hit, X_info, y = train_data.get_data()
X_val_hit, X_val_info, y_val = val_data.get_data()
X_test_hit, X_test_info, y_test = test_data.get_data()

train_input_list = [X_hit, X_info] # [X_hit[:,:,:,:4], X_hit[:,:,:,4:], X_info]
val_input_list = [X_val_hit, X_val_info] # [X_val_hit[:,:,:,:4], X_val_hit[:,:,:,4:], X_val_info]
test_input_list = [X_test_hit, X_test_info] # [X_test_hit[:,:,:,:4], X_test_hit[:,:,:,4:], X_test_info]

model = small_doublet_model(args, train_input_list[0].shape[-1])

if args.verbose:
    model.summary()

print('Training')
# Save the best model during validation and bail out of training early if we're not improving
_, tmpfn = tempfile.mkstemp()
callbacks = [
    EarlyStopping(monitor='val_loss', patience=args.patience), 
    ModelCheckpoint(tmpfn, save_best_only=True, save_weights_only=True), 
    TensorBoard(log_dir=log_dir_tf, histogram_freq=0, write_graph=True, write_images=True)
]
model.fit(train_input_list, y, batch_size=args.batch_size, epochs=args.n_epochs, shuffle=True,
    validation_data=(val_input_list, y_val), callbacks=callbacks, verbose=args.verbose)

# Restore the best found model during validation
model.load_weights(tmpfn)

loss, acc = model.evaluate(test_input_list, y_test, batch_size=args.batch_size)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))


fname = args.log_dir + "/" + args.name 
print("saving model " + fname)
model.save_weights(fname + ".h5", overwrite=True)
with open(fname + ".json", "w") as outfile:
    json.dump(model.to_json(), outfile)

