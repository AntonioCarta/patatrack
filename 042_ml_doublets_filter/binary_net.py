#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""
Train a small multi-layer perceptron with fully connected layers on MNIST data.

This example has some command line arguments that enable different neon features.

Examples:

    python examples/mnist_mlp.py -b gpu -e 10

        Run the example for 10 epochs using the NervanaGPU backend

    python examples/mnist_mlp.py --eval_freq 1

        After each training epoch, process the validation/test data
        set through the model and display the cost.

    python examples/mnist_mlp.py --serialize 1 -s checkpoint.pkl

        After every iteration of training, dump the model to a pickle
        file named "checkpoint.pkl".  Changing the serialize parameter
        changes the frequency at which the model is saved.

    python examples/mnist_mlp.py --model_file checkpoint.pkl

        Before starting to train the model, set the model state to
        the values stored in the checkpoint file named checkpoint.pkl.

"""

from neon.callbacks.callbacks import Callbacks
from neon.data import MNIST
from neon.initializers import Gaussian, Uniform
from neon.layers import GeneralizedCost, Affine, BinaryAffine
from neon.models import Model
from neon.optimizers import GradientDescentMomentum, ShiftAdaMax, ShiftSchedule, MultiOptimizer
from neon.transforms import Rectlin, Sign, Logistic, CrossEntropyBinary, Misclassification, SquareHingeLoss, Identity
from neon.util.argparser import NeonArgparser
from neon import logger as neon_logger
from neon.data import ArrayIterator
import dataset
from dataset import Dataset
import numpy as np
import os

# parse the command line arguments
parser = NeonArgparser(__doc__)
args = parser.parse_args()

data_dir = 'data/bal_data/'

train_files = [data_dir + 'train/' + el for el in os.listdir(data_dir + 'train/')]
train_data = Dataset(train_files)
X_train_hit, X_train_info, y_train = train_data.get_data()

val_files = [data_dir + 'val/' + el for el in os.listdir(data_dir + 'val/')]
val_data = Dataset(val_files)
X_val_hit, X_val_info, y_val = train_data.get_data()

hit_flat = X_train_hit.reshape(X_train_hit.shape[0], -1)
X_train = np.hstack([X_train_info, hit_flat])

hit_flat = X_val_hit.reshape(X_train_hit.shape[0], -1)
X_val = np.hstack([X_val_info, hit_flat])

mu = np.mean(X_train, axis=0).reshape(1, -1)
s = np.std(X_train, axis=0).reshape(1, -1)
s[s == 0] = 1

X_train = (X_train - mu) / s
X_val = (X_val - mu) / s

# X = np.random.rand(10000, 100)
# y = np.sum(X, axis=1).reshape(-1, 1)
# y = np.hstack([y, y])

train_set = ArrayIterator(X=X_train, y=y_train, make_onehot=False)
val_set = ArrayIterator(X=X_val, y=y_val, make_onehot=False)

# setup weight initialization function
init = Uniform(-1, 1)

# setup layers
layers = [
    BinaryAffine(nout=4096, init=init, batch_norm=True, activation=Sign()),
    BinaryAffine(nout=4096, init=init, batch_norm=True, activation=Sign()),
    BinaryAffine(nout=4096, init=init, batch_norm=True, activation=Sign()),
    BinaryAffine(nout=2, init=init, batch_norm=True, activation=Identity())
]

# setup cost function as Square Hinge Loss
cost = GeneralizedCost(costfunc=SquareHingeLoss())

# setup optimizer
LR_start = 1.65e-2


def ShiftAdaMax_with_Scale(LR=1):
    return ShiftAdaMax(learning_rate=LR_start * LR, schedule=ShiftSchedule(2, shift_size=1))


optimizer = MultiOptimizer({
    'default': ShiftAdaMax_with_Scale(),
    'BinaryLinear_0': ShiftAdaMax_with_Scale(57.038),
    'BinaryLinear_1': ShiftAdaMax_with_Scale(73.9008),
    'BinaryLinear_2': ShiftAdaMax_with_Scale(73.9008),
    'BinaryLinear_3': ShiftAdaMax_with_Scale(52.3195)
})

# initialize model object
bnn = Model(layers=layers)

# configure callbacks
callbacks = Callbacks(bnn, eval_set=val_set, **args.callback_args)

# run fit
bnn.fit(train_set, optimizer=optimizer, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
print('Misclassification error = %.1f%%' % (bnn.eval(val_set, metric=Misclassification())*100))

bnn.save_params("bin_model/final_model.prm")
