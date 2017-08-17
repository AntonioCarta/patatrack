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
from neon.backends import gen_backend
from dataset import Dataset
import numpy as np
import os
import time

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

print("!!! TEST STARTED !!!")
for bs in [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
    be = gen_backend('gpu', batch_size=bs)
    bnn = Model("bin_model/final_model.prm")
    tries = 3
    t_start = time.time()
    for i in range(tries):
        out = bnn.get_outputs(train_set)
    t_end = time.time()

    tot_time = t_end - t_start
    n = X_train.shape[0]
    fps = tries * n / tot_time
    print("fps {}; batch size {}; time elapsed: {}".format(fps, bs, tot_time))
