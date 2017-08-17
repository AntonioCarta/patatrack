import keras
import time
import json
from keras.models import load_model
import tensorflow as tf
import keras.backend as K
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from keras.models import model_from_json
from dataset import Dataset
import os


DEBUG = False
if DEBUG:
    print("DEBUG MODE!!!!!")

remote_data = "data/bal_data/"
remote_test_dir = "/eos/cms/store/cmst3/group/dehep/convPixels/TTBar_13TeV_PU35/"
debug_data = ['data/h5data/' + el for el in ['doublets.h5', 'doublets2.h5']]

log_dir = "models/cnn_doublet"
fname = "mini_model_swap_channel.json"  # load_best_in_dir(log_dir)
filename, file_extension = os.path.splitext(fname)
print("Testing model: " + filename)
with open(log_dir + '/' + fname) as f:
    model = model_from_json(json.load(f))
model.load_weights(log_dir + '/' + filename + '.h5')

model.summary()

print("loading test data...")
# Balanced data...
test_files = [remote_data + 'test/' + el for el in os.listdir(remote_data + 'test/')]
test_data = Dataset(test_files)
X_test_hit, X_test_info, y_test = test_data.get_data(crop=True)
y_pred_test = model.predict([X_test_hit, X_test_info])

X_test_hit = X_test_hit.astype(np.float32)
X_test_info = X_test_info.astype(np.float32)
y_test = y_test.astype(np.float32)
print("number of samples: {}".format(y_test.shape[0]))


num_output = 1
write_graph_def_ascii_flag = True
prefix_output_node_names_of_final_network = 'output_node'
out_folder = 'models/cnn_doublet/tf_model'
output_graph_name = 'constant_graph_weights.pb'

K.set_learning_phase(0)

print("!!! TEST STARTED !!!")
n_run = 100
batch_sizes = [2 ** x for x in range(20)]

sess = K.get_session()
with sess.as_default():
    for bs in batch_sizes:
        max_batch_from_data = int(np.floor(y_test.shape[0] / float(bs)))
        t_start = time.time()
        for i in range(n_run):
            jj = i % max_batch_from_data
            res = model.output.eval(feed_dict={
                'hit_shape_input:0': X_test_hit[(jj*bs):(jj+1)*bs, :, :, :],
                'info_input:0': X_test_info[0:bs, :],
                'batch_normalization_1/keras_learning_phase:0': 0
            })
        t_end = time.time()
        tot_time = t_end - t_start
        fps = (n_run * float(bs)) / tot_time
        print("fps {}; batch size {}; time elapsed: {}".format(fps, bs, tot_time))
