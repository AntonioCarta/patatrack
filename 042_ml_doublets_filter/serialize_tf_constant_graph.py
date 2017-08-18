"""
    Extract tensorflow graph from keras model.
    - freeze the graph for efficiency
    - save in .pb format

    the graph weights can be quantized using graph_transform tool as shown:
        https://www.tensorflow.org/performance/quantization

    Using the quantized graphs we can perform inference at lower precision
    faster and (hopefully) without losing too much precision.
"""
import tensorflow as tf
import os
import json
from keras.models import model_from_json
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
import keras.backend as K

log_dir = "models/cnn_doublet"
fname = "mini_model_all_data_single_channel.json"  # load_best_in_dir(log_dir)
filename, file_extension = os.path.splitext(fname)
print("Testing model: " + filename)
with open(log_dir + '/' + fname) as f:
    model = model_from_json(json.load(f))
model.load_weights(log_dir + '/' + filename + '.h5')


num_output = 1
write_graph_def_ascii_flag = True
prefix_output_node_names_of_final_network = 'output_node'
out_folder = 'models/quantize'
output_graph_name = 'mini_model_single_channel_constant_graph_float32.pb'


K.set_learning_phase(0)
net_model = model

pred = [None] * num_output
pred_node_names = [None] * num_output
for i in range(num_output):
    pred_node_names[i] = prefix_output_node_names_of_final_network + str(i)
    pred[i] = tf.identity(net_model.output[i], name=pred_node_names[i])
print('output nodes names are: ', pred_node_names)

sess = K.get_session()
if write_graph_def_ascii_flag:
    f = 'only_the_graph_def.pb.ascii'
    tf.train.write_graph(sess.graph.as_graph_def(), out_folder, f, as_text=True)
    print('saved the graph definition in ascii format at: ', os.path.join(out_folder, f))

constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
graph_io.write_graph(constant_graph, out_folder, output_graph_name, as_text=False)
print('saved the constant graph (ready for inference) at: ', os.path.join(out_folder, output_graph_name))
