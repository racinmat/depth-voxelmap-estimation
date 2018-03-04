# encoding: utf-8

from tensorflow.python.platform import gfile
import tensorflow as tf
import Network
import time

from dataset import DataSet


def print_tensors_in_checkpoint_file(file_name, tensor_name, all_tensors):
    varlist = []
    reader = tf.pywrap_tensorflow.NewCheckpointReader(file_name)
    if all_tensors:
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in sorted(var_to_shape_map):
            # print("tensor_name: ", key)
            varlist.append(key)
    #        print(reader.get_tensor(key))
    # varlist.append(reader.get_tensor(key))
    elif not tensor_name:
        print(reader.debug_string().decode("utf-8"))
    else:
        print("tensor_name: ", tensor_name)
        print(reader.get_tensor(tensor_name))
    return varlist


with tf.Graph().as_default():
    checkpoint_name = r'D:\projekty\GTA-V-extractors\tensorflow-resnet\tensorflow-resnet-pretrained-20160509\ResNet-L152.ckpt'
    meta_file = r'D:\projekty\GTA-V-extractors\tensorflow-resnet\tensorflow-resnet-pretrained-20160509\ResNet-L152.meta'
    saver = tf.train.import_meta_graph(meta_file)

    a = print_tensors_in_checkpoint_file(file_name=checkpoint_name, all_tensors=True, tensor_name=None)
    print(a)
    b = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    sess = tf.Session()
    saver.restore(sess, checkpoint_name)

network = Network.Network()
dataset = DataSet(Network.BATCH_SIZE)
images, depths, invalid_depths = dataset.csv_inputs(Network.TRAIN_FILE)
logits = network.inference(images)
weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
print(weights)

