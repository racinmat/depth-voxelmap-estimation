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

    sess.run(tf.global_variables_initializer())

    mappings_prefix = {
        # 'scale1': 'conv1',
        'scale2/block1': ('resize1', True),  # first is name, second is shortcut present
        'scale2/block2': ('resize2-0', False),
        'scale2/block3': ('resize2-1', False),
        'scale3/block1': ('resize3', True),
        'scale3/block2': ('resize4-0', False),
        'scale3/block3': ('resize4-1', False),
        'scale3/block4': ('resize4-2', False),
        'scale3/block5': ('resize4-3', False),
        'scale3/block6': ('resize4-4', False),
        'scale3/block7': ('resize4-5', False),
        'scale3/block8': ('resize4-6', False),
        'scale4/block1': ('resize5', True),
        'scale4/block2': ('resize6-0', False),
        'scale4/block3': ('resize6-1', False),
        'scale4/block4': ('resize6-2', False),
        'scale4/block5': ('resize6-3', False),
        'scale4/block6': ('resize6-4', False),
        'scale4/block7': ('resize6-5', False),
        'scale4/block8': ('resize6-6', False),
        'scale4/block9': ('resize6-7', False),
        'scale4/block10': ('resize6-8', False),
        'scale4/block11': ('resize6-9', False),
        'scale4/block12': ('resize6-10', False),
        'scale4/block13': ('resize6-11', False),
        'scale4/block14': ('resize6-12', False),
        'scale4/block15': ('resize6-13', False),
        'scale4/block16': ('resize6-14', False),
        'scale4/block17': ('resize6-15', False),
        'scale4/block18': ('resize6-16', False),
        'scale4/block19': ('resize6-17', False),
        'scale4/block20': ('resize6-18', False),
        'scale4/block21': ('resize6-19', False),
        'scale4/block22': ('resize6-20', False),
        'scale4/block23': ('resize6-21', False),
        'scale4/block24': ('resize6-22', False),
        'scale4/block25': ('resize6-23', False),
        'scale4/block26': ('resize6-24', False),
        'scale4/block27': ('resize6-25', False),
        'scale4/block28': ('resize6-26', False),
        'scale4/block29': ('resize6-27', False),
        'scale4/block30': ('resize6-28', False),
        'scale4/block31': ('resize6-29', False),
        'scale4/block32': ('resize6-30', False),
        'scale4/block33': ('resize6-31', False),
        'scale4/block34': ('resize6-32', False),
        'scale4/block35': ('resize6-33', False),
        'scale4/block36': ('resize6-34', False),
        'scale5/block1': ('resize7', True),
        'scale5/block2': ('resize8-0', False),
        'scale5/block3': ('resize8-1', False),
    }

    mappings = {
        'a': 'conv2',
        'b': 'conv3',
        'c': 'conv4',
        'shortcut': 'conv5',
    }

    mappings_suffix = {
        'weights': 'weights',
        'beta': 'batch_norm/beta',
        'gamma': 'batch_norm/gamma',
    }
    print(b[0].shape)
    print(tf.get_default_graph().get_tensor_by_name("network/conv1/weights:0").shape)
    tf.assign(tf.get_default_graph().get_tensor_by_name("network/conv1/weights:0"),
              tf.get_default_graph().get_tensor_by_name("scale1/weights:0"))
    tf.assign(tf.get_default_graph().get_tensor_by_name("network/conv1/batch_norm/beta:0"),
              tf.get_default_graph().get_tensor_by_name("scale1/beta:0"))
    tf.assign(tf.get_default_graph().get_tensor_by_name("network/conv1/batch_norm/gamma:0"),
              tf.get_default_graph().get_tensor_by_name("scale1/gamma:0"))

    for prefix_from, (prefix_to, has_shortcut) in mappings_prefix.items():
        for name_from, name_to in mappings.items():
            if (not has_shortcut) and name_from == 'shortcut':
                continue
            for suffix_from, suffix_to in mappings_suffix.items():
                # all variables in my network are are in same scopre so I can identify them easily
                tf.assign(tf.get_default_graph().get_tensor_by_name("network/{}/{}/{}:0".format(prefix_to, name_to, suffix_to)),
                          tf.get_default_graph().get_tensor_by_name("{}/{}/{}:0".format(prefix_from, name_from, suffix_from)))

    print('everything assigned, running to apply these assignments')
    sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network'))     # so all weights are actually being assigned
    print('tensors runs, going to save them')
    new_saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network'))
    new_saver.save(sess, 'init-weights', global_step=0)
