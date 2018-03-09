# encoding: utf-8

from datetime import datetime
import numpy as np
import tensorflow as tf

import dataset
import train_operation
from dataset import DataSet
import train_operation as op
import os
import time
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope

current_time = time.strftime("%Y-%m-%d--%H-%M-%S", time.gmtime())

# these weights are from resnet: https://github.com/ry/tensorflow-resnet/blob/master/resnet.py
BN_DECAY = 0.9997
BN_EPSILON = 1e-3
CONV_WEIGHT_DECAY = 4e-5
CONV_WEIGHT_STDDEV = 0.1

MAX_EPOCHS = int(1e7)
LOG_DEVICE_PLACEMENT = False
BATCH_SIZE = 8
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
COARSE_DIR = "coarse"
PREDICT_DIR = os.path.join('predict', current_time)
CHECKPOINT_DIR = os.path.join('checkpoint', current_time)  # Directory name to save the checkpoints
LOGS_DIR = 'logs'
GPU_IDX = [0]
WEIGHTS_REGULARIZER = slim.l2_regularizer(CONV_WEIGHT_DECAY)

class Network(object):

    def __init__(self):
        self.sess = None
        self.saver = None
        self.x = None
        self.y = None
        self.y_invalid = None
        self.images = None
        self.images_test = None
        self.depths = None
        self.depths_test = None
        self.invalid_depths = None
        self.invalid_depths_test = None
        self.depth_bins = None
        self.depth_bins_test = None

        # GPU settings
        self.config = tf.ConfigProto(log_device_placement=LOG_DEVICE_PLACEMENT)
        self.config.gpu_options.allow_growth = False
        self.config.gpu_options.allocator_type = 'BFC'
        devices_environ_var = 'CUDA_VISIBLE_DEVICES'
        if devices_environ_var in os.environ:
            available_devices = os.environ[devices_environ_var].split(',')
            if len(available_devices):
                if isinstance(GPU_IDX, list):
                    os.environ[devices_environ_var] = ', '.join([available_devices[gpu] for gpu in GPU_IDX])
                else:
                    gpu = GPU_IDX
                    os.environ[devices_environ_var] = available_devices[gpu]

    def resize_layer(self, scope_name, inputs, small_size, big_size, stride=1, rate=1):
        with arg_scope([layers.conv2d], rate=rate):
            with tf.variable_scope(scope_name) as scope:
                conv1 = slim.conv2d(inputs, num_outputs=small_size, scope='conv2', kernel_size=1, stride=stride,
                                    activation_fn=tf.nn.relu,
                                    )

                conv1 = slim.conv2d(conv1, num_outputs=small_size, scope='conv3', kernel_size=3, stride=1,
                                    activation_fn=tf.nn.relu,
                                    )

                conv1 = slim.conv2d(conv1, num_outputs=big_size, scope='conv4', kernel_size=1, stride=1,
                                    activation_fn=None,
                                    )

                conv1b = slim.conv2d(inputs, num_outputs=big_size, scope='conv5', kernel_size=1, stride=stride,
                                     activation_fn=None,
                                     )

                # concat
                conv1 = conv1 + conv1b
                conv1 = tf.nn.relu(conv1, 'relu')

                return conv1

    def non_resize_layer(self, scope_name, inputs, small_size, big_size, rate=1):
        with arg_scope([layers.conv2d], rate=rate):
            with tf.variable_scope(scope_name) as scope:
                conv1 = slim.conv2d(inputs, num_outputs=small_size, scope='conv2', kernel_size=1, stride=1,
                                    activation_fn=tf.nn.relu,
                                    )
                conv1 = slim.conv2d(conv1, num_outputs=small_size, scope='conv3', kernel_size=3, stride=1,
                                    activation_fn=tf.nn.relu,
                                    )
                conv1 = slim.conv2d(conv1, num_outputs=big_size, scope='conv4', kernel_size=1, stride=1,
                                    activation_fn=None,
                                    )

                # concat
                conv1 = conv1 + inputs
                conv1 = tf.nn.relu(conv1, 'relu')

                return conv1

    def initialize_by_resnet(self):
        # I initialize only trainable variables, not others. Now is unified saving and restoring
        loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='network'))
        loader.restore(self.sess, 'init-weights/resnet')
        print('weights initialized')

    def inference(self):
        batch_norm_params = {
            'decay': BN_DECAY,  # also known as momentum, they are the same
            'updates_collections': None,
            'epsilon': BN_EPSILON,
            'scale': True,
            'scope': 'batch_norm',
        }
        with arg_scope([layers.conv2d, layers.conv2d_transpose],
                       normalizer_fn=layers.batch_norm,
                       normalizer_params=batch_norm_params,
                       weights_initializer=layers.xavier_initializer(uniform=False),
                       biases_initializer=tf.constant_initializer(0.1),
                       weights_regularizer=WEIGHTS_REGULARIZER
                       ):
            with tf.variable_scope('network') as scope:
                self.x = tf.placeholder(tf.float32, shape=[None, dataset.IMAGE_HEIGHT, dataset.IMAGE_WIDTH, 3],
                                        name='x')

                conv = slim.conv2d(self.x, num_outputs=64, scope='conv1', kernel_size=7, stride=2,
                                   activation_fn=tf.nn.relu)
                print("conv1")
                print(conv)

                max1 = slim.max_pool2d(conv, kernel_size=3, stride=2, scope='maxpool1')

                conv = self.resize_layer("resize1", max1, small_size=64, big_size=256)
                print("conv2")
                print(conv)

                for i in range(2):
                    conv = self.non_resize_layer("resize2-" + str(i), conv, small_size=64, big_size=256)

                conv = self.resize_layer("resize3", conv, small_size=128, big_size=512, stride=2)

                l1concat = conv
                print("l1concat")
                print(l1concat)

                for i in range(7):
                    conv = self.non_resize_layer("resize4-" + str(i), conv, small_size=128, big_size=512)

                l2concat = conv
                print("l2concat")
                print(l2concat)

                conv = self.resize_layer("resize5", conv, small_size=256, big_size=1024, rate=2)

                l3concat = conv
                print("l3concat")
                print(l3concat)

                for i in range(35):
                    conv = self.non_resize_layer("resize6-" + str(i), conv, small_size=256, big_size=1024, rate=2)

                l4concat = conv
                print("l4concat")
                print(l4concat)

                conv = self.resize_layer("resize7", conv, small_size=512, big_size=2048, rate=4)

                l5concat = conv
                print("l5concat")
                print(l5concat)

                for i in range(2):
                    conv = self.non_resize_layer("resize8-" + str(i), conv, small_size=512, big_size=2048, rate=4)

                l6concat = conv
                print("l6concat")
                print(l6concat)

                conv = tf.concat([l1concat, l2concat, l3concat, l4concat, l5concat, l6concat], axis=3)

                conv = tf.layers.dropout(conv, rate=0.5)

                conv = slim.conv2d(conv, num_outputs=dataset.DEPTH_DIM + 1, scope='convFinal', kernel_size=3, stride=1,
                                   normalizer_fn=None, activation_fn=None)

                conv = slim.conv2d_transpose(conv, num_outputs=dataset.DEPTH_DIM + 1, kernel_size=8, stride=4,
                                             normalizer_fn=None, activation_fn=None, scope='deconvFinal')

                return conv

    def loss(self, logits):
        H = dataset.TARGET_HEIGHT
        W = dataset.TARGET_WIDTH
        # size is depth dim + 1, because 1 layer is for too distant points, outside of desired area
        self.y = tf.placeholder(tf.float32, shape=[None, H, W, dataset.DEPTH_DIM + 1], name='y')
        self.y_invalid = tf.placeholder(tf.float32, shape=[None, H, W, 1], name='y_invalid')

        print('labels shape:', self.y.shape)
        print('logits shape:', logits.shape)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits))

        tf.summary.scalar("cost", cost)

        return cost

    @staticmethod
    def bins_to_depth(depth_bins):
        weights = np.array(range(dataset.DEPTH_DIM)) * dataset.Q + np.log(dataset.D_MIN)
        sth = tf.expand_dims(tf.constant(weights, dtype=tf.float32), 0)
        sth = tf.expand_dims(sth, 0)
        sth = tf.expand_dims(sth, 0)
        mask = tf.tile(sth, [BATCH_SIZE, dataset.TARGET_HEIGHT, dataset.TARGET_WIDTH, 1])
        depths_bins_without_last = depth_bins[:, :, :, 0:dataset.DEPTH_DIM]
        # depths_bins_without_last = tf.slice(depth_bins, begin=[0, 0, 0, 0], size=[-1, -1, -1, dataset.DEPTH_DIM])  # stripping away the last layer, with not valid depth, no slicing in other dimensions
        mask_multiplied = tf.multiply(mask, tf.cast(depths_bins_without_last, dtype=tf.float32))
        mask_multiplied_sum = tf.reduce_sum(mask_multiplied, axis=3)
        depth = tf.exp(mask_multiplied_sum)
        depth = tf.expand_dims(depth, 3)
        return depth

    def prepare(self):
        data_set = DataSet(BATCH_SIZE)
        data_set.load_params(TRAIN_FILE)
        global_step = tf.Variable(0, trainable=False)
        self.images, self.depths, self.depth_bins, self.invalid_depths = data_set.csv_inputs(TRAIN_FILE)
        self.images_test, self.depths_test, self.depth_bins_test, self.invalid_depths_test = data_set.csv_inputs(
            TEST_FILE)
        estimated_depths = self.inference()
        loss = self.loss(estimated_depths)
        train_op = op.train(loss, global_step, BATCH_SIZE)
        self.saver = tf.train.Saver()  # saver must be initialized after network is set up

        # adding trainable weights to tensorboard
        for var in tf.trainable_variables():
            # print(var.op.name)
            tf.summary.histogram(var.op.name, var)
        # trainable_vars = slim.get_variables(scope='network', collection=tf.GraphKeys.TRAINABLE_VARIABLES)
        # for variable in trainable_vars:
        #     name = variable.name.split(':', 1)[0]
        #     tf.summary.histogram(name, variable)

        estimated_depths_images = self.bins_to_depth(estimated_depths)

        tf.summary.image('input_images', self.images)
        tf.summary.image('ground_truth_depths', self.depths)
        tf.summary.image('predicted_depths', estimated_depths_images)
        # this is last layer, need to expand dim, so the tensor is in shape [batch size, height, width, 1]
        for i in range(0, dataset.DEPTH_DIM, 20):
            tf.summary.image('predicted_layer_'+str(i), tf.expand_dims(estimated_depths[:, :, :, i], 3))

        tf.summary.image('predicted_invalid', tf.expand_dims(estimated_depths[:, :, :, dataset.DEPTH_DIM], 3))

        return data_set, loss, estimated_depths, train_op, estimated_depths_images

    def train(self):
        with tf.Graph().as_default():
            data_set, loss, estimated_depths, train_op, estimated_depths_images = self.prepare()

            # Session
            with tf.Session(config=self.config) as self.sess:
                self.sess.run(tf.global_variables_initializer())
                self.initialize_by_resnet()
                # parameters
                summary = tf.summary.merge_all()  # merge all summaries to dump them for tensorboard
                writer = tf.summary.FileWriter(os.path.join(LOGS_DIR, current_time), self.sess.graph)

                for variable in tf.trainable_variables():
                    variable_name = variable.name
                    print("parameter: %s" % variable_name)
                    if variable_name.find("/") < 0 or variable_name.count("/") != 1:
                        continue

                # train
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

                test_predicted_depths = None
                images_test = None

                index = 0
                num_batches_per_epoch = int(float(train_operation.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN) / BATCH_SIZE)
                for epoch in range(MAX_EPOCHS):
                    for i in range(num_batches_per_epoch):
                        # sending images to sess.run so new batch is loaded
                        images, depths, invalid_depths = self.sess.run(
                            [self.images, self.depth_bins, self.invalid_depths])
                        # training itself
                        _, loss_value, predicted_depths, summary_str = self.sess.run(
                            [train_op, loss, estimated_depths_images, summary],
                            feed_dict={
                                self.x: images,
                                self.y: depths,
                                self.y_invalid: invalid_depths,
                            }
                        )
                        if index % 10 == 0:
                            # updating summary
                            summary_str = self.sess.run(
                                summary,
                                feed_dict={
                                    self.x: images,
                                    self.y: depths,
                                    self.y_invalid: invalid_depths,
                                }
                            )
                            writer.add_summary(summary_str, index)

                            # loading new test batch
                            images_test, depths_test, invalid_depths_test = self.sess.run(
                                [self.images_test, self.depth_bins_test, self.invalid_depths_test])

                            # testing itself
                            test_loss_value, test_predicted_depths = self.sess.run(
                                [loss, estimated_depths_images],
                                feed_dict={
                                    self.x: images_test,
                                    self.y: depths_test,
                                    self.y_invalid: invalid_depths_test,
                                }
                            )
                            print(
                                "%s: %d[epoch]: %d[iteration]: train loss %f" % (datetime.now(), epoch, i, loss_value))
                            print(
                                "%s: %d[epoch]: %d[iteration]: test loss %f" % (
                                datetime.now(), epoch, i, test_loss_value))
                            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                        if index % 500 == 0:
                            data_set.output_predict(predicted_depths, images,
                                                    os.path.join(PREDICT_DIR, "iter_%05d_%05d" % (epoch, i)))
                            data_set.output_predict(test_predicted_depths, images_test,
                                                    os.path.join(PREDICT_DIR, "iter_%05d_%05d_test" % (epoch, i)))
                            self.save_model(self.sess, index)

                        index += 1

                coord.request_stop()
                coord.join(threads)
                writer.flush()
                writer.close()

    def save_model(self, sess, counter):
        self.saver.save(sess, os.path.join(CHECKPOINT_DIR, 'model'),
                        global_step=counter)  # because if there is no folder specified, it is used only as a prefix. Only in folder/prefix combination it puts each run into separate folder
