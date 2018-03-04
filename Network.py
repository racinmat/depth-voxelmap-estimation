# encoding: utf-8

from datetime import datetime
import numpy as np
import tensorflow as tf

import dataset
from dataset import DataSet, output_predict
import model
import train_operation as op
import os
import time
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope

current_time = time.strftime("%Y-%m-%d--%H-%M-%S", time.gmtime())

# these weights are from resnet: https://github.com/ry/tensorflow-resnet/blob/master/resnet.py
MOVING_AVERAGE_DECAY = 0.9997
BN_DECAY = MOVING_AVERAGE_DECAY
BN_EPSILON = 0.001
CONV_WEIGHT_DECAY = 0.00004
CONV_WEIGHT_STDDEV = 0.1

MAX_EPOCHS = 10000000
LOG_DEVICE_PLACEMENT = False
BATCH_SIZE = 8
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
COARSE_DIR = "coarse"
PREDICT_DIR = os.path.join('predict', current_time)
CHECKPOINT_DIR = os.path.join('checkpoint', current_time)  # Directory name to save the checkpoints
LOGS_DIR = 'logs'
GPU_IDX = [0]


class Network(object):

    def __init__(self):
        self.sess = None
        self.saver = None

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

    def inference(self, images):
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
                       weights_initializer=tf.truncated_normal_initializer(stddev=CONV_WEIGHT_STDDEV),
                       biases_initializer=tf.constant_initializer(0.1),
                       weights_regularizer=slim.l2_regularizer(CONV_WEIGHT_DECAY)
                       ):
            conv1 = slim.conv2d(images, num_outputs=64, scope='conv1', kernel_size=7, stride=2, normalizer_fn=None,
                                activation_fn=tf.nn.relu)

            max1 = slim.max_pool2d(conv1, kernel_size=3, stride=2, scope='maxpool1')

            conv1 = self.resize_layer("resize1", max1, small_size=64, big_size=256)
            print("conv1")
            print(conv1)

            for i in range(2):
                conv1 = self.non_resize_layer("resize2-" + str(i), conv1, small_size=64, big_size=256)

            conv1 = self.resize_layer("resize3", conv1, small_size=128, big_size=512, stride=2)

            l1concat = conv1
            print("l1concat")
            print(l1concat)

            for i in range(7):
                conv1 = self.non_resize_layer("resize4-" + str(i), conv1, small_size=128, big_size=512)

            l2concat = conv1
            print("l2concat")
            print(l2concat)

            conv1 = self.resize_layer("resize5", conv1, small_size=256, big_size=1024, rate=2)

            l3concat = conv1
            print("l3concat")
            print(l3concat)

            for i in range(35):
                conv1 = self.non_resize_layer("resize6-" + str(i), conv1, small_size=256, big_size=1024, rate=2)

            l4concat = conv1
            print("l4concat")
            print(l4concat)

            conv1 = self.resize_layer("resize7", conv1, small_size=512, big_size=2048, rate=4)

            l5concat = conv1
            print("l5concat")
            print(l5concat)

            for i in range(2):
                conv1 = self.non_resize_layer("resize8-" + str(i), conv1, small_size=512, big_size=2048, rate=4)

            l6concat = conv1
            print("l6concat")
            print(l6concat)

            conv1 = tf.concat([l1concat, l2concat, l3concat, l4concat, l5concat, l6concat], 3)

            conv1 = tf.layers.dropout(conv1, .5)

            conv1 = slim.conv2d(conv1, num_outputs=200, scope='convFinal', kernel_size=3, stride=1, normalizer_fn=None,
                                activation_fn=None)

            conv1 = tf.layers.conv2d_transpose(conv1, 1, 8, strides=(4, 4), padding='SAME')

            return conv1

    def loss(self, logits, depths, invalid_depths):
        H = dataset.TARGET_HEIGHT
        W = dataset.TARGET_WIDTH
        logits_flat = tf.reshape(logits, [-1, H * W])
        depths_flat = tf.reshape(depths, [-1, H * W])
        print("logits_flat")
        print(logits_flat)
        print("depths_flat")
        print(depths_flat)
        invalid_depths_flat = tf.reshape(invalid_depths, [-1, 120 * 160])

        predict = tf.multiply(logits_flat, invalid_depths_flat)
        target = tf.multiply(depths_flat, invalid_depths_flat)
        d = tf.subtract(predict, target)
        square_d = tf.square(d)
        sum_square_d = tf.reduce_sum(square_d, 1)
        sum_d = tf.reduce_sum(d, 1)
        sqare_sum_d = tf.square(sum_d)
        cost = tf.reduce_mean(sum_square_d / (H * W) - 0.5 * sqare_sum_d / np.math.pow(H * W, 2))
        return cost
        # tf.add_to_collection('losses', cost)
        # return tf.add_n(tf.get_collection('losses'), name='total_loss')

    def train(self):
        dataset = DataSet(BATCH_SIZE)
        with tf.Graph().as_default():
            global_step = tf.Variable(0, trainable=False)
            images, depths, invalid_depths = dataset.csv_inputs(TRAIN_FILE)
            logits = self.inference(images)
            loss = self.loss(logits, depths, invalid_depths)
            train_op = op.train(loss, global_step, BATCH_SIZE)
            init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver()  # saver must be initialized after network is set up

            # Session
            with tf.Session(config=self.config) as self.sess:
                self.sess.run(init_op)
                # parameters
                summary = tf.summary.merge_all()  # merge all summaries to dump them for tensorboard
                writer = tf.summary.FileWriter(os.path.join(LOGS_DIR, current_time), self.sess.graph)
                tf.global_variables_initializer().run()

                for variable in tf.trainable_variables():
                    variable_name = variable.name
                    print("parameter: %s" % variable_name)
                    if variable_name.find("/") < 0 or variable_name.count("/") != 1:
                        continue

                # train
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

                # test_logits_val = None
                # test_images_val = None

                iterations = 1000
                index = 0
                for epoch in range(MAX_EPOCHS):
                    for i in range(iterations):
                        _, loss_value, logits_val, images_val, summary_str = self.sess.run(
                            [train_op, loss, logits, images, summary])
                        writer.add_summary(summary_str, index)
                        if i % 10 == 0:
                            print(
                                "%s: %d[epoch]: %d[iteration]: train loss %f" % (datetime.now(), epoch, i, loss_value))
                            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                        if i % 500 == 0:
                            output_predict(logits_val, images_val,
                                           os.path.join(PREDICT_DIR, "iter_%05d_%05d" % (epoch, i)))
                            self.save_model(self.sess, index)

                        index += 1

                coord.request_stop()
                coord.join(threads)
                writer.flush()
                writer.close()

    def save_model(self, sess, counter):
        self.saver.save(sess, CHECKPOINT_DIR, global_step=counter)
