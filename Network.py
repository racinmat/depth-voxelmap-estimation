# encoding: utf-8

from datetime import datetime
import numpy as np
import tensorflow as tf
from dataset import DataSet, output_predict
import model
import train_operation as op
import os
import time

current_time = time.strftime("%Y-%m-%d--%H-%M-%S", time.gmtime())

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
        # GPU settings
        self.saver = None
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


    def train(self):
        dataset = DataSet(BATCH_SIZE)
        with tf.Graph().as_default():
            global_step = tf.Variable(0, trainable=False)
            images, depths, invalid_depths = dataset.csv_inputs(TRAIN_FILE)
            keep_conv = tf.placeholder(tf.float32, name='keep_conv')
            keep_hidden = tf.placeholder(tf.float32, name='keep_hidden')
            logits = model.inference(images, keep_conv, keep_hidden)
            loss = model.loss(logits, depths, invalid_depths)
            train_op = op.train(loss, global_step, BATCH_SIZE)
            init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver()  # saver must be initialized after network is set up

            # Session
            with tf.Session(config=self.config) as sess:
                sess.run(init_op)
                # parameters
                summary = tf.summary.merge_all()  # merge all summaries to dump them for tensorboard
                writer = tf.summary.FileWriter(LOGS_DIR, sess.graph)
                tf.global_variables_initializer().run()

                for variable in tf.trainable_variables():
                    variable_name = variable.name
                    print("parameter: %s" % variable_name)
                    if variable_name.find("/") < 0 or variable_name.count("/") != 1:
                        continue

                # train
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                # test_logits_val = None
                # test_images_val = None

                iterations = 1000
                index = 0
                for epoch in range(MAX_EPOCHS):
                    for i in range(iterations):
                        _, loss_value, logits_val, images_val, summary_str = sess.run(
                            [train_op, loss, logits, images, summary],
                            feed_dict={keep_conv: 0.8, keep_hidden: 0.5})
                        writer.add_summary(summary_str, index)
                        if i % 10 == 0:
                            print("%s: %d[epoch]: %d[iteration]: train loss %f" % (datetime.now(), epoch, i, loss_value))
                            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                        if i % 500 == 0:
                            output_predict(logits_val, images_val, os.path.join(PREDICT_DIR, "iter_%05d_%05d" % (epoch, i)))
                            self.save_model(saver, sess, index)

                        index += 1

                coord.request_stop()
                coord.join(threads)
                writer.flush()
                writer.close()


    def save_model(self, sess, counter):
        self.saver.save(sess, CHECKPOINT_DIR, global_step=counter)