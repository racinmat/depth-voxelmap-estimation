# encoding: utf-8

from datetime import datetime
from tensorflow.python.platform import gfile
import numpy as np
import tensorflow as tf
from dataset import DataSet
from dataset import output_predict
import model
import train_operation as op
import os
import time

current_time = time.strftime("%Y-%m-%d--%H-%M-%S", time.gmtime())

MAX_STEPS = 10000000
LOG_DEVICE_PLACEMENT = False
BATCH_SIZE = 8
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
COARSE_DIR = "coarse"
PREDICT_DIR = os.path.join('predict', current_time)
CHECKPOINT_DIR = os.path.join('checkpoint', current_time)  # Directory name to save the checkpoints
LOGS_DIR = 'logs'
GPU_IDX = [0]


def train():
    # GPU settings
    config = tf.ConfigProto(log_device_placement=LOG_DEVICE_PLACEMENT)
    config.gpu_options.allow_growth = False
    config.gpu_options.allocator_type = 'BFC'
    devices_environ_var = 'CUDA_VISIBLE_DEVICES'
    if devices_environ_var in os.environ:
        available_devices = os.environ[devices_environ_var].split(',')
        if len(available_devices):
            if isinstance(GPU_IDX, list):
                os.environ[devices_environ_var] = ', '.join([available_devices[gpu] for gpu in GPU_IDX])
            else:
                gpu = GPU_IDX
                os.environ[devices_environ_var] = available_devices[gpu]

    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)
        dataset = DataSet(BATCH_SIZE)
        images, depths, invalid_depths = dataset.csv_inputs(TRAIN_FILE)
        test_images, test_depths, test_invalid_depths = dataset.csv_inputs(TEST_FILE)
        keep_conv = tf.placeholder(tf.float32)
        keep_hidden = tf.placeholder(tf.float32)
        logits = model.inference(images, keep_conv, keep_hidden)
        loss = model.loss(logits, depths, invalid_depths)
        train_op = op.train(loss, global_step, BATCH_SIZE)
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()  # saver must be initialized after network is set up

        # Session
        with tf.Session(config=config) as sess:
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
            for step in range(MAX_STEPS):
                index = 0
                for i in range(iterations):
                    _, loss_value, logits_val, images_val, summary_str = sess.run(
                        [train_op, loss, logits, images, summary],
                        feed_dict={keep_conv: 0.8, keep_hidden: 0.5})
                    if index % 10 == 0:
                        # test_loss_value, test_logits_val, test_images_val = sess.run([loss, test_logits, test_images],
                        #                                                        feed_dict={keep_conv: 0.8,
                        #                                                                   keep_hidden: 0.5})
                        # test_writer.add_summary(summary, index)
                        print("%s: %d[epoch]: %d[iteration]: train loss %f" % (datetime.now(), step, index, loss_value))
                        # print("%s: %d[epoch]: %d[iteration]: test loss %f" % (datetime.now(), step, index, test_loss_value))
                        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                    if index % 500 == 0:
                        output_predict(logits_val, images_val, os.path.join(PREDICT_DIR, "iter_%05d_%05d" % (step, i)))
                        # output_predict(test_logits_val, test_images_val, "data/test_predict_%05d_%05d" % (step, i))
                        save_model(saver, sess, step * iterations + i)

                    writer.add_summary(summary_str, sess)
                    index += 1

            coord.request_stop()
            coord.join(threads)


def save_model(saver, sess, counter):
    saver.save(sess, CHECKPOINT_DIR, global_step=counter)


def main(argv=None):
    if not gfile.Exists("./train"):
        gfile.MakeDirs("./train")
    if not gfile.Exists("./test"):
        gfile.MakeDirs("./test")
    if not gfile.Exists(PREDICT_DIR):
        gfile.MakeDirs(PREDICT_DIR)
    if not gfile.Exists(CHECKPOINT_DIR):
        gfile.MakeDirs(CHECKPOINT_DIR)
    if not gfile.Exists(LOGS_DIR):
        gfile.MakeDirs(LOGS_DIR)
    train()


if __name__ == '__main__':
    tf.app.run()
