import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import dataset
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope

# def playground_loss_function(labels, logits):
#     # in rank 2, [elements, classes]
#
#     # tf.nn.weighted_cross_entropy_with_logits(labels, logits, weights)
#     losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
#     return losses
#
#
# def prob_to_logit(probs):
#     return np.log(probs / (1 - probs))
#
#
# def softmax(x):
#     """Same behaviour as tf.nn.softmax in tensorflow"""
#     e_x = np.exp(x)
#     sum_per_row = np.tile(e_x.sum(axis=1), (x.shape[1], 1)).T
#     print('e_x', '\n', e_x)
#     print('sum_per_row', '\n', sum_per_row)
#     return e_x / sum_per_row
#
#
# def softmax_cross_entropy_loss(labels, logits):
#     """Same behaviour as tf.nn.softmax_cross_entropy_with_logits in tensorflow"""
#     loss_per_row = - np.sum(labels * np.log(softmax(logits)), axis=1)
#     return loss_per_row


def labels_to_info_gain(labels, logits, alpha=0.2):
    last_axis = len(logits.shape) - 1
    label_idx = np.tile(np.argmax(labels, axis=last_axis), (labels.shape[last_axis], 1)).T
    prob_bin_idx = np.tile(range(logits.shape[last_axis]), (labels.shape[0], 1))
    # print('label_idx', '\n', label_idx)
    # print('probs_idx', '\n', prob_bin_idx)
    info_gain = np.exp(-alpha * (label_idx - prob_bin_idx)**2)
    print('info gain', '\n', info_gain)
    return info_gain


def tf_labels_to_info_gain(labels, logits, alpha=0.2):
    # int 16 stačí, protože je to index binu pro hloubku
    last_axis = len(logits.shape) - 1
    label_idx = tf.expand_dims(tf.argmax(labels, axis=last_axis), 0)
    label_idx = tf.cast(label_idx, dtype=tf.int32)
    label_idx = tf.tile(label_idx, [labels.shape[last_axis], 1])
    label_idx = tf.transpose(label_idx)
    prob_bin_idx = tf.expand_dims(tf.range(logits.shape[last_axis], dtype=tf.int32), last_axis)
    prob_bin_idx = tf.transpose(prob_bin_idx)
    prob_bin_idx = tf.tile(prob_bin_idx, [labels.shape[0], 1])
    difference = (label_idx - prob_bin_idx)**2
    difference = tf.cast(difference, dtype=tf.float32)
    info_gain = tf.exp(-alpha * difference)
    return info_gain


def softmax_loss(labels, logits):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))


def information_gain_loss(labels, logits):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_labels_to_info_gain(labels=labels, logits=logits), logits=logits))


def inference():
    with arg_scope([layers.fully_connected],
                   weights_initializer=layers.xavier_initializer(uniform=True),
                   biases_initializer=tf.constant_initializer(0.1),
                   ):
        x = tf.placeholder(tf.float32, shape=[None, 5], name='x')
        l1 = slim.fully_connected(x, num_outputs=10, scope='fc1', activation_fn=tf.nn.relu)
        l2 = slim.fully_connected(l1, num_outputs=5, scope='fc2', activation_fn=None)
        return x, l2


if __name__ == '__main__':
    with tf.Graph().as_default():
        with tf.Session() as sess:
            x, logits = inference()
            probs = slim.softmax(logits)
            labels = tf.placeholder(tf.float32, shape=[None, 5], name='labels')
            loss = softmax_loss(labels=labels, logits=logits)
            optimizer = tf.train.AdamOptimizer()
            train_op = optimizer.minimize(loss)
            tf.summary.scalar("loss", loss)
            tf.summary.scalar("prob0", probs[0, 0])
            tf.summary.scalar("prob1", probs[0, 1])
            tf.summary.scalar("prob2", probs[0, 2])
            tf.summary.scalar("prob3", probs[0, 3])
            tf.summary.scalar("prob4", probs[0, 4])
            summary = tf.summary.merge_all()  # merge all summaries to dump them for tensorboard
            writer = tf.summary.FileWriter('playground/simple', sess.graph)

            sess.run(tf.global_variables_initializer())
            for i in range(1000):
                _, cost, predicted, summary_str = sess.run([train_op, loss, probs, summary], feed_dict={
                    x: np.array([
                        [1, 1, 1, 1, 1],
                        [0, 0, 0, 1, 1],
                    ]),
                    labels: np.array([
                        [0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                    ]),
                })
                writer.add_summary(summary_str, i)
                print('iteration i:', i)
