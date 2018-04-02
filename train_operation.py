# encoding: utf-8

import tensorflow as tf

ITERATIONS_PER_DECAY = 30000
INITIAL_LEARNING_RATE = 1e-4
LEARNING_RATE_DECAY_FACTOR = 0.1  # dividing by 10 every decay


def train(total_loss, global_step, batch_size):
    decay_steps = ITERATIONS_PER_DECAY
    # lr = tf.train.exponential_decay(
    #     INITIAL_LEARNING_RATE,
    #     global_step,
    #     decay_steps,
    #     LEARNING_RATE_DECAY_FACTOR,
    #     staircase=True)
    lr = tf.constant(INITIAL_LEARNING_RATE)
    tf.summary.scalar('learning_rate', lr)
    # opt = tf.train.AdamOptimizer(lr, epsilon=1e-5)
    opt = tf.contrib.opt.NadamOptimizer(lr, epsilon=1e-8)

    return opt.minimize(total_loss, global_step=global_step)
