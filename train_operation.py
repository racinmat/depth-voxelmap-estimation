# encoding: utf-8

import tensorflow as tf

# NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1500 # should be size of dataset so epoch name is correct
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 40 # should be size of dataset so epoch name is correct
NUM_EPOCHS_PER_DECAY = 30
ITERATIONS_PER_DECAY = 20000
INITIAL_LEARNING_RATE = 1e-4
LEARNING_RATE_DECAY_FACTOR = 0.1  # dividing by 10 every decay


def train(total_loss, global_step, batch_size):
    # num_batches_per_epoch = float(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN) / batch_size
    # decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    decay_steps = ITERATIONS_PER_DECAY
    lr = tf.train.exponential_decay(
        INITIAL_LEARNING_RATE,
        global_step,
        decay_steps,
        LEARNING_RATE_DECAY_FACTOR,
        staircase=True)
    tf.summary.scalar('learning_rate', lr)
    opt = tf.train.AdamOptimizer(lr)

    return opt.minimize(total_loss, global_step=global_step)
