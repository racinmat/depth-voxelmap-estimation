import tensorflow as tf


def accuracy_under_treshold(truth_img, predicted_img, treshold=1.25):
    n_pixels = tf.reduce_prod(tf.shape(truth_img))
    delta = tf.maximum(predicted_img / truth_img, truth_img / predicted_img)
    lower = delta < treshold
    ratio = tf.reduce_sum(tf.cast(lower, tf.float32)) / n_pixels
    return ratio


def mean_relative_error(truth_img, predicted_img):
    n_pixels = tf.reduce_prod(tf.shape(truth_img))
    rel = tf.abs(predicted_img - truth_img) / truth_img
    rel = tf.reduce_sum(rel) / n_pixels
    return rel


def root_mean_squared_log_error(truth_img, predicted_img):
    n_pixels = tf.reduce_prod(tf.shape(truth_img))
    rms = (tf.log(predicted_img, 10) - log(truth_img, 10)) ** 2
    rms = tf.sqrt(tf.reduce_sum(rms) / n_pixels)
    return rms


def root_mean_squared_error(truth_img, predicted_img):
    n_pixels = tf.reduce_prod(tf.shape(truth_img))
    rms = (predicted_img - truth_img) ** 2
    rms = tf.sqrt(tf.reduce_sum(rms) / n_pixels)
    return rms


def log10_error(truth_img, predicted_img):
    n_pixels = tf.reduce_prod(tf.shape(truth_img))
    lg10 = tf.abs(tf.log(predicted_img, 10) - log(truth_img, 10))
    lg10 = tf.reduce_sum(lg10) / n_pixels
    return lg10


def log(x, base):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(base, dtype=numerator.dtype))
    return numerator / denominator

