import tensorflow as tf

import losses


def depth_accuracy_under_treshold(truth_img, predicted_img, treshold=1.25):
    n_pixels = tf.cast(tf.reduce_prod(tf.shape(truth_img)), dtype=tf.float32)
    delta = tf.maximum(predicted_img / truth_img, truth_img / predicted_img)
    lower = delta < treshold
    ratio = tf.reduce_sum(tf.cast(lower, dtype=tf.float32)) / n_pixels
    return ratio


def depth_mean_relative_error(truth_img, predicted_img):
    n_pixels = tf.cast(tf.reduce_prod(tf.shape(truth_img)), dtype=tf.float32)
    rel = tf.abs(predicted_img - truth_img) / truth_img
    rel = tf.reduce_sum(rel) / n_pixels
    return rel


def depth_root_mean_squared_log_error(truth_img, predicted_img):
    n_pixels = tf.cast(tf.reduce_prod(tf.shape(truth_img)), dtype=tf.float32)
    rms = (log(predicted_img, 10) - log(truth_img, 10)) ** 2
    rms = tf.sqrt(tf.reduce_sum(rms) / n_pixels)
    return rms


def depth_root_mean_squared_error(truth_img, predicted_img):
    n_pixels = tf.cast(tf.reduce_prod(tf.shape(truth_img)), dtype=tf.float32)
    rms = (predicted_img - truth_img) ** 2
    rms = tf.sqrt(tf.reduce_sum(rms) / n_pixels)
    return rms


def depth_log10_error(truth_img, predicted_img):
    n_pixels = tf.cast(tf.reduce_prod(tf.shape(truth_img)), dtype=tf.float32)
    lg10 = tf.abs(log(predicted_img, 10) - log(truth_img, 10))
    lg10 = tf.reduce_sum(lg10) / n_pixels
    return lg10


def log(x, base):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(base, dtype=numerator.dtype))
    return numerator / denominator


# def voxel_l1_error(voxel_gt, voxel_pred):
#     pass
#

def voxel_false_positive_error(voxel_gt, voxel_pred):
    # formula is FP / FP + TN
    # formula is false positive / (false positive + true negative)
    # TN = true negative means voxel_pred == 0 & voxel_gt == 0
    # FP = false positive means voxel_pred == 1 & voxel_gt == 0
    tn = tf.reduce_sum(tf.equal(voxel_gt, 0) & tf.equal(voxel_pred, 0))
    fp = tf.reduce_sum(tf.equal(voxel_gt, 1) & tf.equal(voxel_pred, 0))
    return fp / (fp + tn)


def voxel_true_positive_error(voxel_gt, voxel_pred):
    # formula is TP / FN + TP
    # formula is true positive / (false negative + true positive)
    # FN = false negative means voxel_pred == 0 & voxel_gt == 1
    # TP = true positive means voxel_pred == 1 & voxel_gt == 1
    obst_pred = tf.equal(voxel_pred, 1)
    obst_gt = tf.equal(voxel_gt, 1)
    free_gt = tf.equal(voxel_gt, 0)
    fn = tf.reduce_sum(free_gt & obst_pred)
    tp = tf.reduce_sum(obst_gt & obst_pred)
    return tp / (fn + tp)


def voxel_iou_error(voxel_gt, voxel_pred):
    # https://arxiv.org/pdf/1604.00449.pdf
    obst_pred = tf.equal(voxel_pred, 1)
    obst_gt = tf.equal(voxel_gt, 1)
    return tf.reduce_sum(obst_gt & obst_pred) / tf.reduce_sum(obst_gt | obst_pred)


def voxel_l1_dist_with_unknown(voxel_gt, voxel_pred):
    # https://arxiv.org/pdf/1612.00101.pdf simple l1 dist, but masked
    known_mask = losses.get_known_mask(voxel_gt)
    obst_pred = tf.equal(voxel_pred, 1)
    obst_gt = tf.equal(voxel_gt, 1)
    return tf.reduce_mean(tf.reduce_sum(known_mask * tf.abs(voxel_gt - voxel_pred), [1, 2, 3]))
