import tensorflow as tf
import numpy as np
import metrics_np
from prettytable import PrettyTable
import os
import Network


def load_model_with_structure(model_name, graph, sess):
    import re
    tf.logging.info(" [*] Loading last checkpoint")

    checkpoint_dir = os.path.join('checkpoint', model_name)
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    if not checkpoint or not checkpoint.model_checkpoint_path:
        tf.logging.info(" [*] Failed to find a checkpoint")
        return False, 0
    checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
    data_file = os.path.join(checkpoint_dir, checkpoint_name)
    meta_file = data_file + '.meta'
    saver = tf.train.import_meta_graph(meta_file)
    saver.restore(sess, data_file)
    counter = int(next(re.finditer("(\d+)(?!.*\d)", checkpoint_name)).group(0))
    self.sampler = graph.get_tensor_by_name(sampler_name)
    tf.logging.info(" [*] Success to read {}".format(checkpoint_name))
    return True, counter


def inference(model, rgb_image, graph, sess):
    image_val = sess.run(model, feed_dict={
        x: rgb_image
    })
    return image_val


def evaluate_model(model_name, rgb_img, truth_img):
    # not running on any GPU, using only CPU
    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    with tf.Graph().as_default() as graph:
        with tf.Session(config=config) as sess:
            model = load_model_with_structure(model_name, graph, sess)
            pred_img = inference(model, rgb_img, graph, sess)

    return {
        'treshold_1.25': metrics_np.accuracy_under_treshold(truth_img, pred_img, 1.25),
        'mean_rel_err': metrics_np.mean_relative_error(truth_img, pred_img),
        'rms': metrics_np.root_mean_squared_error(truth_img, pred_img),
        'rms_log': metrics_np.root_mean_squared_log_error(truth_img, pred_img),
        'log10_err': metrics_np.log10_error(truth_img, pred_img),
    }


def get_evaluation_names():
    return {
        'treshold_1.25',
        'mean_rel_err',
        'rms',
        'rms_log',
        'log10_err',
    }


if __name__ == '__main__':
    model_names = [
        '2018-03-11--23-23-32',
        '2018-03-11--15-30-10',
        '2018-03-11--14-40-26',
    ]
    #
    # images = [
    #     '',
    #     '',
    # ]

    image_rgb = ''
    image_depth = ''
    x = PrettyTable(get_evaluation_names())
    for model_name in model_names:
        accuracies = evaluate_model(model_name, image_rgb, image_depth)
        x.add_row(accuracies)

    print(x)
