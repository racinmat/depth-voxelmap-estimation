import tensorflow as tf
import numpy as np
from PIL import Image
import dataset
import losses
import metrics_np
from prettytable import PrettyTable
import os
import Network
import metrics_tf


def load_model_with_structure(model_name, graph, sess):
    import re
    tf.logging.info(" [*] Loading last checkpoint")

    checkpoint_dir = os.path.join('checkpoint', model_name)
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    if not checkpoint or not checkpoint.model_checkpoint_path:
        print(" [*] Failed to find a checkpoint")
        return False, 0, None
    checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
    data_file = os.path.join(checkpoint_dir, checkpoint_name)
    meta_file = data_file + '.meta'
    saver = tf.train.import_meta_graph(meta_file)
    saver.restore(sess, data_file)
    counter = int(next(re.finditer("(\d+)(?!.*\d)", checkpoint_name)).group(0))
    last_layer = graph.get_tensor_by_name('network/inference:0')
    input = graph.get_tensor_by_name('network/x:0')
    print(" [*] Success to read {} in iteration {}".format(checkpoint_name, counter))
    return True, input, last_layer


def inference(model, input, rgb_image, graph, sess):
    image_val = sess.run(model, feed_dict={
        input: rgb_image
    })
    return image_val


def evaluate_model(model_name, needs_conversion, rgb_img):
    # not running on any GPU, using only CPU
    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    with tf.Graph().as_default() as graph:
        with tf.Session(config=config) as sess:
            _, input, model = load_model_with_structure(model_name, graph, sess)
            if needs_conversion:
                model = Network.Network.bins_to_depth(model)
            pred_img = inference(model, input, rgb_img, graph, sess)

    # return pred_img, {
    #     'treshold_1.25': metrics_np.depth_accuracy_under_treshold(truth_img, pred_img, 1.25),
    #     'mean_rel_err': metrics_np.depth_mean_relative_error(truth_img, pred_img),
    #     'rms': metrics_np.depth_root_mean_squared_error(truth_img, pred_img),
    #     'rms_log': metrics_np.depth_root_mean_squared_log_error(truth_img, pred_img),
    #     'log10_err': metrics_np.depth_log10_error(truth_img, pred_img),
    # }
    return pred_img


def get_accuracies(truth_img, pred_img):
    return [
        metrics_np.accuracy_under_treshold(truth_img, pred_img, 1.25),
        metrics_np.mean_relative_error(truth_img, pred_img),
        metrics_np.root_mean_squared_error(truth_img, pred_img),
        metrics_np.root_mean_squared_log_error(truth_img, pred_img),
        metrics_np.log10_error(truth_img, pred_img),
    ]


def get_accuracies_voxel(truth_voxel, pred_voxel):
    return [
        metrics_tf.voxel_false_positive_error(truth_voxel, pred_voxel),
        metrics_tf.voxel_true_positive_error(truth_voxel, pred_voxel),
        metrics_tf.voxel_iou_error(truth_voxel, pred_voxel),
        losses.softmax_voxelwise_loss_with_undefined(truth_voxel, pred_voxel),
        metrics_tf.voxel_l1_dist_with_unknown(truth_voxel, pred_voxel),
    ]


def get_evaluation_names():
    return [
        'treshold_1.25',
        'mean_rel_err',
        'rms',
        'rms_log',
        'log10_err',
    ]


if __name__ == '__main__':
    model_names = [
        # format is name, needs conversion from bins
        ['2018-03-11--02-04-09', True],
        # ['2018-03-11--15-30-10', True],
        # ['2018-03-11--14-40-26', True],
        # ['2018-03-04--23-16-47', False],
    ]

    # images = np.array([
    #     ['data/nyu_datasets/00836.jpg', 'data/nyu_datasets/00836.png'],
    #     ['data/nyu_datasets/00952.jpg', 'data/nyu_datasets/00952.png'],
    #     ['data/nyu_datasets/00953.jpg', 'data/nyu_datasets/00953.png'],
    # ])
    images = np.array([
        ['ml-datasets/2018-03-07--18-24-37--499.jpg', 'ml-datasets/2018-03-07--18-24-37--499.png'],
        ['ml-datasets/2018-03-07--17-25-35--384.jpg', 'ml-datasets/2018-03-07--17-25-35--384.png'],
        ['ml-datasets/2018-03-07--16-58-59--208.jpg', 'ml-datasets/2018-03-07--16-58-59--208.png'],
        ['ml-datasets/2018-03-07--16-22-31--875.jpg', 'ml-datasets/2018-03-07--16-22-31--875.png'],
        ['ml-datasets/2018-03-07--17-31-39--573.jpg', 'ml-datasets/2018-03-07--17-31-39--573.png'],
        ['ml-datasets/2018-03-07--16-08-50--454.jpg', 'ml-datasets/2018-03-07--16-08-50--454.png'],
    ])
    Network.BATCH_SIZE = len(images)
    ds = dataset.DataSet(len(images))
    filename_list = tf.data.Dataset.from_tensor_slices((images[:, 0], images[:, 1]))
    images, depths, _, _ = ds.filenames_to_batch(filename_list)

    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    with tf.Session(config=config) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        batch_rgb, batch_depth = sess.run(
            [images, depths])
        coord.request_stop()
        coord.join(threads)
    print('evaluation dataset loaded')

    for i in range(Network.BATCH_SIZE):
        im = Image.fromarray(batch_rgb[i, :, :, :].astype(np.uint8))
        im.save("evaluate-depths/orig-rgb-{}.png".format(i))

        depth = batch_depth[i, :, :, :]
        if len(depth.shape) == 3 and depth.shape[2] > 1:
            raise Exception('oh, boi, shape is going wild', depth.shape)
        depth = depth[:, :, 0]

        if np.max(depth) != 0:
            depth = (depth / np.max(depth)) * 255.0
        else:
            depth = depth * 255.0
        im = Image.fromarray(depth.astype(np.uint8), mode="L")
        im.save("evaluate-depths/orig-depth-{}.png".format(i))

    column_names = get_evaluation_names()
    column_names.append('name')
    x = PrettyTable(column_names)

    for model_name, needs_conv in model_names:
        pred_img = evaluate_model(model_name, needs_conv, batch_rgb)
        accuracies = get_accuracies(batch_rgb, batch_depth)
        # accuracies['name'] = model_name
        # x.add_row(accuracies.values())
        accuracies.append(model_name)
        x.add_row(accuracies)

        # saving images
        for i in range(Network.BATCH_SIZE):
            depth = pred_img[i, :, :, :]
            if len(depth.shape) == 3 and depth.shape[2] > 1:
                raise Exception('oh, boi, shape is going wild', depth.shape)
            depth = depth[:, :, 0]

            if np.max(depth) != 0:
                depth = (depth / np.max(depth)) * 255.0
            else:
                depth = depth * 255.0
            im = Image.fromarray(depth.astype(np.uint8), mode="L")
            im.save("evaluate-depths/predicted-{}-{}.png".format(i, model_name))

    print(x)
# for checking validity of gzip: nohup $(gzip -t open_virtualscapes.tar.gz && echo ok || echo bad) &> is_open_ok.txt &
# for gzip in nohup: nohup gzip open_virtualscapes.tar &