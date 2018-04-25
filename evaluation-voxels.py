import tensorflow as tf
import numpy as np
from PIL import Image
import dataset
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
        print(" [*] Failed to find a checkpoint")
        return False, 0, None
    checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
    data_file = os.path.join(checkpoint_dir, checkpoint_name)
    meta_file = data_file + '.meta'
    saver = tf.train.import_meta_graph(meta_file)
    saver.restore(sess, data_file)
    counter = int(next(re.finditer("(\d+)(?!.*\d)", checkpoint_name)).group(0))
    last_layer = graph.get_tensor_by_name('network/softmaxFinal/Reshape_1:0')
    input = graph.get_tensor_by_name('network/x:0')
    print(" [*] Success to read {} in iteration {}".format(checkpoint_name, counter))
    return True, input, last_layer


def inference(model, input, rgb_image, graph, sess):
    image_val = sess.run(model, feed_dict={
        input: rgb_image
    })
    return image_val


def evaluate_model(model_name, needs_conversion, rgb_img, truth_img):
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
    #     'treshold_1.25': metrics_np.accuracy_under_treshold(truth_img, pred_img, 1.25),
    #     'mean_rel_err': metrics_np.mean_relative_error(truth_img, pred_img),
    #     'rms': metrics_np.root_mean_squared_error(truth_img, pred_img),
    #     'rms_log': metrics_np.root_mean_squared_log_error(truth_img, pred_img),
    #     'log10_err': metrics_np.log10_error(truth_img, pred_img),
    # }
    return pred_img, [
        metrics_np.accuracy_under_treshold(truth_img, pred_img, 1.25),
        metrics_np.mean_relative_error(truth_img, pred_img),
        metrics_np.root_mean_squared_error(truth_img, pred_img),
        metrics_np.root_mean_squared_log_error(truth_img, pred_img),
        metrics_np.log10_error(truth_img, pred_img),
    ]


def get_evaluation_names():
    return [
        'treshold_1.25',
        'mean_rel_err',
        'rms',
        'rms_log',
        'log10_err',
    ]


def grid_voxelmap_to_pointcloud():
    proj_matrix = np.array([[1.21006660e+00, 0.00000000e+00, 0.00000000e+00,
                             0.00000000e+00],
                            [0.00000000e+00, 2.14450692e+00, 0.00000000e+00,
                             0.00000000e+00],
                            [0.00000000e+00, 0.00000000e+00, 1.49965283e-04,
                             1.50022495e+00],
                            [0.00000000e+00, 0.00000000e+00, -1.00000000e+00,
                             0.00000000e+00]])


if __name__ == '__main__':
    model_names = [
        # format is name, needs conversion from bins
        '2018-04-23--08-15-23',
    ]

    images = np.array([
        ['ml-datasets-voxel/2018-03-07--17-52-29--004.jpg', 'ml-datasets-voxel/2018-03-07--17-52-29--004.jpg'],
        ['ml-datasets-voxel/2018-03-07--16-40-51--211.jpg', 'ml-datasets-voxel/2018-03-07--16-40-51--211.jpg'],
        ['ml-datasets-voxel/2018-03-07--15-44-35--835.jpg', 'ml-datasets-voxel/2018-03-07--15-44-35--835.jpg'],
        ['ml-datasets-voxel/2018-03-07--15-22-14--222.jpg', 'ml-datasets-voxel/2018-03-07--15-22-14--222.jpg'],
    ])

    Network.BATCH_SIZE = len(images)
    ds = dataset.DataSet(len(images))
    records = tf.train.input_producer(images)
    res = records.dequeue()
    images, depths, _, _ = ds.filenames_to_batch_voxel(res)
    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    with tf.Session(config=config) as sess:
        batch_rgb, batch_voxels = sess.run(
            [images, depths])
    print('evaluation dataset loaded')

    for i in range(Network.BATCH_SIZE):
        im = Image.fromarray(batch_rgb[i, :, :, :].astype(np.uint8))
        im.save("evaluate/orig-rgb-{}.png".format(i))

        voxels = batch_voxels[i, :, :, :]

        voxels.save()

    column_names = get_evaluation_names()
    column_names.append('name')
    x = PrettyTable(column_names)

    for model_name, needs_conv in model_names:
        pred_img, accuracies = evaluate_model(model_name, needs_conv, batch_rgb, batch_voxels)
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
            im.save("evaluate/predicted-{}-{}.png".format(i, model_name))

    print(x)
