import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.python.saved_model import tag_constants
import os
import Network
import dataset
import metrics_tf
from dataset import DataSet
import pickle
from shutil import copyfile
from os.path import join


def output_image(image, name, mode='RGB'):
    pil_image = Image.fromarray(np.uint8(image), mode=mode)
    pil_image.save(name)


def output_predictions(images_dict, multi_images_dict, output_dir):
    print("output predict into %s" % output_dir)
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    for name, images in images_dict.items():
        for i, _ in enumerate(images):
            image = images[i]
            image_name = name % (output_dir, i)
            if image.shape[2] == 1:
                mode = 'L'
                image = image[:, :, 0]
            else:
                mode = 'RGB'
            output_image(image, image_name, mode)

    for name, multi_images in multi_images_dict.items():
        for i, _ in enumerate(multi_images):
            multi_image = multi_images[i]
            for j in range(dataset.DEPTH_DIM):
                # ra_depth = multi_image[:, :, j] * 255.0
                depth_discr_name = name % (output_dir, i, j)
                image = multi_image[:, :, j]
                output_image(image, depth_discr_name, 'L')


if __name__ == '__main__':
    # filename = 'ml-datasets/2018-03-07--15-18-12--849.jpg'
    # depth_filename = 'ml-datasets/2018-03-07--15-18-12--849.png'
    filename = 'ml-datasets/2018-03-07--15-27-34--483.jpg'
    depth_filename = 'ml-datasets/2018-03-07--15-27-34--483.png'
    dataset.IS_GTA_DATA = True
    out_name = 'output-gta'

    # filename = 'data/nyu_datasets/00016.jpg'
    # depth_filename = 'data/nyu_datasets/00016.png'
    # dataset.IS_GTA_DATA = False
    # out_name = 'output-nyu'

    # checkpoint_model = 'checkpoint/2018-03-19--04-14-04'
    checkpoint_model = 'checkpoint/2018-03-28--13-55-48'
    out_dir = 'inspections'

    with tf.Graph().as_default() as graph:
        with tf.Session() as sess:
            image = DataSet.filename_to_input_image(filename)
            depth = DataSet.filename_to_target_image(depth_filename)
            depth_discretized = DataSet.discretize_depth(depth)
            depths_discretized_ft = tf.cast(depth_discretized, dtype=tf.float32)

            depth_png = tf.read_file(depth_filename)
            depth_raw = tf.image.decode_png(depth_png, channels=1, dtype=tf.uint16)
            depth_raw = tf.cast(depth_raw, tf.float32)

            batch_size = 1
            # generate batch
            images, depths, depths_discretized = tf.train.batch(
                [image, depth, depth_discretized],
                batch_size=batch_size,
                num_threads=4,
                capacity=40)

            depth_reconstructed = Network.Network.bins_to_depth(depths_discretized)

            # loading saved network
            checkpoint = tf.train.get_checkpoint_state(checkpoint_model)
            if not checkpoint or not checkpoint.model_checkpoint_path:
                raise Exception('not any checkpoint found')
            saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
            tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta')
            saver.restore(sess, checkpoint.model_checkpoint_path)

            input = graph.get_tensor_by_name('network/x:0')
            estimated_depths = graph.get_tensor_by_name('network/inference:0')
            estimated_depths_images = Network.Network.bins_to_depth(estimated_depths)

            # metrics for discretization itself
            treshold_gt = metrics_tf.accuracy_under_treshold(depth, depth_reconstructed, 1.25)
            mre_gt = metrics_tf.mean_relative_error(depth, depth_reconstructed)
            rms_gt = metrics_tf.root_mean_squared_error(depth, depth_reconstructed)
            rmls_gt = metrics_tf.root_mean_squared_log_error(depth, depth_reconstructed)

            # metrics for training
            treshold = metrics_tf.accuracy_under_treshold(depth_reconstructed, estimated_depths_images, 1.25)
            mre = metrics_tf.mean_relative_error(depth_reconstructed, estimated_depths_images)
            rms = metrics_tf.root_mean_squared_error(depth_reconstructed, estimated_depths_images)
            rmls = metrics_tf.root_mean_squared_log_error(depth_reconstructed, estimated_depths_images)

            # metrics for training in wrong implementation to check the behaviour
            treshold_wrong = metrics_tf.accuracy_under_treshold(depths_discretized_ft, estimated_depths_images, 1.25)
            mre_wrong = metrics_tf.mean_relative_error(depths_discretized_ft, estimated_depths_images)
            rms_wrong = metrics_tf.root_mean_squared_error(depths_discretized_ft, estimated_depths_images)
            rmls_wrong = metrics_tf.root_mean_squared_log_error(depths_discretized_ft, estimated_depths_images)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # evaluating only inputs
            images_val, depths_val, depths_discretized_val, depth_reconstructed_val, depth_raw_val = sess.run(
                [images, depths, depths_discretized, depth_reconstructed, depth_raw])
            # evaluation input metrics
            treshold_gt_val, mre_gt_val, rms_gt_val, rmls_gt_val = sess.run(
                [treshold_gt, mre_gt, rms_gt, rmls_gt])
            # evaluating depth estimations
            estimated_depths_val, estimated_depths_images_val = sess.run([estimated_depths, estimated_depths_images],
                                                                         feed_dict={
                                                                             input: images_val,
                                                                         })
            # evaluation depth estimation metrics
            treshold_val, mre_val, rms_val, rmls_val = sess.run([treshold, mre, rms, rmls],
                                                                feed_dict={
                                                                    input: images_val,
                                                                })
            treshold_val, mre_val, rms_val, rmls_val = sess.run([treshold, mre, rms, rmls],
                                                                feed_dict={
                                                                    input: images_val,
                                                                })
            treshold_wrong_val, mre_wrong_val, rms_wrong_val, rmls_wrong_val = sess.run([treshold_wrong, mre_wrong, rms_wrong, rmls_wrong],
                                                                feed_dict={
                                                                    input: images_val,
                                                                })
            # output_predictions({
            #     '%s/%03d_input.png': images_val,
            #     '%s/%03d_output_gt.png': depths_val * 255.0,
            #     '%s/%03d_output.png': estimated_depths_images_val * 255.0,
            #     '%s/%03d_reconstructed.png': depth_reconstructed_val * 255.0,
            # }, {
            #     '%s/%03d_input_%03d_discr.png': depths_discretized_val,
            #     '%s/%03d_output_%03d_discr.png': estimated_depths_val,
            # }, 'predict-test-nyu')

            with open(join(out_dir, out_name + '.rick'), 'wb+') as f:
                pickle.dump([images_val, depths_val, estimated_depths_images_val, depth_reconstructed_val,
                             depths_discretized_val, estimated_depths_val, depth_raw_val,
                             treshold_gt_val, mre_gt_val, rms_gt_val, rmls_gt_val,
                             treshold_val, mre_val, rms_val, rmls_val,
                             treshold_wrong_val, mre_wrong_val, rms_wrong_val, rmls_wrong_val,
                             ], f)

            # copy original files to show any problems with loading and preprocessing images
            copyfile(filename, join(out_dir, out_name + '-in-image.jpg'))
            copyfile(depth_filename, join(out_dir, out_name + '-in-depth.png'))

            coord.request_stop()
            coord.join(threads)
