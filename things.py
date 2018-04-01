import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import Network
import dataset
from Network import BATCH_SIZE
from dataset import DataSet


def output_predict(depths, images, depths_discretized, depths_reconstructed, output_dir):
    print("output predict into %s" % output_dir)
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)
    for i, _ in enumerate(images):
        image, depth, depth_discretized, depth_reconstructed = images[i], depths[i], depths_discretized[i], \
                                                               depths_reconstructed[i]

        pilimg = Image.fromarray(np.uint8(image))
        image_name = "%s/%03d_org.png" % (output_dir, i)
        pilimg.save(image_name)
        depth = depth.transpose(2, 0, 1)
        if np.max(depth) != 0:
            ra_depth = (depth / np.max(depth)) * 255.0
        else:
            ra_depth = depth * 255.0
        depth_pil = Image.fromarray(np.uint8(ra_depth[0]), mode="L")
        depth_name = "%s/%03d.png" % (output_dir, i)
        depth_pil.save(depth_name)

        for j in range(dataset.DEPTH_DIM):
            ra_depth = depth_discretized[:, :, j] * 255.0
            depth_discr_pil = Image.fromarray(np.uint8(ra_depth), mode="L")
            depth_discr_name = "%s/%03d_%03d_discr.png" % (output_dir, i, j)
            depth_discr_pil.save(depth_discr_name)

        # for j in range(DEPTH_DIM):
        #     ra_depth = mask[:, :, j]
        #     depth_discr_pil = Image.fromarray(np.uint8(ra_depth), mode="L")
        #     depth_discr_name = "%s/%03d_%03d_discr_m.png" % (output_dir, i, j)
        #     depth_discr_pil.save(depth_discr_name)
        #
        # for j in range(DEPTH_DIM):
        #     ra_depth = mask_lower[:, :, j]
        #     depth_discr_pil = Image.fromarray(np.uint8(ra_depth), mode="L")
        #     depth_discr_name = "%s/%03d_%03d_discr_ml.png" % (output_dir, i, j)
        #     depth_discr_pil.save(depth_discr_name)

        depth = depth_reconstructed[:, :, 0]
        if np.max(depth) != 0:
            ra_depth = (depth / np.max(depth)) * 255.0
        else:
            ra_depth = depth * 255.0
        depth_pil = Image.fromarray(np.uint8(ra_depth), mode="L")
        depth_name = "%s/%03d_reconstructed.png" % (output_dir, i)
        depth_pil.save(depth_name)


def playground_loss_function(labels, logits):
    # in rank 2, [elements, classes]

    # tf.nn.weighted_cross_entropy_with_logits(labels, logits, weights)
    losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return losses


def prob_to_logit(probs):
    return np.log(probs / (1 - probs))


def softmax(x):
    """Same behaviour as tf.nn.softmax in tensorflow"""
    e_x = np.exp(x)
    sum_per_row = np.tile(e_x.sum(axis=1), (x.shape[1], 1)).T
    print('e_x', '\n', e_x)
    print('sum_per_row', '\n', sum_per_row)
    return e_x / sum_per_row


def softmax_cross_entropy_loss(labels, logits):
    """Same behaviour as tf.nn.softmax_cross_entropy_with_logits in tensorflow"""
    loss_per_row = - np.sum(labels * np.log(softmax(logits)), axis=1)
    return loss_per_row


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


def informed_cross_entropy_loss(labels, logits):
    """Same behaviour as tf.nn.weighted_cross_entropy_with_logits in tensorflow"""
    probs = softmax(logits)
    print('probs', '\n', probs)
    logged_probs = np.log(probs)
    print('logged probs', '\n', logged_probs)
    loss_per_row = - np.sum(labels_to_info_gain(labels, logits) * logged_probs, axis=1)
    return loss_per_row


def playing_with_losses():
    labels = np.array([
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        # [0, 1, 0, 0, 0],
        # [0, 0, 1, 0, 0],
        # [0, 0, 0, 1, 0],
        # [0, 0, 0, 0, 1],
        # [1, 0, 0, 0, 0],
    ])
    logits = np.array([
        [0, 20, 0, 0, 0],
        [0, 10, 0, 0, 0],
        [0, 2, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 0, 1],
        [0, 1, 0, 0, 0],
        # [3, 1, 1, 1, 1],
        # [0, 10, 0, 0, 0],
        # [1, 5, 1, 1, 1],
        # [0, 0, 1, 0, 0],
        # [1, 1, 4, 1, 1],
        # [1, 1, 1, 4, 1],
        # [1, 1, 1, 1, 4],
        # [4, 1, 1, 1, 1],
    ])
    probs = softmax(logits)
    loss = softmax_cross_entropy_loss(labels=labels, logits=logits)
    new_loss = informed_cross_entropy_loss(labels=labels, logits=logits)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            logits_tf = tf.constant(logits, dtype=tf.float32)
            labels_tf = tf.constant(labels, dtype=tf.float32)
            probs_tf = sess.run(tf.nn.softmax(logits_tf))
            loss_tf = sess.run(tf.nn.softmax_cross_entropy_with_logits(labels=labels_tf, logits=logits_tf))
            new_loss_tf = sess.run(tf.nn.softmax_cross_entropy_with_logits(labels=tf_labels_to_info_gain(labels, logits_tf), logits=logits_tf))

    # print('labels', '\n', labels)
    # print('logits', '\n', logits)
    # print('probs', '\n', probs)
    # print('probs diff', '\n', probs - probs_tf)
    print('loss', '\n', loss)
    print('loss_tf', '\n', loss_tf)
    print('loss diff', '\n', loss - loss_tf)

    print('new_loss', '\n', new_loss)
    print('new_loss_tf', '\n', new_loss_tf)
    print('new loss diff', '\n', new_loss - new_loss_tf)

    # f, axarr = plt.subplots(2, 3)
    # axarr[0, 0].set_title('sample 1')
    # axarr[0, 0].plot(probs[0, :])
    # axarr[0, 1].set_title('sample 2')
    # axarr[0, 1].plot(probs[1, :])
    # axarr[1, 0].set_title('sample 3')
    # axarr[1, 0].plot(probs[2, :])
    # axarr[1, 1].set_title('sample 4')
    # axarr[1, 1].plot(probs[3, :])
    plt.plot(probs[0, :], color='r')
    plt.plot(probs[1, :], color='g')
    plt.plot(probs[2, :], color='b')
    plt.plot(probs[3, :], color='y')

    plt.show()


def tf_dataset_experiments():
    filename_queue = tf.train.string_input_producer(['train-gta.csv'], shuffle=True)
    reader = tf.TextLineReader()
    _, serialized_example = reader.read(filename_queue)
    filename, depth_filename = tf.decode_csv(serialized_example, [["path"], ["annotation"]])
    num_records = reader.num_records_produced()
    print(reader.num_records_produced())
    with tf.Graph().as_default():
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            res = sess.run(num_records)
            print(res)


if __name__ == '__main__':
    # playing_with_losses()
    tf_dataset_experiments()


    # arr = np.array([
    #     [1, 1, 1, 2],
    #     [2, 2, 2, 4],
    #     [4, 4, 4, 8],
    # ])
    # with tf.Graph().as_default():
    #     with tf.Session() as sess:
    #         logits_tf = tf.constant(arr, dtype=tf.float32)
    #         tf_mean = sess.run(tf.reduce_mean(logits_tf))
    #         print('tf_mean\n', tf_mean)
    #
    # print('mean\n', np.mean(arr))
    # print('sum_per_row\n', np.sum(arr, axis=1))
    # print('mean_of_sum\n', np.mean(np.sum(arr, axis=1), axis=0))

    # ds = DataSet(8)
    # ds.load_params('train.csv')
    #
    # d = list(range(1, 100))
    # d_min = np.min(d)
    # d_max = 20
    # num_bins = 10
    # q_calc = (np.log(np.max(d)) - np.log(d_min)) / (num_bins - 1)
    # # q = 0.5  # width of quantization bin
    # l = np.round((np.log(d) - np.log(d_min)) / q_calc)
    #
    # print(d)
    # print(l)
    #
    # print('q_calc', q_calc)
    #
    # f, axarr = plt.subplots(2, 2)
    # axarr[0, 0].plot(d)
    # axarr[0, 1].plot(np.log(d))
    # axarr[1, 0].plot(np.log(d) - np.log(d_min))
    # axarr[1, 1].plot((np.log(d) - np.log(d_min)) / q_calc)
    # plt.show()

    # with tf.Graph().as_default():
    #     with tf.Session() as sess:
    #         x = tf.constant(d)
    #
    #         # for i in range(500):
    #         #     if i % 500 == 0:
    #         #         print('hi', i)
    #
    #         IMAGE_HEIGHT = 240
    #         IMAGE_WIDTH = 320
    #         TARGET_HEIGHT = 120
    #         TARGET_WIDTH = 160
    #         DEPTH_DIM = 10
    #
    #         filename_queue = tf.train.string_input_producer(['train.csv'], shuffle=True)
    #         reader = tf.TextLineReader()
    #         _, serialized_example = reader.read(filename_queue)
    #         filename, depth_filename = tf.decode_csv(serialized_example, [["path"], ["annotation"]])
    #         # input
    #         jpg = tf.read_file(filename)
    #         image = tf.image.decode_jpeg(jpg, channels=3)
    #         image = tf.cast(image, tf.float32)
    #         # target
    #         depth_png = tf.read_file(depth_filename)
    #         depth = tf.image.decode_png(depth_png, channels=1)
    #         depth = tf.cast(depth, tf.float32)
    #         depth = depth / 255.0
    #         # depth = tf.cast(depth, tf.int64)
    #         # resize
    #         image = tf.image.resize_images(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
    #         depth = tf.image.resize_images(depth, (TARGET_HEIGHT, TARGET_WIDTH))
    #
    #         depth_discretized = dataset.DataSet.discretize_depth(depth)
    #
    #         invalid_depth = tf.sign(depth)
    #
    #         batch_size = 8
    #         # generate batch
    #         images, depths, depths_discretized, invalid_depths = tf.train.batch(
    #             [image, depth, depth_discretized, invalid_depth],
    #             batch_size=batch_size,
    #             num_threads=4,
    #             capacity=40)
    #
    #         depth_reconstructed, weights, mask, mask_multiplied, mask_multiplied_sum = Network.Network.bins_to_depth(depths_discretized)
    #
    #         print('weights: ', weights)
    #
    #         coord = tf.train.Coordinator()
    #         threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #
    #         images_val, depths_val, depths_discretized_val, invalid_depths_val, depth_reconstructed_val, mask_val, mask_multiplied_val, mask_multiplied_sum_val = sess.run(
    #             [images, depths, depths_discretized, invalid_depths, depth_reconstructed, mask, mask_multiplied, mask_multiplied_sum])
    #         sess.run(images)
    #
    #         output_predict(depths_val, images_val, depths_discretized_val,
    #                        depth_reconstructed_val, 'kunda')
    #
    #         depth_reconstructed_val = depth_reconstructed_val[:, :, :, 0]
    #         coord.request_stop()
    #         coord.join(threads)
    #
    #         layer = 2
    #         f, axarr = plt.subplots(2, 3)
    #         axarr[0, 0].set_title('masks_val')
    #         axarr[0, 0].imshow(mask_val[0, :, :, layer])
    #         axarr[0, 1].set_title('mask_multiplied_val')
    #         axarr[0, 1].imshow(mask_multiplied_val[0, :, :, layer])
    #         axarr[1, 0].set_title('depths_val')
    #         axarr[1, 0].imshow(depths_val[0, :, :, 0])
    #         axarr[1, 1].set_title('depths_discretized_val')
    #         axarr[1, 1].imshow(depths_discretized_val[0, :, :, layer])
    #         axarr[0, 2].set_title('mask_multiplied_sum_val')
    #         axarr[0, 2].imshow(mask_multiplied_sum_val[0, :, :])
    #         axarr[1, 2].set_title('depth_reconstructed_val')
    #         axarr[1, 2].imshow(depth_reconstructed_val[0, :, :])
    #         plt.show()

    # network = Network.Network()
    # network.prepare()
    # total_vars = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    # print('trainable vars: ', total_vars)
# for output bins = 200: 73 696 786
# for output bins = 100: 65 312 586