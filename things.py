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

        for j in range(DEPTH_DIM):
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


if __name__ == '__main__':
    ds = DataSet(8)
    ds.load_params('train.csv')

    d = list(range(1, 100))
    d_min = np.min(d)
    d_max = 20
    num_bins = 10
    q_calc = (np.log(np.max(d)) - np.log(d_min)) / (num_bins - 1)
    # q = 0.5  # width of quantization bin
    l = np.round((np.log(d) - np.log(d_min)) / q_calc)

    print(d)
    print(l)

    print('q_calc', q_calc)
    #
    # f, axarr = plt.subplots(2, 2)
    # axarr[0, 0].plot(d)
    # axarr[0, 1].plot(np.log(d))
    # axarr[1, 0].plot(np.log(d) - np.log(d_min))
    # axarr[1, 1].plot((np.log(d) - np.log(d_min)) / q_calc)
    # plt.show()

    with tf.Graph().as_default():
        with tf.Session() as sess:
            x = tf.constant(d)

            # for i in range(500):
            #     if i % 500 == 0:
            #         print('hi', i)

            IMAGE_HEIGHT = 240
            IMAGE_WIDTH = 320
            TARGET_HEIGHT = 120
            TARGET_WIDTH = 160
            DEPTH_DIM = 10

            filename_queue = tf.train.string_input_producer(['train.csv'], shuffle=True)
            reader = tf.TextLineReader()
            _, serialized_example = reader.read(filename_queue)
            filename, depth_filename = tf.decode_csv(serialized_example, [["path"], ["annotation"]])
            # input
            jpg = tf.read_file(filename)
            image = tf.image.decode_jpeg(jpg, channels=3)
            image = tf.cast(image, tf.float32)
            # target
            depth_png = tf.read_file(depth_filename)
            depth = tf.image.decode_png(depth_png, channels=1)
            depth = tf.cast(depth, tf.float32)
            depth = depth / 255.0
            # depth = tf.cast(depth, tf.int64)
            # resize
            image = tf.image.resize_images(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
            depth = tf.image.resize_images(depth, (TARGET_HEIGHT, TARGET_WIDTH))

            depth_discretized = dataset.DataSet.discretize_depth(depth)

            invalid_depth = tf.sign(depth)

            batch_size = 8
            # generate batch
            images, depths, depths_discretized, invalid_depths = tf.train.batch(
                [image, depth, depth_discretized, invalid_depth],
                batch_size=batch_size,
                num_threads=4,
                capacity=40)

            depth_reconstructed, weights, mask, mask_multiplied, mask_multiplied_sum = Network.Network.bins_to_depth(depths_discretized)

            print('weights: ', weights)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            images_val, depths_val, depths_discretized_val, invalid_depths_val, depth_reconstructed_val, mask_val, mask_multiplied_val, mask_multiplied_sum_val = sess.run(
                [images, depths, depths_discretized, invalid_depths, depth_reconstructed, mask, mask_multiplied, mask_multiplied_sum])
            sess.run(images)

            output_predict(depths_val, images_val, depths_discretized_val,
                           depth_reconstructed_val, 'kunda')

            depth_reconstructed_val = depth_reconstructed_val[:, :, :, 0]
            coord.request_stop()
            coord.join(threads)

            layer = 2
            f, axarr = plt.subplots(2, 3)
            axarr[0, 0].set_title('masks_val')
            axarr[0, 0].imshow(mask_val[0, :, :, layer])
            axarr[0, 1].set_title('mask_multiplied_val')
            axarr[0, 1].imshow(mask_multiplied_val[0, :, :, layer])
            axarr[1, 0].set_title('depths_val')
            axarr[1, 0].imshow(depths_val[0, :, :, 0])
            axarr[1, 1].set_title('depths_discretized_val')
            axarr[1, 1].imshow(depths_discretized_val[0, :, :, layer])
            axarr[0, 2].set_title('mask_multiplied_sum_val')
            axarr[0, 2].imshow(mask_multiplied_sum_val[0, :, :])
            axarr[1, 2].set_title('depth_reconstructed_val')
            axarr[1, 2].imshow(depth_reconstructed_val[0, :, :])
            plt.show()
