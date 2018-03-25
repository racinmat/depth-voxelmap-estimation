import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import Network
import dataset
from dataset import DataSet


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
            output_image(image, image_name)

    for name, multi_images in multi_images_dict.items():
        for i, _ in enumerate(multi_images):
            multi_image = multi_images[i]
            for j in range(dataset.DEPTH_DIM):
                # ra_depth = multi_image[:, :, j] * 255.0
                depth_discr_name = name % (output_dir, i, j)
                output_image(multi_image, depth_discr_name, 'L')

                image_name = name % (output_dir, i)
                output_image(multi_images[i], image_name)


if __name__ == '__main__':
    with tf.Graph().as_default():
        with tf.Session() as sess:
            network = Network.Network()
            data_set, loss, estimated_depths, _, estimated_depths_images = network.prepare()

            filename = 'ml-datasets/2018-03-07--15-18-12--849.jpg'
            depth_filename = 'ml-datasets/2018-03-07--15-18-12--849.png'

            dataset.IS_GTA_DATA = True
            image = DataSet.filename_to_input_image(filename)
            depth = DataSet.filename_to_target_image(depth_filename)
            depth_discretized = dataset.DataSet.discretize_depth(depth)

            batch_size = 1
            # generate batch
            images, depths, depths_discretized = tf.train.batch(
                [image, depth, depth_discretized],
                batch_size=batch_size,
                num_threads=4,
                capacity=40)

            depth_reconstructed, weights, mask, mask_multiplied, mask_multiplied_sum = Network.Network.bins_to_depth(
                depths_discretized)

            print('weights: ', weights)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            images_val, depths_val, depths_discretized_val, depth_reconstructed_val, mask_val, mask_multiplied_val, mask_multiplied_sum_val = sess.run(
                [images, depths, depths_discretized, depth_reconstructed, mask, mask_multiplied, mask_multiplied_sum])
            estimated_depths_val, estimated_depths_images_val = sess.run([estimated_depths, estimated_depths_images])

            coord.request_stop()
            coord.join(threads)

            output_predictions({
                '%s/%03d_input.png': images_val,
                '%s/%03d_output.png': depths_val,
                '%s/%03d_reconstructed.png': depth_reconstructed_val,
            }, {
                '%s/%03d_input_%03d_discr.png': depths_discretized_val,
                '%s/%03d_output_%03d_discr.png': estimated_depths_val,
            }, 'predict-test')

            # depth_reconstructed_val = depth_reconstructed_val[:, :, :, 0]

            # layer = 2
            # f, axarr = plt.subplots(2, 3)
            # axarr[0, 0].set_title('masks_val')
            # axarr[0, 0].imshow(mask_val[0, :, :, layer])
            # axarr[0, 1].set_title('mask_multiplied_val')
            # axarr[0, 1].imshow(mask_multiplied_val[0, :, :, layer])
            # axarr[1, 0].set_title('depths_val')
            # axarr[1, 0].imshow(depths_val[0, :, :, 0])
            # axarr[1, 1].set_title('depths_discretized_val')
            # axarr[1, 1].imshow(depths_discretized_val[0, :, :, layer])
            # axarr[0, 2].set_title('mask_multiplied_sum_val')
            # axarr[0, 2].imshow(mask_multiplied_sum_val[0, :, :])
            # axarr[1, 2].set_title('depth_reconstructed_val')
            # axarr[1, 2].imshow(depth_reconstructed_val[0, :, :])
            # plt.show()
