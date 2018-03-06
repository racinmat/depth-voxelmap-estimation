import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from PIL import Image

IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320
TARGET_HEIGHT = 120
TARGET_WIDTH = 160
DEPTH_DIM = 200

MIN_DEQUE_EXAMPLES = 500 # should be relatively big compared to dataset, see https://stackoverflow.com/questions/43028683/whats-going-on-in-tf-train-shuffle-batch-and-tf-train-batch


class DataSet:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def csv_inputs(self, csv_file_path):
        filename_queue = tf.train.string_input_producer([csv_file_path], shuffle=True)
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
        depth = tf.div(depth, [255.0])
        # depth = tf.cast(depth, tf.int64)
        # resize
        image = tf.image.resize_images(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        depth = tf.image.resize_images(depth, (TARGET_HEIGHT, TARGET_WIDTH))
        # depth_bins = tf.image.resize_images(depth, (TARGET_HEIGHT, TARGET_WIDTH, DEPTH_DIM))

        invalid_depth = tf.sign(depth)
        # generate batch
        images, depths, invalid_depths = tf.train.shuffle_batch(
            [image, depth, invalid_depth],
            batch_size=self.batch_size,
            num_threads=4,
            capacity=MIN_DEQUE_EXAMPLES + 5 * self.batch_size,
            min_after_dequeue=MIN_DEQUE_EXAMPLES)
        return images, depths, invalid_depths

    def discretize_depth(self, depth):
        d_min = tf.reduce_min(depth)
        d_max = tf.reduce_max(depth)
        q = (tf.log(d_max) - tf.log(d_min)) / (DEPTH_DIM - 1)
        bin_idx = tf.round((tf.log(depth) - tf.log(d_min)) / q)
        mask = tf.ones((TARGET_HEIGHT, IMAGE_WIDTH, DEPTH_DIM))
        indices = tf.constant(np.array(range(DEPTH_DIM)))
        mask = tf.round(mask * tf.log(d_min) + tf.exp(q * indices)) # values corresponding to this bin, for comparison
        depth_discretized = tf.cast(tf.equal(mask, depth), tf.int8)
        return depth_discretized


def output_predict(depths, images, output_dir):
    print("output predict into %s" % output_dir)
    if not gfile.Exists(output_dir):
        gfile.MakeDirs(output_dir)
    for i, (image, depth) in enumerate(zip(images, depths)):
        pilimg = Image.fromarray(np.uint8(image))
        image_name = "%s/%05d_org.png" % (output_dir, i)
        pilimg.save(image_name)
        depth = depth.transpose(2, 0, 1)
        if np.max(depth) != 0:
            ra_depth = (depth / np.max(depth)) * 255.0
        else:
            ra_depth = depth * 255.0
        depth_pil = Image.fromarray(np.uint8(ra_depth[0]), mode="L")
        depth_name = "%s/%05d.png" % (output_dir, i)
        depth_pil.save(depth_name)
