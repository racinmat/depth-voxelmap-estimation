import csv

import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
from PIL import Image

IMAGE_HEIGHT = 240
IMAGE_WIDTH = 320
TARGET_HEIGHT = 120
TARGET_WIDTH = 160

# DEPTH_DIM = 200
DEPTH_DIM = 100
# DEPTH_DIM = 10

D_MIN = 0.5
D_MAX = 50
Q = (np.log(D_MAX) - np.log(D_MIN)) / (DEPTH_DIM - 1)

MIN_DEQUE_EXAMPLES = 500  # should be relatively big compared to dataset, see https://stackoverflow.com/questions/43028683/whats-going-on-in-tf-train-shuffle-batch-and-tf-train-batch
IS_GTA_DATA = True
THRESHOLD = 1000
MAXIMUM = np.iinfo(np.uint16).max


class DataSet:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    @staticmethod
    def load_params(train_file_path):
        filenames = np.recfromcsv(train_file_path, delimiter=',', dtype=None)
        depths = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, len(filenames)))
        for i, (rgb_name, depth_name) in enumerate(filenames):
            img = Image.open(depth_name)
            img.load()
            img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.ANTIALIAS)
            data = np.asarray(img, dtype="int32")
            depths[:, :, i] = data

    @staticmethod
    def get_dataset_size(filename):
        with open(filename, newline='') as csv_file:
            file_object = csv.reader(csv_file)
            row_count = sum(1 for row in file_object)
        # print("dataset size is: "+str(row_count))
        # print("dataset file name is: "+str(filename))
        return row_count

    @staticmethod
    def filename_to_input_image(filename):
        jpg = tf.read_file(filename)
        image = tf.image.decode_jpeg(jpg, channels=3)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize_images(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        return image

    @staticmethod
    def filename_to_target_image(filename):
        depth_png = tf.read_file(filename)
        depth = tf.image.decode_png(depth_png, channels=1, dtype=tf.uint16)
        depth = tf.cast(depth, tf.float32)
        if IS_GTA_DATA:
            depth = DataSet.depth_from_integer_range(depth)
        depth = tf.image.resize_images(depth, (TARGET_HEIGHT, TARGET_WIDTH))
        return depth

    @staticmethod
    def filename_to_target_voxelmap(filename):
        voxelmap_bin = tf.read_file(filename)
        voxelmap = tf.decode_raw(voxelmap_bin, out_type=tf.uint16)
        voxelmap = tf.cast(voxelmap, tf.float32)
        return voxelmap

    def filenames_to_batch(self, filename, depth_filename, dataset_size=np.inf):
        # input
        image = self.filename_to_input_image(filename)
        # target
        depth = self.filename_to_target_image(depth_filename)
        depth_bins = self.discretize_depth(depth)
        depth_reconstructed = self.tf_bins_to_depth(depth_bins)

        # size = min(MIN_DEQUE_EXAMPLES, TOTAL_DATASET_SIZE)
        # capacity cannot be higher than dataset size, because then it throws exceptions
        min_deque_size = min(MIN_DEQUE_EXAMPLES + 5 * self.batch_size, dataset_size)
        min_after_deque = min(MIN_DEQUE_EXAMPLES, dataset_size - 1)

        # generate batch
        images, depths, depths_bins, depths_reconstructed = tf.train.shuffle_batch(
            [image, depth, depth_bins, depth_reconstructed],
            batch_size=self.batch_size,
            num_threads=4,
            capacity=min_deque_size,
            min_after_dequeue=min_after_deque)
        return images, depths, depths_bins, depths_reconstructed

    def filenames_to_batch_voxel(self, filename, voxelmap_filename, dataset_size=np.inf):
        # input
        image = self.filename_to_input_image(filename)
        # target
        voxelmap = self.filename_to_target_voxelmap(voxelmap_filename)
        depth_reconstructed = self.tf_voxelmap_to_depth(voxelmap)

        # size = min(MIN_DEQUE_EXAMPLES, TOTAL_DATASET_SIZE)
        # capacity cannot be higher than dataset size, because then it throws exceptions
        min_deque_size = min(MIN_DEQUE_EXAMPLES + 5 * self.batch_size, dataset_size)
        min_after_deque = min(MIN_DEQUE_EXAMPLES, dataset_size - 1)

        # generate batch
        images, voxelmaps, depths_reconstructed = tf.train.shuffle_batch(
            [image, voxelmap, depth_reconstructed],
            batch_size=self.batch_size,
            num_threads=4,
            capacity=min_deque_size,
            min_after_dequeue=min_after_deque)
        return images, voxelmaps, depths_reconstructed

    def csv_inputs(self, csv_file_path):
        filename_queue = tf.train.string_input_producer([csv_file_path], shuffle=True)
        reader = tf.TextLineReader()
        _, serialized_example = reader.read(filename_queue)
        filename, depth_filename = tf.decode_csv(serialized_example, [["path"], ["annotation"]])
        return self.filenames_to_batch(filename, depth_filename, self.get_dataset_size(csv_file_path))

    def csv_inputs_voxels(self, csv_file_path):
        filename_queue = tf.train.string_input_producer([csv_file_path], shuffle=True)
        reader = tf.TextLineReader()
        _, serialized_example = reader.read(filename_queue)
        filename, depth_filename = tf.decode_csv(serialized_example, [["path"], ["annotation"]])
        return self.filenames_to_batch_voxel(filename, depth_filename, self.get_dataset_size(csv_file_path))

    @staticmethod
    def discretize_depth(depth):
        d_min = tf.constant(D_MIN, dtype=tf.float32)
        q = tf.constant(Q, dtype=tf.float32)
        ones_vec = tf.ones((TARGET_HEIGHT, TARGET_WIDTH, DEPTH_DIM + 1))
        sth = tf.expand_dims(tf.constant(np.append(np.array(range(DEPTH_DIM)), np.inf)), 0)
        sth = tf.expand_dims(sth, 0)
        indices_vec = tf.tile(sth, [TARGET_HEIGHT, TARGET_WIDTH, 1])
        indices_vec_lower = indices_vec - 1
        # indices = ones_vec * indices_vec
        # indices = ones_vec * indices_vec
        # bin value = bin_idx * q + log(d_min)
        d_min_tensor = ones_vec * tf.log(d_min)
        bin_value = q * tf.cast(indices_vec, tf.float32)
        bin_value_lower = q * tf.cast(indices_vec_lower, tf.float32)
        logged = d_min_tensor + bin_value
        logged_lower = d_min_tensor + bin_value_lower
        mask = tf.exp(logged)  # values corresponding to this bin, for comparison
        mask_lower = tf.exp(logged_lower)  # values corresponding to this bin, for comparison
        depth_discretized = tf.cast(tf.less_equal(depth, mask), tf.int8) * tf.cast(tf.greater(depth, mask_lower),
                                                                                   tf.int8)
        return depth_discretized

    @staticmethod
    def np_bins_to_depth(depth_bins):
        # same as Network.bins_to_depth, but only for one image
        weights = np.array(range(DEPTH_DIM)) * Q + np.log(D_MIN)
        mask = np.tile(weights, (TARGET_HEIGHT, TARGET_WIDTH, 1))
        depth = np.exp(np.sum(np.multiply(mask, depth_bins), axis=2))
        return depth

    @staticmethod
    def tf_bins_to_depth(depth_bins):
        # same as Network.bins_to_depth, but only for one image
        weights = np.array(range(DEPTH_DIM)) * Q + np.log(D_MIN)
        print('weight shape', weights.shape)
        sth = tf.expand_dims(tf.constant(weights, dtype=tf.float32), 0)
        print('sth shape', sth.shape)
        sth = tf.expand_dims(sth, 0)
        print('sth shape', sth.shape)
        mask = tf.tile(sth, [TARGET_HEIGHT, TARGET_WIDTH, 1])
        print('mask shape', mask.shape)
        # depth = np.exp(np.sum(np.multiply(mask, depth_bins), axis=2))
        mask_multiplied = tf.multiply(mask, tf.cast(depth_bins[:, :, 0:DEPTH_DIM], dtype=tf.float32))
        print('mask_multiplied shape', mask_multiplied.shape)
        mask_multiplied_sum = tf.reduce_sum(mask_multiplied, axis=2)
        print('mask_multiplied_sum shape', mask_multiplied_sum.shape)
        depth = tf.exp(mask_multiplied_sum)
        print('depth shape', depth.shape)

        return depth

    @staticmethod
    def tf_voxelmap_to_depth(voxels):
        # this visualizes voxelmap as depth image
        voxels = tf.reverse(voxels, axis=2)
        depth_size = voxels.shape[2]
        depth = np.argmax(voxels, axis=2)

        depth *= int(255 / depth_size)  # normalizing to use all of classing png values

        return depth

    @staticmethod
    def output_predict(depths, images, gt_depths, output_dir):
        print("output predict into %s" % output_dir)
        if not gfile.Exists(output_dir):
            gfile.MakeDirs(output_dir)

        assert len(depths) == len(images) and len(depths) == len(gt_depths)
        for i in range(len(images)):
            image = images[i]
            depth = depths[i]
            gt_depth = gt_depths[i]

            # print('depth shape:', depth.shape)
            if len(depth.shape) == 3 and depth.shape[2] > 1:
                raise Exception('oh, boi, shape is going wild', depth.shape)
            if len(depth.shape) == 3 and depth.shape[2] > 1:
                raise Exception('oh, boi, shape is going wild', depth.shape)
            depth = depth[:, :, 0]
            gt_depth = gt_depth[:, :, 0]

            # input image
            pilimg = Image.fromarray(np.uint8(image))
            image_name = "%s/%05d_org.png" % (output_dir, i)
            pilimg.save(image_name)

            # estimated depths
            ra_depth = (depth / np.max(depth)) * 255.0
            depth_pil = Image.fromarray(np.uint8(ra_depth), mode="L")
            depth_name = "%s/%05d.png" % (output_dir, i)
            depth_pil.save(depth_name)

            # ground truth depth
            ra_depth = (gt_depth / np.max(gt_depth)) * 255.0
            gt_depth_pil = Image.fromarray(np.uint8(ra_depth), mode="L")
            gt_depth_name = "%s/%05d_ground_truth.png" % (output_dir, i)
            gt_depth_pil.save(gt_depth_name)

    @staticmethod
    def depth_from_integer_range(depth):
        tf.cast(depth, dtype=tf.float32)
        # then we rescale to integer32
        ratio = THRESHOLD / MAXIMUM
        return depth * tf.constant(ratio)
