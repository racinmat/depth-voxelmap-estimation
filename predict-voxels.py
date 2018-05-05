import tensorflow as tf
import numpy as np
from PIL import Image
import dataset
import losses
import metrics_np
from prettytable import PrettyTable
import os
import Network
from evaluation import load_model_with_structure
from evaluation_voxels import predict_voxels_to_pointcloud, evaluate_model, grid_voxelmap_to_pointcloud
from gta_math import grid_to_ndc_pcl_linear_view, ndc_to_view
from visualization import save_pointcloud_csv


def predict_voxels_to_pointcloud_without_gt(batch_rgb, batch_depths, model_names):
    for i in range(Network.BATCH_SIZE):
        im = Image.fromarray(batch_rgb[i, :, :, :].astype(np.uint8))
        im.save("evaluate/orig-rgb-{}.png".format(i))

    for model_name in model_names:
        pred_voxels, _ = evaluate_model(model_name, batch_rgb, batch_depths)

        # saving images
        for i in range(Network.BATCH_SIZE):
            pred_voxelmap = pred_voxels[i, :, :, :]
            np.save("evaluate/pred-voxelmap-{}-{}.npy".format(i, model_name), pred_voxelmap)
            pcl = grid_voxelmap_to_pointcloud(losses.is_obstacle(pred_voxelmap))
            save_pointcloud_csv(pcl.T[:, 0:3], "evaluate/pred-voxelmap-{}-{}.csv".format(i, model_name))


def main():
    model_names = [
        '2018-05-04--22-57-49',
        '2018-05-04--23-03-46',
    ]

    # images = np.array([
    #     ['ml-datasets-voxel/2018-03-07--17-52-29--004.jpg', 'ml-datasets-voxel/2018-03-07--17-52-29--004.npy'],
    #     ['ml-datasets-voxel/2018-03-07--16-40-51--211.jpg', 'ml-datasets-voxel/2018-03-07--16-40-51--211.npy'],
    #     ['ml-datasets-voxel/2018-03-07--15-44-35--835.jpg', 'ml-datasets-voxel/2018-03-07--15-44-35--835.npy'],
    #     ['ml-datasets-voxel/2018-03-07--15-22-14--222.jpg', 'ml-datasets-voxel/2018-03-07--15-22-14--222.npy'],
    # ])
    # these are from testing set
    images = np.array([
        ['ml-datasets-voxel/2018-03-07--16-40-42--901.jpg', 'ml-datasets-voxel/2018-03-07--16-40-42--901.npy'],
        ['ml-datasets-voxel/2018-03-07--17-41-16--827.jpg', 'ml-datasets-voxel/2018-03-07--17-41-16--827.npy'],
        ['ml-datasets-voxel/2018-03-07--16-12-57--023.jpg', 'ml-datasets-voxel/2018-03-07--16-12-57--023.npy'],
        ['ml-datasets-voxel/2018-03-07--15-44-56--353.jpg', 'ml-datasets-voxel/2018-03-07--15-44-56--353.npy'],
    ])

    Network.BATCH_SIZE = len(images)
    ds = dataset.DataSet(len(images))
    filename_list = tf.data.Dataset.from_tensor_slices((images[:, 0], images[:, 1]))
    images, depths, _ = ds.filenames_to_batch_voxel(filename_list)
    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    with tf.Session(config=config) as sess:
        batch_rgb, batch_depths = sess.run(
            [images, depths])
    print('evaluation dataset loaded')

    predict_voxels_to_pointcloud(batch_rgb, batch_depths, model_names)


if __name__ == '__main__':
    main()
