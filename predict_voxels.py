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
from evaluation_voxels import evaluate_model, grid_voxelmap_to_pointcloud
from gta_math import grid_to_ndc_pcl_linear_view, ndc_to_view
from visualization import save_pointcloud_csv


def predict_voxels_to_pointcloud_without_gt(batch_rgb, model_names):
    for i in range(Network.BATCH_SIZE):
        im = Image.fromarray(batch_rgb[i, :, :, :].astype(np.uint8))
        im.save("evaluate-test/orig-rgb-{}.png".format(i))

    for model_name in model_names:
        pred_voxels = evaluate_model(model_name, batch_rgb)

        # saving images
        for i in range(Network.BATCH_SIZE):
            pred_voxelmap = pred_voxels[i, :, :, :]
            np.save("evaluate-test/pred-voxelmap-{}-{}.npy".format(i, model_name), pred_voxelmap)
            pcl = grid_voxelmap_to_pointcloud(losses.is_obstacle(pred_voxelmap))
            save_pointcloud_csv(pcl.T[:, 0:3], "evaluate-test/pred-voxelmap-{}-{}.csv".format(i, model_name))


def main():
    model_names = [
        # '2018-05-04--22-57-49',
        # '2018-05-04--23-03-46',
        '2018-05-07--17-22-10',
    ]

    images = np.array([
        'evaluate-test/2018-03-07--16-32-39--933.jpg',
        'evaluate-test/2018-03-30--05-27-14--498.jpg',
        'evaluate-test/2018-03-30--05-27-23--208.jpg',
        'evaluate-test/2018-03-30--06-58-56--715.jpg',
        'evaluate-test/2018-03-30--08-22-05--068.jpg',
        'evaluate-test/2018-03-30--09-14-13--507.jpg',
        'evaluate-test/31934476_10215502017054352_182588598875324416_n.jpg',
        'evaluate-test/31946042_10215502017774370_705717582623145984_n.jpg',
        'evaluate-test/2018-03-30--09-19-17--598.jpg',
        'evaluate-test/1367521646_6ebde5f80f_z.jpg',
        'evaluate-test/2272564735_63f59857b7_z.jpg',
        'evaluate-test/4682987533_530cff53e1_z.jpg',
        'evaluate-test/72228787_79fe46ba25_z.jpg',
        'evaluate-test/3859520659_e117d00fdc_z.jpg',
        'evaluate-test/8240796375_54235112e5_z.jpg',
    ])

    Network.BATCH_SIZE = len(images)
    ds = dataset.DataSet(len(images))
    filename_list = tf.data.Dataset.from_tensor_slices(images)
    images = ds.filenames_to_batch_voxel_rgb_only(filename_list)
    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    with tf.Session(config=config) as sess:
        batch_rgb = sess.run(images)
    print('evaluation dataset loaded')

    predict_voxels_to_pointcloud_without_gt(batch_rgb, model_names)


if __name__ == '__main__':
    main()
