import tensorflow as tf
import numpy as np
from PIL import Image
import dataset
import losses
import metrics_np
from prettytable import PrettyTable
import os
import Network
from evaluation import load_model_with_structure, get_evaluation_names, get_accuracies, get_accuracies_voxel
from gta_math import grid_to_ndc_pcl_linear_view, ndc_to_view
from visualization import save_pointcloud_csv


def inference(model, input, rgb_image, sess):
    image_val = sess.run(model, feed_dict={
        input: rgb_image
    })
    return image_val


def calc_loss(input, rgb_image, gt_depth, graph, sess):
    y = graph.get_tensor_by_name('y:0')
    loss = graph.get_tensor_by_name('loss:0')
    image_val = sess.run(loss, feed_dict={
        input: rgb_image,
        y: gt_depth
    })
    return image_val


def evaluate_model(model_name, rgb_img):
    # not running on any GPU, using only CPU
    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    with tf.Graph().as_default() as graph:
        with tf.Session(config=config) as sess:
            _, input, model = load_model_with_structure(model_name, graph, sess)
            pred_voxels = inference(model, input, rgb_img, sess)
    return pred_voxels


def evaluate_model_with_loss(model_name, rgb_img, gt_voxel):
    # not running on any GPU, using only CPU
    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    with tf.Graph().as_default() as graph:
        with tf.Session(config=config) as sess:
            _, input, model = load_model_with_structure(model_name, graph, sess)
            pred_voxels = inference(model, input, rgb_img, sess)
    pred_loss = calc_loss(input, rgb_img, gt_voxel, graph, sess)
    return pred_voxels, pred_loss


def calculate_voxel_metrics(model_name, rgb_image, voxels_gt):
    # not running on any GPU, using only CPU
    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    with tf.Graph().as_default() as graph:
        with tf.Session(config=config) as sess:
            _, input, model = load_model_with_structure(model_name, graph, sess)
            voxel_metrics = get_accuracies_voxel(voxels_gt, model)
            pred_voxels, metrics = sess.run([model, voxel_metrics], feed_dict={
                input: rgb_image
    })
    return metrics, pred_voxels


def grid_voxelmap_to_pointcloud(ndc_grid):
    ndc_grid = np.transpose(ndc_grid, (1, 0, 2))    # because of form it is in the output of network
    z_meters_min = 1.5
    z_meters_max = 25
    proj_matrix = np.array([[1.21006660e+00, 0.00000000e+00, 0.00000000e+00,
                             0.00000000e+00],
                            [0.00000000e+00, 2.14450692e+00, 0.00000000e+00,
                             0.00000000e+00],
                            [0.00000000e+00, 0.00000000e+00, 1.49965283e-04,
                             1.50022495e+00],
                            [0.00000000e+00, 0.00000000e+00, -1.00000000e+00,
                             0.00000000e+00]])
    ndc_points_reconst = grid_to_ndc_pcl_linear_view(ndc_grid, proj_matrix, z_meters_min, z_meters_max)
    ndc_points_reconst = np.hstack((ndc_points_reconst, np.ones((ndc_points_reconst.shape[0], 1)))).T

    view_points_reconst = ndc_to_view(ndc_points_reconst, proj_matrix)
    return view_points_reconst


def grid_voxelmap_to_paraview_pointcloud(ndc_grid):
    ndc_grid = np.transpose(ndc_grid, (1, 0, 2))    # because of form it is in the output of network
    # underlying functinons expect true/false points and only reconstruct true points, so we must make grid full of trues to get it working
    positions_grid = np.ones_like(ndc_grid, dtype=bool)
    z_meters_min = 1.5
    z_meters_max = 25
    proj_matrix = np.array([[1.21006660e+00, 0.00000000e+00, 0.00000000e+00,
                             0.00000000e+00],
                            [0.00000000e+00, 2.14450692e+00, 0.00000000e+00,
                             0.00000000e+00],
                            [0.00000000e+00, 0.00000000e+00, 1.49965283e-04,
                             1.50022495e+00],
                            [0.00000000e+00, 0.00000000e+00, -1.00000000e+00,
                             0.00000000e+00]])
    ndc_points_reconst = grid_to_ndc_pcl_linear_view(positions_grid, proj_matrix, z_meters_min, z_meters_max)
    ndc_points_reconst = np.hstack((ndc_points_reconst, np.ones((ndc_points_reconst.shape[0], 1)))).T

    points = np.argwhere(positions_grid)  # now I get all coords as a list of points so I can get values by them
    view_points_reconst = ndc_to_view(ndc_points_reconst, proj_matrix)
    # print(view_points_reconst.shape)
    # print(points.shape)
    # print(np.array([[0,0,0], [1,1,1], [1,1,2], [2,1,2]]).shape)
    # print(ndc_grid.shape)
    # print(ndc_grid[[0,1,1,2], [0,1,1,1], [0,1,2,2]].shape)
    # print(view_points_reconst.T.shape)
    # print(ndc_grid[points[:, 0], points[:, 1], points[:, 2]].shape)
    # print(view_points_reconst[0:3, :].T.shape)
    # print(ndc_grid[points[:, 0], points[:, 1], points[:, 2]][:, np.newaxis].shape)
    return np.hstack((view_points_reconst[0:3, :].T, ndc_grid[points[:, 0], points[:, 1], points[:, 2]][:, np.newaxis])).T


def evaluate_depth_metrics(batch_rgb, batch_depths, model_names):
    for i in range(Network.BATCH_SIZE):
        im = Image.fromarray(batch_rgb[i, :, :, :].astype(np.uint8))
        im.save("evaluate/orig-rgb-{}.png".format(i))

        depths = batch_depths[i, :, :, :]
        # todo: implement later

    column_names = get_evaluation_names()
    column_names.append('name')
    x = PrettyTable(column_names)

    for model_name in model_names:
        pred_voxels = evaluate_model(model_name, batch_rgb)
        accuracies = get_accuracies(batch_rgb, batch_depths)

        # accuracies['name'] = model_name
        # x.add_row(accuracies.values())
        accuracies.append(model_name)
        x.add_row(accuracies)

        # saving images
        for i in range(Network.BATCH_SIZE):
            depth = pred_voxels[i, :, :, :]
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


def evaluate_voxel_metrics(batch_rgb, batch_depths, model_names):
    column_names = get_evaluation_names()
    column_names.append('name')
    column_names.append('loss')
    x = PrettyTable(column_names)

    for model_name in model_names:
        pred_voxels, loss = evaluate_model_with_loss(model_name, batch_rgb, batch_depths)
        accuracies = get_accuracies_voxel(batch_rgb, batch_depths)

        # accuracies['name'] = model_name
        # x.add_row(accuracies.values())
        accuracies.append(model_name)
        accuracies.append(loss)
        x.add_row(accuracies)

        # saving images
        for i in range(Network.BATCH_SIZE):
            depth = pred_voxels[i, :, :, :]
            if len(depth.shape) == 3 and depth.shape[2] > 1:
                raise Exception('oh, boi, shape is going wild', depth.shape)
            depth = depth[:, :, 0]

            if np.max(depth) != 0:
                depth = (depth / np.max(depth)) * 255.0
            else:
                depth = depth * 255.0
            im = Image.fromarray(depth.astype(np.uint8), mode="L")
            im.save("evaluate-voxel/predicted-{}-{}.png".format(i, model_name))

    print(x)


def predict_voxels_to_pointcloud(batch_rgb, batch_depths, model_names):
    for i in range(Network.BATCH_SIZE):
        im = Image.fromarray(batch_rgb[i, :, :, :].astype(np.uint8))
        im.save("evaluate/orig-rgb-{}.png".format(i))

        voxels = batch_depths[i, :, :, :]
        pcl = grid_voxelmap_to_pointcloud(voxels)
        save_pointcloud_csv(pcl.T[:, 0:3], "evaluate/orig-voxelmap-{}.csv".format(i))

    for model_name in model_names:
        # pred_voxels = evaluate_model(model_name, batch_rgb)
        metrics, pred_voxels = calculate_voxel_metrics(model_name, batch_rgb, batch_depths)

        print('metrics', metrics)

        # saving images
        for i in range(Network.BATCH_SIZE):
            pred_voxelmap = pred_voxels[i, :, :, :]
            np.save("evaluate/pred-voxelmap-{}-{}.npy".format(i, model_name), pred_voxelmap)
            pcl = grid_voxelmap_to_pointcloud(losses.is_obstacle(pred_voxelmap))
            pcl_values = grid_voxelmap_to_paraview_pointcloud(pred_voxelmap)
            save_pointcloud_csv(pcl.T[:, 0:3], "evaluate/pred-voxelmap-{}-{}.csv".format(i, model_name))
            save_pointcloud_csv(pcl_values.T[:, 0:4], "evaluate/pred-voxelmap-paraview-{}-{}.csv".format(i, model_name), True)


def do_metrics_evaluation():
    model_names = [
        # format is name, needs conversion from bins
        '2018-05-04--22-57-49',
        '2018-05-04--23-03-46',
        '2018-05-06--00-03-04',
        '2018-05-06--00-05-58',
        '2018-05-07--17-22-10',
        '2018-05-08--23-37-07',
        '2018-05-11--00-10-54',
    ]
    images = np.array([
        ['ml-datasets-voxel/2018-03-07--16-40-42--901.jpg', 'ml-datasets-voxel/2018-03-07--16-40-42--901.npy'],
        ['ml-datasets-voxel/2018-03-07--17-41-16--827.jpg', 'ml-datasets-voxel/2018-03-07--17-41-16--827.npy'],
        ['ml-datasets-voxel/2018-03-07--16-12-57--023.jpg', 'ml-datasets-voxel/2018-03-07--16-12-57--023.npy'],
        ['ml-datasets-voxel/2018-03-07--15-44-56--353.jpg', 'ml-datasets-voxel/2018-03-07--15-44-56--353.npy'],
    ])

    Network.BATCH_SIZE = len(images)
    ds = dataset.DataSet(len(images))
    filename_list = tf.data.Dataset.from_tensor_slices((images[:, 0], images[:, 1]))
    images, voxels, depths = ds.filenames_to_batch_voxel(filename_list)

    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    with tf.Session(config=config) as sess:
        batch_images, batch_voxels, batch_depths = sess.run(
            [images, voxels, depths])

    evaluate_voxel_metrics(batch_images, batch_voxels, model_names)


def main():
    model_names = [
        '2018-05-04--22-57-49',
        '2018-05-04--23-03-46',
        '2018-05-07--17-22-10',
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

    # evaluate_depth_metrics(batch_rgb, batch_depths, model_names)
    predict_voxels_to_pointcloud(batch_rgb, batch_depths, model_names)


if __name__ == '__main__':
    # main()
    do_metrics_evaluation()
