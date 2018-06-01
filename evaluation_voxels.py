import csv
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


def calc_loss(input, rgb_image, gt_depth, model, graph, sess):
    y = graph.get_tensor_by_name('y:0')
    loss = graph.get_tensor_by_name('loss:0')
    loss_val = sess.run(loss, feed_dict={
        input: rgb_image,
        y: gt_depth
    })
    fpr, tpr, iou, softmax, l1_dist = sess.run(get_accuracies_voxel(gt_depth, model), feed_dict={
        input: rgb_image,
        y: gt_depth
    })
    return loss_val, fpr, tpr, iou, softmax, l1_dist


def evaluate_model(model_name, rgb_img):
    # not running on any GPU, using only CPU
    config = tf.ConfigProto(device_count={'GPU': 0})
    # for GPU
    # config = tf.ConfigProto(log_device_placement=False)
    # config.gpu_options.allow_growth = False
    # config.gpu_options.allocator_type = 'BFC'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
            pred_loss, fpr, tpr, iou, softmax, l1_dist = calc_loss(input, rgb_img, gt_voxel, model, graph, sess)
    return pred_voxels, pred_loss, fpr, tpr, iou, softmax, l1_dist


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
    ndc_grid = np.transpose(ndc_grid, (1, 0, 2))  # because of form it is in the output of network
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
    ndc_grid = np.transpose(ndc_grid, (1, 0, 2))  # because of form it is in the output of network
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
    return np.hstack(
        (view_points_reconst[0:3, :].T, ndc_grid[points[:, 0], points[:, 1], points[:, 2]][:, np.newaxis])).T


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
    column_names = [
        'loss',
        'fpr',
        'tpr',
        'iou',
        'softmax',
        'l1_dist',
        'name'
    ]
    x = PrettyTable(column_names)

    for model_name in model_names:
        accuracies = evaluate_model_with_loss(model_name, batch_rgb, batch_depths)
        accuracies = list(accuracies)[1:]  # because first are predicted voxels
        accuracies.append(model_name)
        x.add_row(accuracies)
        #
        # # saving images
        # for i in range(Network.BATCH_SIZE):
        #     depth = pred_voxels[i, :, :, :]
        #     if len(depth.shape) == 3 and depth.shape[2] > 1:
        #         raise Exception('oh, boi, shape is going wild', depth.shape)
        #     depth = depth[:, :, 0]
        #
        #     if np.max(depth) != 0:
        #         depth = (depth / np.max(depth)) * 255.0
        #     else:
        #         depth = depth * 255.0
        #     im = Image.fromarray(depth.astype(np.uint8), mode="L")
        #     im.save("evaluate-voxel/predicted-{}-{}.png".format(i, model_name))

    print(x)


def predict_voxels_to_pointcloud(batch_rgb, batch_depths, model_names, batch=0):
    for i in range(Network.BATCH_SIZE):
        im = Image.fromarray(batch_rgb[i, :, :, :].astype(np.uint8))
        im.save("evaluate-voxel/orig-rgb-{}-batch-{}.png".format(i, batch))

        voxels = batch_depths[i, :, :, :]
        pcl = grid_voxelmap_to_pointcloud(voxels)
        save_pointcloud_csv(pcl.T[:, 0:3], "evaluate-voxel/orig-voxelmap-{}-batch-{}.csv".format(i, batch))

    for model_name in model_names:
        pred_voxels = evaluate_model(model_name, batch_rgb)
        # metrics, pred_voxels = calculate_voxel_metrics(model_name, batch_rgb, batch_depths)
        #
        # print('metrics', metrics)

        # saving images
        for i in range(Network.BATCH_SIZE):
            pred_voxelmap = pred_voxels[i, :, :, :]
            np.save("evaluate-voxel/pred-voxelmap-{}-{}-batch-{}.npy".format(i, model_name, batch), pred_voxelmap)
            pcl = grid_voxelmap_to_pointcloud(losses.is_obstacle(pred_voxelmap))
            pcl_values = grid_voxelmap_to_paraview_pointcloud(pred_voxelmap)
            save_pointcloud_csv(pcl.T[:, 0:3], "evaluate-voxel/pred-voxelmap-{}-{}-batch-{}.csv".format(i, model_name, batch))
            save_pointcloud_csv(pcl_values.T[:, 0:4],
                                "evaluate-voxel/pred-voxelmap-paraview-{}-{}-batch-{}.csv".format(i, model_name, batch),
                                True)


def predict_voxels_to_pointcloud_multibatch(len_images, imgs, depths, model_names):
    # for CPU
    # config = tf.ConfigProto(device_count={'GPU': 0})

    # for GPU
    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = False
    config.gpu_options.allocator_type = 'BFC'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    with tf.Session(config=config) as sess:
        for batch in range(int(len_images / Network.BATCH_SIZE)):
            batch_rgb, batch_depths = sess.run(
                [imgs, depths])

            for i in range(Network.BATCH_SIZE):
                im = Image.fromarray(batch_rgb[i, :, :, :].astype(np.uint8))
                im.save("evaluate-voxel/orig-rgb-{}-batch-{}.png".format(i, batch))

                voxels = batch_depths[i, :, :, :]
                pcl = grid_voxelmap_to_pointcloud(voxels)
                save_pointcloud_csv(pcl.T[:, 0:3], "evaluate-voxel/orig-voxelmap-{}-batch-{}.csv".format(i, batch))

    print('evaluation loaded, ging to evaluation on dataset')

    for model_name in model_names:
        with tf.Graph().as_default() as graph:
            with tf.Session(config=config) as sess:
                _, input, model = load_model_with_structure(model_name, graph, sess)
                for batch in range(int(len_images / Network.BATCH_SIZE)):
                    batch_rgb, batch_depths = sess.run(
                        [imgs, depths])
                    pred_voxels = inference(model, input, batch_rgb, sess)

                    # saving images
                    for i in range(Network.BATCH_SIZE):
                        pred_voxelmap = pred_voxels[i, :, :, :]
                        np.save("evaluate-voxel/pred-voxelmap-{}-{}-batch-{}.npy".format(i, model_name, batch), pred_voxelmap)
                        pcl = grid_voxelmap_to_pointcloud(losses.is_obstacle(pred_voxelmap))
                        pcl_values = grid_voxelmap_to_paraview_pointcloud(pred_voxelmap)
                        save_pointcloud_csv(pcl.T[:, 0:3], "evaluate-voxel/pred-voxelmap-{}-{}-batch-{}.csv".format(i, model_name, batch))
                        save_pointcloud_csv(pcl_values.T[:, 0:4],
                                            "evaluate-voxel/pred-voxelmap-paraview-{}-{}-batch-{}.csv".format(i, model_name, batch),
                                            True)


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
    # images = np.array([
    #     ['ml-datasets-voxel/2018-03-07--16-40-42--901.jpg', 'ml-datasets-voxel/2018-03-07--16-40-42--901.npy'],
    #     ['ml-datasets-voxel/2018-03-07--17-41-16--827.jpg', 'ml-datasets-voxel/2018-03-07--17-41-16--827.npy'],
    #     ['ml-datasets-voxel/2018-03-07--16-12-57--023.jpg', 'ml-datasets-voxel/2018-03-07--16-12-57--023.npy'],
    #     ['ml-datasets-voxel/2018-03-07--15-44-56--353.jpg', 'ml-datasets-voxel/2018-03-07--15-44-56--353.npy'],
    # ])
    # just loading them from CSV:
    with open(Network.TEST_FILE, newline='') as csvfile:
        images = csv.reader(csvfile, delimiter=',')
        images = np.array(list(images))[0:10, :]
    print('images: ', images)

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
    # model_names = [
    #     '2018-05-04--22-57-49',
    #     '2018-05-04--23-03-46',
    #     '2018-05-07--17-22-10',
    # ]
    model_names = [
        '2018-05-04--22-57-49',
        '2018-05-11--00-10-54',
    ]

    images = np.array([
        ['ml-datasets-voxel/2018-05-24--21-28-12--997.jpg', 'ml-datasets-voxel/2018-05-24--21-28-12--997.npy'],
        ['ml-datasets-voxel/2018-05-24--21-28-15--514.jpg', 'ml-datasets-voxel/2018-05-24--21-28-15--514.npy'],
        ['ml-datasets-voxel/2018-05-24--21-28-17--558.jpg', 'ml-datasets-voxel/2018-05-24--21-28-17--558.npy'],
        ['ml-datasets-voxel/2018-05-24--21-28-36--227.jpg', 'ml-datasets-voxel/2018-05-24--21-28-36--227.npy'],
        ['ml-datasets-voxel/2018-05-24--21-28-38--166.jpg', 'ml-datasets-voxel/2018-05-24--21-28-38--166.npy'],
        ['ml-datasets-voxel/2018-05-24--21-28-40--042.jpg', 'ml-datasets-voxel/2018-05-24--21-28-40--042.npy'],
        ['ml-datasets-voxel/2018-05-24--21-29-06--939.jpg', 'ml-datasets-voxel/2018-05-24--21-29-06--939.npy'],
        ['ml-datasets-voxel/2018-05-24--21-29-08--961.jpg', 'ml-datasets-voxel/2018-05-24--21-29-08--961.npy'],
        ['ml-datasets-voxel/2018-05-24--21-29-10--969.jpg', 'ml-datasets-voxel/2018-05-24--21-29-10--969.npy'],
        ['ml-datasets-voxel/2018-05-24--21-29-20--179.jpg', 'ml-datasets-voxel/2018-05-24--21-29-20--179.npy'],
        ['ml-datasets-voxel/2018-05-24--21-29-22--135.jpg', 'ml-datasets-voxel/2018-05-24--21-29-22--135.npy'],
        ['ml-datasets-voxel/2018-05-24--21-29-24--080.jpg', 'ml-datasets-voxel/2018-05-24--21-29-24--080.npy'],
        ['ml-datasets-voxel/2018-05-24--21-30-52--087.jpg', 'ml-datasets-voxel/2018-05-24--21-30-52--087.npy'],
        ['ml-datasets-voxel/2018-05-24--21-30-54--235.jpg', 'ml-datasets-voxel/2018-05-24--21-30-54--235.npy'],
        ['ml-datasets-voxel/2018-05-24--21-30-56--609.jpg', 'ml-datasets-voxel/2018-05-24--21-30-56--609.npy'],
        ['ml-datasets-voxel/2018-05-24--21-35-19--669.jpg', 'ml-datasets-voxel/2018-05-24--21-35-19--669.npy'],
        ['ml-datasets-voxel/2018-05-24--21-35-21--579.jpg', 'ml-datasets-voxel/2018-05-24--21-35-21--579.npy'],
        ['ml-datasets-voxel/2018-05-24--21-35-30--031.jpg', 'ml-datasets-voxel/2018-05-24--21-35-30--031.npy'],
        ['ml-datasets-voxel/2018-05-24--21-35-31--992.jpg', 'ml-datasets-voxel/2018-05-24--21-35-31--992.npy'],
        ['ml-datasets-voxel/2018-05-24--21-35-33--917.jpg', 'ml-datasets-voxel/2018-05-24--21-35-33--917.npy'],
        ['ml-datasets-voxel/2018-05-24--21-37-01--072.jpg', 'ml-datasets-voxel/2018-05-24--21-37-01--072.npy'],
        ['ml-datasets-voxel/2018-05-24--21-37-03--113.jpg', 'ml-datasets-voxel/2018-05-24--21-37-03--113.npy'],
        ['ml-datasets-voxel/2018-05-24--21-37-05--137.jpg', 'ml-datasets-voxel/2018-05-24--21-37-05--137.npy'],
        ['ml-datasets-voxel/2018-05-24--21-37-12--883.jpg', 'ml-datasets-voxel/2018-05-24--21-37-12--883.npy'],
        ['ml-datasets-voxel/2018-05-24--21-37-14--886.jpg', 'ml-datasets-voxel/2018-05-24--21-37-14--886.npy'],
        ['ml-datasets-voxel/2018-05-24--21-37-16--949.jpg', 'ml-datasets-voxel/2018-05-24--21-37-16--949.npy'],
        ['ml-datasets-voxel/2018-05-24--21-37-42--667.jpg', 'ml-datasets-voxel/2018-05-24--21-37-42--667.npy'],
        ['ml-datasets-voxel/2018-05-24--21-37-44--690.jpg', 'ml-datasets-voxel/2018-05-24--21-37-44--690.npy'],
        ['ml-datasets-voxel/2018-05-24--21-37-46--636.jpg', 'ml-datasets-voxel/2018-05-24--21-37-46--636.npy'],
        ['ml-datasets-voxel/2018-05-24--21-38-46--411.jpg', 'ml-datasets-voxel/2018-05-24--21-38-46--411.npy'],
        ['ml-datasets-voxel/2018-05-24--21-38-48--403.jpg', 'ml-datasets-voxel/2018-05-24--21-38-48--403.npy'],
        ['ml-datasets-voxel/2018-05-24--21-38-50--337.jpg', 'ml-datasets-voxel/2018-05-24--21-38-50--337.npy'],
        ['ml-datasets-voxel/2018-05-24--21-38-57--770.jpg', 'ml-datasets-voxel/2018-05-24--21-38-57--770.npy'],
        ['ml-datasets-voxel/2018-05-24--21-38-59--953.jpg', 'ml-datasets-voxel/2018-05-24--21-38-59--953.npy'],
        ['ml-datasets-voxel/2018-05-24--21-39-02--104.jpg', 'ml-datasets-voxel/2018-05-24--21-39-02--104.npy'],
        ['ml-datasets-voxel/2018-05-24--21-43-43--995.jpg', 'ml-datasets-voxel/2018-05-24--21-43-43--995.npy'],
        ['ml-datasets-voxel/2018-05-24--21-43-45--818.jpg', 'ml-datasets-voxel/2018-05-24--21-43-45--818.npy'],
        ['ml-datasets-voxel/2018-05-24--21-43-47--596.jpg', 'ml-datasets-voxel/2018-05-24--21-43-47--596.npy'],
        ['ml-datasets-voxel/2018-05-24--21-44-00--765.jpg', 'ml-datasets-voxel/2018-05-24--21-44-00--765.npy'],
        ['ml-datasets-voxel/2018-05-24--21-44-02--579.jpg', 'ml-datasets-voxel/2018-05-24--21-44-02--579.npy'],
        ['ml-datasets-voxel/2018-05-24--21-44-04--343.jpg', 'ml-datasets-voxel/2018-05-24--21-44-04--343.npy'],
        ['ml-datasets-voxel/2018-05-24--21-44-28--064.jpg', 'ml-datasets-voxel/2018-05-24--21-44-28--064.npy'],
        ['ml-datasets-voxel/2018-05-24--21-44-29--900.jpg', 'ml-datasets-voxel/2018-05-24--21-44-29--900.npy'],
        ['ml-datasets-voxel/2018-05-24--21-44-31--704.jpg', 'ml-datasets-voxel/2018-05-24--21-44-31--704.npy'],
        ['ml-datasets-voxel/2018-05-24--21-44-35--645.jpg', 'ml-datasets-voxel/2018-05-24--21-44-35--645.npy'],
        ['ml-datasets-voxel/2018-05-24--21-44-37--470.jpg', 'ml-datasets-voxel/2018-05-24--21-44-37--470.npy'],
        ['ml-datasets-voxel/2018-05-24--21-44-39--226.jpg', 'ml-datasets-voxel/2018-05-24--21-44-39--226.npy'],
        ['ml-datasets-voxel/2018-05-24--21-44-58--176.jpg', 'ml-datasets-voxel/2018-05-24--21-44-58--176.npy'],
        ['ml-datasets-voxel/2018-05-24--21-44-59--942.jpg', 'ml-datasets-voxel/2018-05-24--21-44-59--942.npy'],
        ['ml-datasets-voxel/2018-05-24--21-45-01--741.jpg', 'ml-datasets-voxel/2018-05-24--21-45-01--741.npy'],
        ['ml-datasets-voxel/2018-05-24--21-45-44--259.jpg', 'ml-datasets-voxel/2018-05-24--21-45-44--259.npy'],
        ['ml-datasets-voxel/2018-05-24--21-45-46--086.jpg', 'ml-datasets-voxel/2018-05-24--21-45-46--086.npy'],
        ['ml-datasets-voxel/2018-05-24--21-45-47--880.jpg', 'ml-datasets-voxel/2018-05-24--21-45-47--880.npy'],
        ['ml-datasets-voxel/2018-05-24--21-47-25--657.jpg', 'ml-datasets-voxel/2018-05-24--21-47-25--657.npy'],
        ['ml-datasets-voxel/2018-05-24--21-47-27--591.jpg', 'ml-datasets-voxel/2018-05-24--21-47-27--591.npy'],
        ['ml-datasets-voxel/2018-05-24--21-47-29--373.jpg', 'ml-datasets-voxel/2018-05-24--21-47-29--373.npy'],
        ['ml-datasets-voxel/2018-05-24--21-48-04--689.jpg', 'ml-datasets-voxel/2018-05-24--21-48-04--689.npy'],
        ['ml-datasets-voxel/2018-05-24--21-48-06--535.jpg', 'ml-datasets-voxel/2018-05-24--21-48-06--535.npy'],
        ['ml-datasets-voxel/2018-05-24--21-48-08--373.jpg', 'ml-datasets-voxel/2018-05-24--21-48-08--373.npy'],
        ['ml-datasets-voxel/2018-05-24--21-48-31--013.jpg', 'ml-datasets-voxel/2018-05-24--21-48-31--013.npy'],
        ['ml-datasets-voxel/2018-05-24--21-48-33--026.jpg', 'ml-datasets-voxel/2018-05-24--21-48-33--026.npy'],
        ['ml-datasets-voxel/2018-05-24--21-48-35--098.jpg', 'ml-datasets-voxel/2018-05-24--21-48-35--098.npy'],
        ['ml-datasets-voxel/2018-05-24--21-48-47--614.jpg', 'ml-datasets-voxel/2018-05-24--21-48-47--614.npy'],
        ['ml-datasets-voxel/2018-05-24--21-48-49--538.jpg', 'ml-datasets-voxel/2018-05-24--21-48-49--538.npy'],
        ['ml-datasets-voxel/2018-05-24--21-48-51--452.jpg', 'ml-datasets-voxel/2018-05-24--21-48-51--452.npy'],
        ['ml-datasets-voxel/2018-05-24--21-49-30--789.jpg', 'ml-datasets-voxel/2018-05-24--21-49-30--789.npy'],
        ['ml-datasets-voxel/2018-05-24--21-49-32--627.jpg', 'ml-datasets-voxel/2018-05-24--21-49-32--627.npy'],
        ['ml-datasets-voxel/2018-05-24--21-49-34--651.jpg', 'ml-datasets-voxel/2018-05-24--21-49-34--651.npy'],
        ['ml-datasets-voxel/2018-05-24--21-51-14--642.jpg', 'ml-datasets-voxel/2018-05-24--21-51-14--642.npy'],
        ['ml-datasets-voxel/2018-05-24--21-51-16--415.jpg', 'ml-datasets-voxel/2018-05-24--21-51-16--415.npy'],
        ['ml-datasets-voxel/2018-05-24--21-51-18--158.jpg', 'ml-datasets-voxel/2018-05-24--21-51-18--158.npy'],
        ['ml-datasets-voxel/2018-05-24--21-52-06--218.jpg', 'ml-datasets-voxel/2018-05-24--21-52-06--218.npy'],
        ['ml-datasets-voxel/2018-05-24--21-52-07--985.jpg', 'ml-datasets-voxel/2018-05-24--21-52-07--985.npy'],
        ['ml-datasets-voxel/2018-05-24--21-52-09--770.jpg', 'ml-datasets-voxel/2018-05-24--21-52-09--770.npy'],
        ['ml-datasets-voxel/2018-05-24--21-54-17--324.jpg', 'ml-datasets-voxel/2018-05-24--21-54-17--324.npy'],
        ['ml-datasets-voxel/2018-05-24--21-54-19--088.jpg', 'ml-datasets-voxel/2018-05-24--21-54-19--088.npy'],
        ['ml-datasets-voxel/2018-05-24--21-54-20--839.jpg', 'ml-datasets-voxel/2018-05-24--21-54-20--839.npy'],
        ['ml-datasets-voxel/2018-05-24--21-54-44--776.jpg', 'ml-datasets-voxel/2018-05-24--21-54-44--776.npy'],
        ['ml-datasets-voxel/2018-05-24--21-54-46--356.jpg', 'ml-datasets-voxel/2018-05-24--21-54-46--356.npy'],
        ['ml-datasets-voxel/2018-05-24--21-54-48--092.jpg', 'ml-datasets-voxel/2018-05-24--21-54-48--092.npy'],
        ['ml-datasets-voxel/2018-05-24--21-55-06--980.jpg', 'ml-datasets-voxel/2018-05-24--21-55-06--980.npy'],
        ['ml-datasets-voxel/2018-05-24--21-55-08--756.jpg', 'ml-datasets-voxel/2018-05-24--21-55-08--756.npy'],
        ['ml-datasets-voxel/2018-05-24--21-55-10--507.jpg', 'ml-datasets-voxel/2018-05-24--21-55-10--507.npy'],
        ['ml-datasets-voxel/2018-05-24--21-56-06--683.jpg', 'ml-datasets-voxel/2018-05-24--21-56-06--683.npy'],
        ['ml-datasets-voxel/2018-05-24--21-56-08--477.jpg', 'ml-datasets-voxel/2018-05-24--21-56-08--477.npy'],
        ['ml-datasets-voxel/2018-05-24--21-56-10--255.jpg', 'ml-datasets-voxel/2018-05-24--21-56-10--255.npy'],
        ['ml-datasets-voxel/2018-05-24--21-58-02--783.jpg', 'ml-datasets-voxel/2018-05-24--21-58-02--783.npy'],
        ['ml-datasets-voxel/2018-05-24--21-58-04--606.jpg', 'ml-datasets-voxel/2018-05-24--21-58-04--606.npy'],
        ['ml-datasets-voxel/2018-05-24--21-58-06--393.jpg', 'ml-datasets-voxel/2018-05-24--21-58-06--393.npy'],
        ['ml-datasets-voxel/2018-05-24--21-58-29--042.jpg', 'ml-datasets-voxel/2018-05-24--21-58-29--042.npy'],
        ['ml-datasets-voxel/2018-05-24--21-58-30--858.jpg', 'ml-datasets-voxel/2018-05-24--21-58-30--858.npy'],
        ['ml-datasets-voxel/2018-05-24--21-58-32--671.jpg', 'ml-datasets-voxel/2018-05-24--21-58-32--671.npy'],
        ['ml-datasets-voxel/2018-05-24--21-59-25--558.jpg', 'ml-datasets-voxel/2018-05-24--21-59-25--558.npy'],
        ['ml-datasets-voxel/2018-05-24--21-59-27--352.jpg', 'ml-datasets-voxel/2018-05-24--21-59-27--352.npy'],
        ['ml-datasets-voxel/2018-05-24--21-59-29--145.jpg', 'ml-datasets-voxel/2018-05-24--21-59-29--145.npy'],
        ['ml-datasets-voxel/2018-05-24--22-00-13--004.jpg', 'ml-datasets-voxel/2018-05-24--22-00-13--004.npy'],
        ['ml-datasets-voxel/2018-05-24--22-00-14--770.jpg', 'ml-datasets-voxel/2018-05-24--22-00-14--770.npy'],
        ['ml-datasets-voxel/2018-05-24--22-00-16--565.jpg', 'ml-datasets-voxel/2018-05-24--22-00-16--565.npy'],
        ['ml-datasets-voxel/2018-05-24--22-00-51--441.jpg', 'ml-datasets-voxel/2018-05-24--22-00-51--441.npy'],
        ['ml-datasets-voxel/2018-05-24--22-00-53--464.jpg', 'ml-datasets-voxel/2018-05-24--22-00-53--464.npy'],
        ['ml-datasets-voxel/2018-05-24--22-00-55--216.jpg', 'ml-datasets-voxel/2018-05-24--22-00-55--216.npy'],
        ['ml-datasets-voxel/2018-05-24--22-01-14--663.jpg', 'ml-datasets-voxel/2018-05-24--22-01-14--663.npy'],
        ['ml-datasets-voxel/2018-05-24--22-01-16--858.jpg', 'ml-datasets-voxel/2018-05-24--22-01-16--858.npy'],
        ['ml-datasets-voxel/2018-05-24--22-01-18--611.jpg', 'ml-datasets-voxel/2018-05-24--22-01-18--611.npy'],
        ['ml-datasets-voxel/2018-05-24--22-05-05--408.jpg', 'ml-datasets-voxel/2018-05-24--22-05-05--408.npy'],
        ['ml-datasets-voxel/2018-05-24--22-05-07--398.jpg', 'ml-datasets-voxel/2018-05-24--22-05-07--398.npy'],
        ['ml-datasets-voxel/2018-05-24--22-05-09--165.jpg', 'ml-datasets-voxel/2018-05-24--22-05-09--165.npy'],
        ['ml-datasets-voxel/2018-05-24--22-05-40--872.jpg', 'ml-datasets-voxel/2018-05-24--22-05-40--872.npy'],
        ['ml-datasets-voxel/2018-05-24--22-05-42--700.jpg', 'ml-datasets-voxel/2018-05-24--22-05-42--700.npy'],
        ['ml-datasets-voxel/2018-05-24--22-05-44--497.jpg', 'ml-datasets-voxel/2018-05-24--22-05-44--497.npy'],
    ])

    Network.BATCH_SIZE = len(images)
    ds = dataset.DataSet(len(images))
    filename_list = tf.data.Dataset.from_tensor_slices((images[:, 0], images[:, 1]))
    imgs, depths, _ = ds.filenames_to_batch_voxel(filename_list)
    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    with tf.Session(config=config) as sess:
        batch_rgb, batch_depths = sess.run(
            [imgs, depths])
    print('evaluation dataset loaded')

    # evaluate_depth_metrics(batch_rgb, batch_depths, model_names)
    predict_voxels_to_pointcloud(batch_rgb, batch_depths, model_names)


def main_multibatch():
    model_names = [
        '2018-05-04--22-57-49',
        '2018-05-11--00-10-54',
    ]

    images = np.array([
        ['ml-datasets-voxel/2018-05-24--21-28-12--997.jpg', 'ml-datasets-voxel/2018-05-24--21-28-12--997.npy'],
        ['ml-datasets-voxel/2018-05-24--21-28-15--514.jpg', 'ml-datasets-voxel/2018-05-24--21-28-15--514.npy'],
        ['ml-datasets-voxel/2018-05-24--21-28-17--558.jpg', 'ml-datasets-voxel/2018-05-24--21-28-17--558.npy'],
        ['ml-datasets-voxel/2018-05-24--21-28-36--227.jpg', 'ml-datasets-voxel/2018-05-24--21-28-36--227.npy'],
        ['ml-datasets-voxel/2018-05-24--21-28-38--166.jpg', 'ml-datasets-voxel/2018-05-24--21-28-38--166.npy'],
        ['ml-datasets-voxel/2018-05-24--21-28-40--042.jpg', 'ml-datasets-voxel/2018-05-24--21-28-40--042.npy'],
        ['ml-datasets-voxel/2018-05-24--21-29-06--939.jpg', 'ml-datasets-voxel/2018-05-24--21-29-06--939.npy'],
        ['ml-datasets-voxel/2018-05-24--21-29-08--961.jpg', 'ml-datasets-voxel/2018-05-24--21-29-08--961.npy'],
        ['ml-datasets-voxel/2018-05-24--21-29-10--969.jpg', 'ml-datasets-voxel/2018-05-24--21-29-10--969.npy'],
        ['ml-datasets-voxel/2018-05-24--21-29-20--179.jpg', 'ml-datasets-voxel/2018-05-24--21-29-20--179.npy'],
        ['ml-datasets-voxel/2018-05-24--21-29-22--135.jpg', 'ml-datasets-voxel/2018-05-24--21-29-22--135.npy'],
        ['ml-datasets-voxel/2018-05-24--21-29-24--080.jpg', 'ml-datasets-voxel/2018-05-24--21-29-24--080.npy'],
        ['ml-datasets-voxel/2018-05-24--21-30-52--087.jpg', 'ml-datasets-voxel/2018-05-24--21-30-52--087.npy'],
        ['ml-datasets-voxel/2018-05-24--21-30-54--235.jpg', 'ml-datasets-voxel/2018-05-24--21-30-54--235.npy'],
        ['ml-datasets-voxel/2018-05-24--21-30-56--609.jpg', 'ml-datasets-voxel/2018-05-24--21-30-56--609.npy'],
        ['ml-datasets-voxel/2018-05-24--21-35-19--669.jpg', 'ml-datasets-voxel/2018-05-24--21-35-19--669.npy'],
        ['ml-datasets-voxel/2018-05-24--21-35-21--579.jpg', 'ml-datasets-voxel/2018-05-24--21-35-21--579.npy'],
        ['ml-datasets-voxel/2018-05-24--21-35-30--031.jpg', 'ml-datasets-voxel/2018-05-24--21-35-30--031.npy'],
        ['ml-datasets-voxel/2018-05-24--21-35-31--992.jpg', 'ml-datasets-voxel/2018-05-24--21-35-31--992.npy'],
        ['ml-datasets-voxel/2018-05-24--21-35-33--917.jpg', 'ml-datasets-voxel/2018-05-24--21-35-33--917.npy'],
        ['ml-datasets-voxel/2018-05-24--21-37-01--072.jpg', 'ml-datasets-voxel/2018-05-24--21-37-01--072.npy'],
        ['ml-datasets-voxel/2018-05-24--21-37-03--113.jpg', 'ml-datasets-voxel/2018-05-24--21-37-03--113.npy'],
        ['ml-datasets-voxel/2018-05-24--21-37-05--137.jpg', 'ml-datasets-voxel/2018-05-24--21-37-05--137.npy'],
        ['ml-datasets-voxel/2018-05-24--21-37-12--883.jpg', 'ml-datasets-voxel/2018-05-24--21-37-12--883.npy'],
        ['ml-datasets-voxel/2018-05-24--21-37-14--886.jpg', 'ml-datasets-voxel/2018-05-24--21-37-14--886.npy'],
        ['ml-datasets-voxel/2018-05-24--21-37-16--949.jpg', 'ml-datasets-voxel/2018-05-24--21-37-16--949.npy'],
        ['ml-datasets-voxel/2018-05-24--21-37-42--667.jpg', 'ml-datasets-voxel/2018-05-24--21-37-42--667.npy'],
        ['ml-datasets-voxel/2018-05-24--21-37-44--690.jpg', 'ml-datasets-voxel/2018-05-24--21-37-44--690.npy'],
        ['ml-datasets-voxel/2018-05-24--21-37-46--636.jpg', 'ml-datasets-voxel/2018-05-24--21-37-46--636.npy'],
        ['ml-datasets-voxel/2018-05-24--21-38-46--411.jpg', 'ml-datasets-voxel/2018-05-24--21-38-46--411.npy'],
        ['ml-datasets-voxel/2018-05-24--21-38-48--403.jpg', 'ml-datasets-voxel/2018-05-24--21-38-48--403.npy'],
        ['ml-datasets-voxel/2018-05-24--21-38-50--337.jpg', 'ml-datasets-voxel/2018-05-24--21-38-50--337.npy'],
        ['ml-datasets-voxel/2018-05-24--21-38-57--770.jpg', 'ml-datasets-voxel/2018-05-24--21-38-57--770.npy'],
        ['ml-datasets-voxel/2018-05-24--21-38-59--953.jpg', 'ml-datasets-voxel/2018-05-24--21-38-59--953.npy'],
        ['ml-datasets-voxel/2018-05-24--21-39-02--104.jpg', 'ml-datasets-voxel/2018-05-24--21-39-02--104.npy'],
        ['ml-datasets-voxel/2018-05-24--21-43-43--995.jpg', 'ml-datasets-voxel/2018-05-24--21-43-43--995.npy'],
        ['ml-datasets-voxel/2018-05-24--21-43-45--818.jpg', 'ml-datasets-voxel/2018-05-24--21-43-45--818.npy'],
        ['ml-datasets-voxel/2018-05-24--21-43-47--596.jpg', 'ml-datasets-voxel/2018-05-24--21-43-47--596.npy'],
        ['ml-datasets-voxel/2018-05-24--21-44-00--765.jpg', 'ml-datasets-voxel/2018-05-24--21-44-00--765.npy'],
        ['ml-datasets-voxel/2018-05-24--21-44-02--579.jpg', 'ml-datasets-voxel/2018-05-24--21-44-02--579.npy'],
        ['ml-datasets-voxel/2018-05-24--21-44-04--343.jpg', 'ml-datasets-voxel/2018-05-24--21-44-04--343.npy'],
        ['ml-datasets-voxel/2018-05-24--21-44-28--064.jpg', 'ml-datasets-voxel/2018-05-24--21-44-28--064.npy'],
        ['ml-datasets-voxel/2018-05-24--21-44-29--900.jpg', 'ml-datasets-voxel/2018-05-24--21-44-29--900.npy'],
        ['ml-datasets-voxel/2018-05-24--21-44-31--704.jpg', 'ml-datasets-voxel/2018-05-24--21-44-31--704.npy'],
        ['ml-datasets-voxel/2018-05-24--21-44-35--645.jpg', 'ml-datasets-voxel/2018-05-24--21-44-35--645.npy'],
        ['ml-datasets-voxel/2018-05-24--21-44-37--470.jpg', 'ml-datasets-voxel/2018-05-24--21-44-37--470.npy'],
        ['ml-datasets-voxel/2018-05-24--21-44-39--226.jpg', 'ml-datasets-voxel/2018-05-24--21-44-39--226.npy'],
        ['ml-datasets-voxel/2018-05-24--21-44-58--176.jpg', 'ml-datasets-voxel/2018-05-24--21-44-58--176.npy'],
        ['ml-datasets-voxel/2018-05-24--21-44-59--942.jpg', 'ml-datasets-voxel/2018-05-24--21-44-59--942.npy'],
        ['ml-datasets-voxel/2018-05-24--21-45-01--741.jpg', 'ml-datasets-voxel/2018-05-24--21-45-01--741.npy'],
        ['ml-datasets-voxel/2018-05-24--21-45-44--259.jpg', 'ml-datasets-voxel/2018-05-24--21-45-44--259.npy'],
        ['ml-datasets-voxel/2018-05-24--21-45-46--086.jpg', 'ml-datasets-voxel/2018-05-24--21-45-46--086.npy'],
        ['ml-datasets-voxel/2018-05-24--21-45-47--880.jpg', 'ml-datasets-voxel/2018-05-24--21-45-47--880.npy'],
        ['ml-datasets-voxel/2018-05-24--21-47-25--657.jpg', 'ml-datasets-voxel/2018-05-24--21-47-25--657.npy'],
        ['ml-datasets-voxel/2018-05-24--21-47-27--591.jpg', 'ml-datasets-voxel/2018-05-24--21-47-27--591.npy'],
        ['ml-datasets-voxel/2018-05-24--21-47-29--373.jpg', 'ml-datasets-voxel/2018-05-24--21-47-29--373.npy'],
        ['ml-datasets-voxel/2018-05-24--21-48-04--689.jpg', 'ml-datasets-voxel/2018-05-24--21-48-04--689.npy'],
        ['ml-datasets-voxel/2018-05-24--21-48-06--535.jpg', 'ml-datasets-voxel/2018-05-24--21-48-06--535.npy'],
        ['ml-datasets-voxel/2018-05-24--21-48-08--373.jpg', 'ml-datasets-voxel/2018-05-24--21-48-08--373.npy'],
        ['ml-datasets-voxel/2018-05-24--21-48-31--013.jpg', 'ml-datasets-voxel/2018-05-24--21-48-31--013.npy'],
        ['ml-datasets-voxel/2018-05-24--21-48-33--026.jpg', 'ml-datasets-voxel/2018-05-24--21-48-33--026.npy'],
        ['ml-datasets-voxel/2018-05-24--21-48-35--098.jpg', 'ml-datasets-voxel/2018-05-24--21-48-35--098.npy'],
        ['ml-datasets-voxel/2018-05-24--21-48-47--614.jpg', 'ml-datasets-voxel/2018-05-24--21-48-47--614.npy'],
        ['ml-datasets-voxel/2018-05-24--21-48-49--538.jpg', 'ml-datasets-voxel/2018-05-24--21-48-49--538.npy'],
        ['ml-datasets-voxel/2018-05-24--21-48-51--452.jpg', 'ml-datasets-voxel/2018-05-24--21-48-51--452.npy'],
        ['ml-datasets-voxel/2018-05-24--21-49-30--789.jpg', 'ml-datasets-voxel/2018-05-24--21-49-30--789.npy'],
        ['ml-datasets-voxel/2018-05-24--21-49-32--627.jpg', 'ml-datasets-voxel/2018-05-24--21-49-32--627.npy'],
        ['ml-datasets-voxel/2018-05-24--21-49-34--651.jpg', 'ml-datasets-voxel/2018-05-24--21-49-34--651.npy'],
        ['ml-datasets-voxel/2018-05-24--21-51-14--642.jpg', 'ml-datasets-voxel/2018-05-24--21-51-14--642.npy'],
        ['ml-datasets-voxel/2018-05-24--21-51-16--415.jpg', 'ml-datasets-voxel/2018-05-24--21-51-16--415.npy'],
        ['ml-datasets-voxel/2018-05-24--21-51-18--158.jpg', 'ml-datasets-voxel/2018-05-24--21-51-18--158.npy'],
        ['ml-datasets-voxel/2018-05-24--21-52-06--218.jpg', 'ml-datasets-voxel/2018-05-24--21-52-06--218.npy'],
        ['ml-datasets-voxel/2018-05-24--21-52-07--985.jpg', 'ml-datasets-voxel/2018-05-24--21-52-07--985.npy'],
        ['ml-datasets-voxel/2018-05-24--21-52-09--770.jpg', 'ml-datasets-voxel/2018-05-24--21-52-09--770.npy'],
        ['ml-datasets-voxel/2018-05-24--21-54-17--324.jpg', 'ml-datasets-voxel/2018-05-24--21-54-17--324.npy'],
        ['ml-datasets-voxel/2018-05-24--21-54-19--088.jpg', 'ml-datasets-voxel/2018-05-24--21-54-19--088.npy'],
        ['ml-datasets-voxel/2018-05-24--21-54-20--839.jpg', 'ml-datasets-voxel/2018-05-24--21-54-20--839.npy'],
        ['ml-datasets-voxel/2018-05-24--21-54-44--776.jpg', 'ml-datasets-voxel/2018-05-24--21-54-44--776.npy'],
        ['ml-datasets-voxel/2018-05-24--21-54-46--356.jpg', 'ml-datasets-voxel/2018-05-24--21-54-46--356.npy'],
        ['ml-datasets-voxel/2018-05-24--21-54-48--092.jpg', 'ml-datasets-voxel/2018-05-24--21-54-48--092.npy'],
        ['ml-datasets-voxel/2018-05-24--21-55-06--980.jpg', 'ml-datasets-voxel/2018-05-24--21-55-06--980.npy'],
        ['ml-datasets-voxel/2018-05-24--21-55-08--756.jpg', 'ml-datasets-voxel/2018-05-24--21-55-08--756.npy'],
        ['ml-datasets-voxel/2018-05-24--21-55-10--507.jpg', 'ml-datasets-voxel/2018-05-24--21-55-10--507.npy'],
        ['ml-datasets-voxel/2018-05-24--21-56-06--683.jpg', 'ml-datasets-voxel/2018-05-24--21-56-06--683.npy'],
        ['ml-datasets-voxel/2018-05-24--21-56-08--477.jpg', 'ml-datasets-voxel/2018-05-24--21-56-08--477.npy'],
        ['ml-datasets-voxel/2018-05-24--21-56-10--255.jpg', 'ml-datasets-voxel/2018-05-24--21-56-10--255.npy'],
        ['ml-datasets-voxel/2018-05-24--21-58-02--783.jpg', 'ml-datasets-voxel/2018-05-24--21-58-02--783.npy'],
        ['ml-datasets-voxel/2018-05-24--21-58-04--606.jpg', 'ml-datasets-voxel/2018-05-24--21-58-04--606.npy'],
        ['ml-datasets-voxel/2018-05-24--21-58-06--393.jpg', 'ml-datasets-voxel/2018-05-24--21-58-06--393.npy'],
        ['ml-datasets-voxel/2018-05-24--21-58-29--042.jpg', 'ml-datasets-voxel/2018-05-24--21-58-29--042.npy'],
        ['ml-datasets-voxel/2018-05-24--21-58-30--858.jpg', 'ml-datasets-voxel/2018-05-24--21-58-30--858.npy'],
        ['ml-datasets-voxel/2018-05-24--21-58-32--671.jpg', 'ml-datasets-voxel/2018-05-24--21-58-32--671.npy'],
        ['ml-datasets-voxel/2018-05-24--21-59-25--558.jpg', 'ml-datasets-voxel/2018-05-24--21-59-25--558.npy'],
        ['ml-datasets-voxel/2018-05-24--21-59-27--352.jpg', 'ml-datasets-voxel/2018-05-24--21-59-27--352.npy'],
        ['ml-datasets-voxel/2018-05-24--21-59-29--145.jpg', 'ml-datasets-voxel/2018-05-24--21-59-29--145.npy'],
        ['ml-datasets-voxel/2018-05-24--22-00-13--004.jpg', 'ml-datasets-voxel/2018-05-24--22-00-13--004.npy'],
        ['ml-datasets-voxel/2018-05-24--22-00-14--770.jpg', 'ml-datasets-voxel/2018-05-24--22-00-14--770.npy'],
        ['ml-datasets-voxel/2018-05-24--22-00-16--565.jpg', 'ml-datasets-voxel/2018-05-24--22-00-16--565.npy'],
        ['ml-datasets-voxel/2018-05-24--22-00-51--441.jpg', 'ml-datasets-voxel/2018-05-24--22-00-51--441.npy'],
        ['ml-datasets-voxel/2018-05-24--22-00-53--464.jpg', 'ml-datasets-voxel/2018-05-24--22-00-53--464.npy'],
        ['ml-datasets-voxel/2018-05-24--22-00-55--216.jpg', 'ml-datasets-voxel/2018-05-24--22-00-55--216.npy'],
        ['ml-datasets-voxel/2018-05-24--22-01-14--663.jpg', 'ml-datasets-voxel/2018-05-24--22-01-14--663.npy'],
        ['ml-datasets-voxel/2018-05-24--22-01-16--858.jpg', 'ml-datasets-voxel/2018-05-24--22-01-16--858.npy'],
        ['ml-datasets-voxel/2018-05-24--22-01-18--611.jpg', 'ml-datasets-voxel/2018-05-24--22-01-18--611.npy'],
        ['ml-datasets-voxel/2018-05-24--22-05-05--408.jpg', 'ml-datasets-voxel/2018-05-24--22-05-05--408.npy'],
        ['ml-datasets-voxel/2018-05-24--22-05-07--398.jpg', 'ml-datasets-voxel/2018-05-24--22-05-07--398.npy'],
        ['ml-datasets-voxel/2018-05-24--22-05-09--165.jpg', 'ml-datasets-voxel/2018-05-24--22-05-09--165.npy'],
        ['ml-datasets-voxel/2018-05-24--22-05-40--872.jpg', 'ml-datasets-voxel/2018-05-24--22-05-40--872.npy'],
        ['ml-datasets-voxel/2018-05-24--22-05-42--700.jpg', 'ml-datasets-voxel/2018-05-24--22-05-42--700.npy'],
        ['ml-datasets-voxel/2018-05-24--22-05-44--497.jpg', 'ml-datasets-voxel/2018-05-24--22-05-44--497.npy'],
    ])

    # Network.BATCH_SIZE = len(images)
    Network.BATCH_SIZE = 4
    ds = dataset.DataSet(len(images))
    filename_list = tf.data.Dataset.from_tensor_slices((images[:, 0], images[:, 1]))
    imgs, depths, _ = ds.filenames_to_batch_voxel(filename_list)

    predict_voxels_to_pointcloud_multibatch(len(images), imgs, depths, model_names)


if __name__ == '__main__':
    main()
    # do_metrics_evaluation()
