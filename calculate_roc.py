import pickle
import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve, auc

import dataset
import losses
import metrics_np
from prettytable import PrettyTable
import os
import Network
from evaluation import load_model_with_structure, get_evaluation_names, get_accuracies, get_accuracies_voxel
from evaluation_voxels import evaluate_model
from gta_math import grid_to_ndc_pcl_linear_view, ndc_to_view
from visualization import save_pointcloud_csv
import matplotlib.pyplot as plt


def predict_voxels(batch_rgb, batch_voxels, model_names):
    results = dict()
    for model_name in model_names:
        pred_voxels = evaluate_model(model_name, batch_rgb)
        results[model_name] = pred_voxels
    return results


def calc_and_persist_roc(pred_voxels, gt_voxels, model_name, suffix):
    fpr, tpr, _ = roc_curve(gt_voxels.flatten(), pred_voxels.flatten(), 1, gt_voxels.flatten() != -1)  # because of masking
    roc_auc = auc(fpr, tpr)

    with open('evaluate/roc-{}-{}.rick'.format(model_name, suffix), 'wb+') as f:
        pickle.dump((fpr, tpr, roc_auc), f)


def main():
    model_names = [
        '2018-05-04--22-57-49',
        '2018-05-04--23-03-46',
        '2018-05-07--17-22-10',
        '2018-05-08--23-37-07',
        '2018-05-11--00-10-54'
    ]

    # counting ROC on first 20 samples from randomized dataset
    # Network.BATCH_SIZE = len(images)
    Network.BATCH_SIZE = 30
    data_set = dataset.DataSet(Network.BATCH_SIZE)

    images, voxelmaps, _ = data_set.csv_inputs_voxels(Network.TRAIN_FILE)
    images_test, voxelmaps_test, _ = data_set.csv_inputs_voxels(Network.TEST_FILE)

    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    with tf.Session(config=config) as sess:
        batch_rgb, batch_voxels = sess.run(
            [images, voxelmaps])
        batch_rgb_test, batch_voxels_test = sess.run(
            [images, voxelmaps])
    print('evaluation dataset loaded')

    results = predict_voxels(batch_rgb, batch_voxels, model_names)
    for model_name, pred_voxels in results.items():
        calc_and_persist_roc(pred_voxels, batch_voxels, model_name, 'train')

    results_test = predict_voxels(batch_rgb_test, batch_voxels_test, model_names)
    for model_name, pred_voxels in results_test.items():
        calc_and_persist_roc(pred_voxels, batch_voxels_test, model_name, 'test')


if __name__ == '__main__':
    main()
