import scipy.special
import scipy.io
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import roc_curve, auc
import numpy as np


def plot_roc(fpr, tpr, roc_auc, model_name):
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('{}.png'.format(model_name))


def process_calculated_all_roc(model_names):
    for model_name in model_names:
        with open('evaluate/roc-{}-train.rick'.format(model_name), 'rb') as f:
            fpr, tpr, roc_auc = pickle.load(f)

        plot_roc(fpr, tpr, roc_auc, model_name+'-train')

        with open('evaluate/roc-{}-test.rick'.format(model_name), 'rb') as f:
            fpr, tpr, roc_auc = pickle.load(f)

        plot_roc(fpr, tpr, roc_auc, model_name+'-test')


def calc_roc(pred_voxels, gt_voxels, model_name, suffix):
    print('calculating roc for', model_name, suffix)
    known_mask = gt_voxels.flatten() != -1
    gt_to_roc = gt_voxels.flatten()[known_mask]
    # gt_to_roc[gt_to_roc == 0] = -1
    pred_to_roc = pred_voxels.flatten()[known_mask]
    print('size to roc', gt_to_roc.shape, pred_to_roc.shape)
    # normalization of predictions to [0,1] range
    pred_to_roc = scipy.special.expit(pred_to_roc)

    num_free = np.sum(gt_to_roc == 0)
    num_occup = np.sum(gt_to_roc == 1)
    print(num_free)
    print(num_occup)
    # weights = np.ones_like(gt_to_roc, dtype=np.float32)
    # weights[gt_to_roc == -1] = 1/num_free
    # weights[gt_to_roc == 1] = 1/num_occup
    # fpr, tpr, _ = roc_curve(gt_voxels.flatten(), pred_voxels.flatten(), 1, gt_voxels.flatten() != -1)  # because of masking
    # fpr, tpr, _ = roc_curve(gt_to_roc, pred_to_roc, 1, weights)
    fpr, tpr, _ = roc_curve(gt_to_roc, pred_to_roc, 1)
    roc_auc = auc(fpr, tpr)
    plot_roc(fpr, tpr, roc_auc, model_name+'-'+suffix)


def calculate_all_roc():
    # with open('evaluate/roc-dump-gt.rick', 'rb') as f:
    #     batch_voxels = pickle.load(f)
    with open('evaluate/roc-dump-gt-test.rick', 'rb') as f:
        batch_voxels_test = pickle.load(f)
    # with open('evaluate/roc-dump-train.rick', 'rb') as f:
    #     results = pickle.load(f)
    with open('evaluate/roc-dump-test.rick', 'rb') as f:
        results_test = pickle.load(f)

    print('data loaded, going to process')
    # for model_name, res in results.items():
    #     pred_voxels, fn_val, tn_val, tp_val, fp_val = res
    #     calc_roc(pred_voxels, batch_voxels, model_name, 'train')

    scipy.io.savemat('voxel_gt.mat', {'voxel_gt': batch_voxels_test})
    for model_name, res in results_test.items():
        pred_voxels, fn_val, tn_val, tp_val, fp_val = res
        scipy.io.savemat('voxel_test.mat', {'voxel_pred': pred_voxels})
        calc_roc(pred_voxels, batch_voxels_test, model_name, 'test')
        break


def print_rates(model_names):
    for model_name in model_names:
        with open('evaluate/rates-{}-train.rick'.format(model_name), 'rb') as f:
            fn, tn, tp, fp = pickle.load(f)
        print('model {}, train'.format(model_name))
        print('fn: {}, tn: {}, tp: {}, fp: {}'.format(fn, tn, tp, fp))
        fpr = fp / (fp + tn)
        tpr = tp / (fn + tp)
        print('fpr: {}, tpr: {}'.format(fpr, tpr))

        with open('evaluate/rates-{}-test.rick'.format(model_name), 'rb') as f:
            fn, tn, tp, fp = pickle.load(f)
        print('model {}, test'.format(model_name))
        print('fn: {}, tn: {}, tp: {}, fp: {}'.format(fn, tn, tp, fp))
        fpr = fp / (fp + tn)
        tpr = tp / (fn + tp)
        print('fpr: {}, tpr: {}'.format(fpr, tpr))


if __name__ == '__main__':
    model_names = [
        '2018-05-04--22-57-49',
        '2018-05-04--23-03-46',
        '2018-05-07--17-22-10',
        '2018-05-08--23-37-07',
        '2018-05-11--00-10-54',
    ]

    # process_calculated_all_roc(model_names)
    calculate_all_roc()
    # print_rates(model_names)