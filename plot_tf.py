import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set()


def plot_depths():
    all_metrics = ['cost', 'under_treshold_1.25', 'mean_relative_error', 'root_mean_square_error',
                   'root_mean_log_square_error']
    all_names = ['Loss', 'Under threshold ${\\tau=1.25}$', 'Mean Relative Error', 'RMSE',
                 'RMLSE']
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']
    runs = ['2018-03-29--12-41-37', '2018-04-01--00-25-06', '2018-04-01--00-26-49',
            '2018-04-01--00-32-39', '2018-04-02--02-51-28', '2018-04-02--02-52-07',
            '2018-04-02--02-59-31', '2018-04-05--09-15-19', '2018-04-05--09-22-22']
    for j, metrics in enumerate(all_metrics):
        directory = 'tf-dumps'

        if metrics == 'cost':
            plt.figure(figsize=(12, 6))
        else:
            plt.figure()
        for i, run in enumerate(runs):
            name = '{}/run_{}-tag-{}.csv'.format(directory, run, metrics)
            print(i + 1, name)
            df = pd.read_csv(name)
            # df['Value'].plot(label='setup {}'.format(i + 1))
            # plt.plot(x=df['Step'].values, y=df['Value'].values, label='setup {}'.format(i + 1))
            plt.plot(df['Step'], df['Value'].rolling(150, center=True, min_periods=1).mean(),
                     label='setup {}'.format(i + 1), color=colors[i])

        plt.xlim([0, 160000])
        plt.xlabel('# iterations [-]')
        plt.ylabel('value [-]')
        plt.title(all_names[j])
        plt.legend()
        # plt.show()
        plt.savefig('tf-res/depth-{}.png'.format(metrics), bbox_inches='tight')


def plot_3d():
    all_metrics = ['cost', 'false_positive_rate', 'true_positive_rate', 'iou', 'l1_dist_on_known']
    all_names = ['Loss', 'False Positive Rate', 'True Positive Rate', 'Intersection over union', '$L_1$ distance']
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628']
    runs = ['2018-05-04--22-57-49', '2018-05-04--23-03-46', '2018-05-06--00-03-04',
            '2018-05-06--00-05-58', '2018-05-07--17-22-10', '2018-05-08--23-37-07',
            '2018-05-11--00-10-54']
    plot_3d_inner(all_metrics, all_names, colors, runs, '3d')


def plot_3d_better():
    all_metrics = ['cost', 'false_positive_rate', 'true_positive_rate', 'iou', 'l1_dist_on_known']
    all_names = ['Loss', 'False Positive Rate', 'True Positive Rate', 'Intersection over union', '$L_1$ distance']
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    runs = ['2018-05-04--22-57-49', '2018-05-07--17-22-10', '2018-05-08--23-37-07',
            '2018-05-11--00-10-54']
    plot_3d_inner(all_metrics, all_names, colors, runs, '3dbetter')


def plot_3d_inner(all_metrics, all_names, colors, runs, prefix):
    for j, metrics in enumerate(all_metrics):
        directory = 'tf-dumps'

        plt.figure(figsize=(12, 6))
        for i, run in enumerate(runs):
            name = '{}/run_{}-tag-{}.csv'.format(directory, run, metrics)
            print(i + 1, name)
            df = pd.read_csv(name)
            # df['Value'].plot(label='setup {}'.format(i + 1))
            # plt.plot(x=df['Step'].values, y=df['Value'].values, label='setup {}'.format(i + 1))
            plt.plot(df['Step'], df['Value'].rolling(150, center=True, min_periods=1).mean(),
                     label='setup {}'.format(i + 1), color=colors[i])

        plt.xlim([0, 180000])
        plt.xlabel('# iterations [-]')
        plt.ylabel('value [-]')
        plt.title(all_names[j])
        plt.legend(loc=1)
        # plt.show()
        plt.savefig('tf-res/{}-{}.png'.format(prefix, metrics), bbox_inches='tight')


if __name__ == '__main__':
    plot_depths()
    plot_3d()
    plot_3d_better()
