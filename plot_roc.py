import matplotlib.pyplot as plt
import pickle


def plot_roc(fpr, tpr, roc_auc, model_name):
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('{}.png'.format(model_name))


if __name__ == '__main__':
    model_names = [
        '2018-05-04--22-57-49',
        '2018-05-04--23-03-46',
        '2018-05-07--17-22-10',
        '2018-05-08--23-37-07',
        '2018-05-11--00-10-54',
    ]

    for model_name in model_names:
        with open('evaluate/roc-{}.rick'.format(model_name), 'rb') as f:
            fpr, tpr, roc_auc = pickle.load(f)

        plot_roc(fpr, tpr, roc_auc, model_name)
