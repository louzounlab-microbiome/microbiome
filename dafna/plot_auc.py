from sklearn.metrics import roc_curve, auc
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools


def roc_auc(y_test, y_score, visualize=False, graph_title='ROC Curve', save=False, folder=None, fontsize=17):
    fpr, tpr, thresholds = roc_curve(np.array(y_test), np.array(y_score))
    roc_auc = auc(fpr, tpr)
    print('ROC AUC = {:7}'.format(roc_auc))
    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(fpr, tpr, color='red',
             label='ROC curve (area = %0.3f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title(f'{graph_title}\nroc={round(roc_auc, 3)}', fontsize=fontsize)
        plt.legend(loc="lower right")
        plt.xlabel('Specificity', fontsize=fontsize)
        plt.ylabel('Sensitivity', fontsize=fontsize)
        #plt.show()

        if save:
            if folder:
                plt.savefig(os.path.join(folder, graph_title.replace(" ", "_").replace("\n", "_") + "_" +
                                         str(round(roc_auc, 3)) + ".svg"), bbox_inches='tight', format='svg')
            else:
                plt.savefig(graph_title.replace(" ", "_").replace("\n", "_") + "_" + str(round(roc_auc, 3)) + ".svg",
                bbox_inches='tight', format='svg')
            plt.close()
    return fpr, tpr, thresholds, roc_auc


# y_test and y_score should be np array type
def multi_class_roc_auc(y_test, y_score, labels_names, graph_title='ROC Curve', save=False, folder=None, fontsize=17):
    n_classes = len(set(y_test))
    lw = 2
    # Compute macro-average ROC curve and ROC area and ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # change t from array of tag as class number to one hot matrix, where each row is a sample,
    # and the '1' index is the class - one hot.
    y_test_as_one_hot = [[0 for n in range(n_classes)] for y in y_test]
    for i, y in enumerate(y_test_as_one_hot):
        y_test_as_one_hot[i][y_test[i]] = 1
    y_test_as_one_hot = np.array(y_test_as_one_hot)

    if y_score.shape != y_test_as_one_hot.shape:
        y_score_as_one_hot = [[0 for n in range(n_classes)] for y in y_test]
        for i, y in enumerate(y_score_as_one_hot):
            y_score_as_one_hot[i][y_score[i]] = 1
        y_score_as_one_hot = np.array(y_score_as_one_hot)
    else:
        y_score_as_one_hot = y_score

    # calculate auc for each class.
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_as_one_hot[:, i], y_score_as_one_hot[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_as_one_hot.ravel(), y_score_as_one_hot.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=2)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=2)

    colors = itertools.cycle(['aqua', 'darkviolet', 'gold', 'red', 'greenyellow', 'darkorange', 'darkgrey', 'darkgreen'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(labels_names[i].replace("_", " "), roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Specificity', fontsize=fontsize)
    plt.ylabel('Sensitivity', fontsize=fontsize)
    plt.title(graph_title + "\nroc= " + str(round(roc_auc[i], 3)), fontsize=fontsize)
    plt.legend(loc="lower right")
    # plt.show()

    if save:
        if folder:
            plt.savefig(os.path.join(folder, graph_title.replace(" ", "_").replace("\n", "_") + ".svg"),
                        bbox_inches='tight', format='svg')
        else:
            plt.savefig(graph_title.replace(" ", "_").replace("\n", "_") + ".svg", bbox_inches='tight', format='svg')
        plt.close()

    return fpr, tpr, roc_auc

