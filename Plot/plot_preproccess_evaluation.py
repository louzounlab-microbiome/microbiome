import os
import numpy as np
import matplotlib.pyplot as plt


def plot_task_comparision(task_results, results_path, title, pca_options, tax_options=[None]):
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    x_ticks_names = []
    train_auc_values = []
    test_auc_values = []
    for tax in tax_options:
        for pca_n in pca_options:
            if tax:
                x_ticks_names.append("tax=" + str(tax) + " pca=" + str(pca_n))
                train, test = task_results[(tax, pca_n)]
            else:
                x_ticks_names.append("pca=" + str(pca_n))
                train, test = task_results[pca_n]
            train_auc_values.append(train)
            test_auc_values.append(test)

    N = len(x_ticks_names)
    ind = np.arange(N)  # the x locations for the groups
    width = 10 / N  # the width of the bars: can also be len(x) sequence
    plt.ylim((0, 1))


    p2 = plt.bar(ind, train_auc_values, width)
    p1 = plt.bar(ind, test_auc_values, width)

    plt.ylabel('AUC')
    plt.title(title.replace("_", " ").capitalize() + '\nAUC by taxonomy level and pca components')
    plt.xticks(ind, x_ticks_names, rotation='vertical')
    plt.legend((p2[0], p1[0]), ('Train', 'Test'))

    if not os.path.exists(results_path):
        os.mkdir(results_path)
    plt.savefig(os.path.join(results_path, title.replace("_", " ").replace("\n", "_") + ".svg"),
                bbox_inches='tight', format='svg')
    plt.show()

