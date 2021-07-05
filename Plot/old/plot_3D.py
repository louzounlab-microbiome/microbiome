import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
from scipy import stats
import pandas as pd


def plot_data_3d(X_3d: pd.DataFrame, color: pd.Series, color_dict=None, labels_dict: dict = None, size=10):
    fig = plt.figure()
    ax = Axes3D(fig)
    groups = X_3d.groupby(color)
    for name, group in groups:
        if color_dict is None and labels_dict is None:
            ax.scatter(group.iloc[:, 0], group.iloc[:, 1], group.iloc[:, 2])
        elif color_dict is None:
            ax.scatter(group.iloc[:, 0], group.iloc[:, 1], group.iloc[:, 2], label=labels_dict[name])
        elif labels_dict is None:
            ax.scatter(group.iloc[:, 0], group.iloc[:, 1], group.iloc[:, 2], c=color_dict[name])
        else:
            ax.scatter(group.iloc[:, 0], group.iloc[:, 1], group.iloc[:, 2],
                       c=color_dict[name], label=labels_dict[name])
    ax.set_xlabel(X_3d.columns[0], size=size)
    ax.set_ylabel(X_3d.columns[1], size=size)
    ax.set_zlabel(X_3d.columns[2], size=size)
    fig.legend()

    return fig, ax


def plot_data_2d(X_all, y, data_name, save=False, folder=None):
    plt.close()
    X_2D = np.array([x[:2] for x in X_all])

    x_min, x_max = X_2D[:, 0].min() - .5, X_2D[:, 0].max() + .5
    y_min, y_max = X_2D[:, 1].min() - .5, X_2D[:, 1].max() + .5

    plt.figure(2, figsize=(8, 6))
    plt.clf()

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    # y_ = ['c' if item == 0 else 'm' for item in y]
    y_ = []
    for item in y:
        if item == 0:
            y_.append('c')
        else:
            y_.append('m')

    # To getter a better understanding of interaction of the dimensions
    # plot the first three PCA dimensions
    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)  # Axes3D(fig, elev=-150, azim=110)
    plt.xticks([], [])
    plt.yticks([], [])
    X_reduced = PCA(n_components=2).fit_transform(X_all)
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_,
               edgecolor='k')  # cmap=plt.cm.Set3,  , s=25)
    ax.set_title(data_name.replace("_", " ") + "\nFirst two PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.set_ylabel("2nd eigenvector")

    if save:
        if folder:
            plt.savefig(os.path.join(folder, data_name + "_2D_pca_display.svg"), bbox_inches='tight', format='svg')
        else:
            plt.savefig(data_name + "_2D_pca_display.svg", bbox_inches='tight', format='svg')
    else:
        plt.show()
    plt.close()


def PCA_t_test(group_1, group_2, title="t_test", save=False, folder=None):
    # loop each dimension
    result = []
    statistic = []
    pvalue = []
    for dim in range(len(group_1[0])):
        a = [g[dim] for g in group_1]
        b = [g[dim] for g in group_2]
        result.append(stats.ttest_ind(a, b))

    if save:
        if folder:
            if not os.path.exists(folder):
                os.makedirs(folder)
            with open(os.path.join(folder, title.replace(" ", "_") + ".txt"), "w") as file:
                for i, res in enumerate(result):
                    file.write("dim " + str(i + 1) + "\n" + "statistic=" + str(round(res[0], 5)) + " pvalue=" + str(
                        round(res[1], 5)) + "\n")
        else:
            with open(title.replace(" ", "_") + ".txt", "w") as file:
                for i, res in enumerate(result):
                    file.write("dim " + str(i + 1) + "\n" + "statistic=" + str(round(res[0], 5)) + " pvalue=" + str(
                        round(res[1], 5)) + "\n")

    return result


if __name__ == "__main__":
    PCA_t_test()
