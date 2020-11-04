import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
from scipy import stats


def plot_data_3d(X_all, y, data_name, save=False, folder=None):
    X_3D = np.array([x[:3] for x in X_all])

    x_min, x_max = X_3D[:, 0].min() - .5, X_3D[:, 0].max() + .5
    y_min, y_max = X_3D[:, 1].min() - .5, X_3D[:, 1].max() + .5

    plt.figure(2, figsize=(8, 6))
    plt.clf()

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())

    # To getter a better understanding of interaction of the dimensions
    # plot the first three PCA dimensions
    fig = plt.figure(1, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    X_reduced = PCA(n_components=3).fit_transform(X_all)
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
               cmap=plt.cm.Set3, edgecolor='k') #, s=25)
    ax.set_title(data_name.replace("_", " ") + "\nFirst three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])
    if save:
        if folder:
            plt.savefig(os.path.join(folder, data_name + "_3D_pca_display.svg"), bbox_inches='tight', format='svg')
        else:
            plt.savefig(data_name + "_3D_pca_display.svg", bbox_inches='tight', format='svg')
    else:
        plt.show()
    plt.close()



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
                edgecolor='k')  #cmap=plt.cm.Set3,  , s=25)
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
                    file.write("dim " + str(i+1) + "\n" + "statistic=" + str(round(res[0], 5)) + " pvalue=" + str(round(res[1], 5)) + "\n")
        else:
            with open(title.replace(" ", "_") + ".txt", "w") as file:
                for i, res in enumerate(result):
                    file.write("dim " + str(i + 1) + "\n" + "statistic=" + str(round(res[0], 5)) + " pvalue=" + str(
                        round(res[1], 5)) + "\n")

    return result


if __name__ == "__main__":
    PCA_t_test()