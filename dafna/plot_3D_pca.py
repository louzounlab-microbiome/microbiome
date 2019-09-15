import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import os


def plot_data_3d(X_all, X_3D, y, data_name, save=False, folder=None):
    x_min, x_max = X_3D[:, 0].min() - .5, X_3D[:, 0].max() + .5
    y_min, y_max = X_3D[:, 1].min() - .5, X_3D[:, 1].max() + .5

    plt.figure(2, figsize=(8, 6))
    plt.clf()

    # Plot the training points
    plt.scatter(X_3D[:, 0], X_3D[:, 1], c=y, cmap=plt.cm.Set3,
                edgecolor='k')
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')

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
               cmap=plt.cm.Set3, edgecolor='k', s=40)
    ax.set_title(data_name + "\nFirst three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])
    if save:
        if folder:
            plt.savefig(os.path.join(folder, data_name + "_3D_pca_display.png"))
        else:
            plt.savefig(data_name + "_3D_pca_display.png")
    else:
        plt.show()