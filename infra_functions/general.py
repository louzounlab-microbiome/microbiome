from sklearn.decomposition import PCA
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import numpy as np

# from sc

def apply_pca(data, n_components=15, visualize=False):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    data_components = pca.fit_transform(data)

    print("Explained variance per component: \n" +
          '\n'.join(['Component ' + str(i) + ': ' +
                     str(component) + ', Accumalative variance: ' + str(accu_var) for accu_var, (i, component) in zip(pca.explained_variance_ratio_.cumsum(), enumerate(pca.explained_variance_ratio_))]))
    print("Total explained variance: " + str(pca.explained_variance_ratio_.sum()))
    if visualize:
        plt.figure()
        plt.plot(pca.explained_variance_ratio_.cumsum())
        plt.bar(np.arange(0,n_components), height=pca.explained_variance_ratio_)
        plt.title('PCA - Explained variance - total:'+str(pca.explained_variance_ratio_.sum()))
        plt.show()
    return pd.DataFrame(data_components).set_index(data.index), pca

def use_spearmanr(x,y, axis=None):
    if x is None or y is None:
        print('Got None')
    rho, pvalue = spearmanr(x, y ,axis)
    return {'rho': rho, 'pvalue': pvalue}

def use_pearsonr(x,y):
    rho, pvalue = pearsonr(x, y)
    return {'rho': rho, 'pvalue': pvalue}
