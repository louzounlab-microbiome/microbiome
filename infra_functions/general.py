from sklearn.decomposition import PCA
import pandas as pd
from scipy.stats import spearmanr

def apply_pca(data, n_components=15):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    data_components = pca.fit_transform(data)
    print("Explained variance per component: \n" +
          '\n'.join(['Component ' + str(i) + ': ' +
                     str(component) for (i, component) in enumerate(pca.explained_variance_ratio_)]))
    print("Total explained variance: " + str(pca.explained_variance_ratio_.sum()))
    return pd.DataFrame(data_components).set_index(data.index), pca

def use_spearmanr(x,y, axis=None):
    rho, pvalue = spearmanr(x, y ,axis)
    return {'rho': rho, 'pvalue': pvalue}
