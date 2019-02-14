from sklearn.decomposition import PCA
import pandas as pd
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

# from sc

def apply_pca(data, n_components=15, visualize=False):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    data_components = pca.fit_transform(data)

    str_to_print = str("Explained variance per component: \n" +
          '\n'.join(['Component ' + str(i) + ': ' +
                     str(component) + ', Accumalative variance: ' + str(accu_var) for accu_var, (i, component) in zip(pca.explained_variance_ratio_.cumsum(), enumerate(pca.explained_variance_ratio_))]))

    str_to_print += str("\nTotal explained variance: " + str(pca.explained_variance_ratio_.sum()))

    print(str_to_print)
    if visualize:
        plt.figure()
        plt.plot(pca.explained_variance_ratio_.cumsum())
        plt.bar(np.arange(0,n_components), height=pca.explained_variance_ratio_)
        plt.title(f'PCA - Explained variance using {n_components} components: {pca.explained_variance_ratio_.sum()}')
        plt.xlabel('PCA #')
        plt.xticks(list(range(0,n_components)), list(range(1,n_components+1)))

        plt.ylabel('Explained Variance')
        plt.show()
    return pd.DataFrame(data_components).set_index(data.index), pca, str_to_print

def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


def draw_horizontal_bar_chart(data, names=None, title=None, ylabel=None, xlabel=None, use_pos_neg_colors=True, left_padding=0.4):
    fig, ax = plt.subplots()
    y_pos = np.arange(len(data))
    if use_pos_neg_colors:
        coeff_color = ['blue' if x else 'red' for x in data >= 0]
    else:
        coeff_color = ['blue' for x in data >= 0]
    ax.barh(y_pos, data, color=coeff_color)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    plt.title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    fig.subplots_adjust(left=left_padding)
    # set_size(5, 5, ax)
    plt.show()

def convert_pca_back_orig(pca_components, w, original_names=None, visualize=False, title='Bacteria Coeff', ylabel='Bacteria', xlabel='Coeff Value'):
    coeff = np.dot(w, pca_components)
    if original_names is None:
        object_to_return =  pd.DataFrame({'Coefficients': coeff})
    else:
        object_to_return = pd.DataFrame(
        {'Taxonome': original_names,
         'Coefficients': coeff
         })

    if visualize:
        draw_horizontal_bar_chart(coeff, original_names, title, ylabel, xlabel)
    return object_to_return

def use_spearmanr(x,y, axis=None):
    if x is None or y is None:
        print('Got None')
    rho, pvalue = spearmanr(x, y ,axis)
    return {'rho': rho, 'pvalue': pvalue}

def use_pearsonr(x,y):
    rho, pvalue = pearsonr(x, y)
    return {'rho': rho, 'pvalue': pvalue}


def roc_auc(y_test, y_score, verbose=False, visualize=False, graph_title='ROC Curve'):
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    print(f'ROC AUC = {roc_auc}')
    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(fpr, tpr)
        plt.title(f'{graph_title}\nroc={roc_auc}')
        plt.xlabel('fpr')
        plt.ylabel('tpr')
    return fpr, tpr, thresholds, roc_auc