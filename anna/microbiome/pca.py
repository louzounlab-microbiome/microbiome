import numpy as np
import pandas as pd
import matplotlib as plt
import plotly
import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls
from sklearn.decomposition import PCA

def visualize_pca(new_matrix):
    cov_mat = np.cov(new_matrix.T)
    eig_vals = np.linalg.eigvals(cov_mat)
    eig_vals = eig_vals.real
    tot = sum(eig_vals)
    var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    plotly.tools.set_credentials_file(username='anna.bel_', api_key='CYSlYoBg7X1NdK3b1TyU')
    trace1 = Bar(
        x=['PC %s' % i for i in range(1, 50)],
        y=var_exp,
        showlegend=False)
    trace2 = Scatter(
        x=['PC %s' % i for i in range(1, 50)],
        y=cum_var_exp,
        name='cumulative explained variance')
    data = Data([trace1, trace2])

    layout = Layout(
        yaxis=YAxis(title='Explained variance in percent'),
        title='Explained variance by different principal components')

    fig = Figure(data=data, layout=layout)
    py.plot(fig)

def apply_pca(data, n_components=15, print_data=True):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    data_components = pca.fit_transform(data)
    #i = np.identity(data.shape[1])
    #coef = pca.transform(i)
    if print_data:
        print("Explained variance per component: \n" +
          '\n'.join(['Component ' + str(i) + ': ' +
                     str(component) for (i, component) in enumerate(pca.explained_variance_ratio_)]))
        print("Total explained variance: " + str(pca.explained_variance_ratio_.sum()))
    return pd.DataFrame(data_components).set_index(data.index), pca.components_

def apply_pca_with_diff_data(data1, data2, n_components=15):
    pca = PCA(n_components=n_components)
    pca.fit(data1)
    data_components = pca.transform(data2)
    #i = np.identity(data.shape[1])
    #coef = pca.transform(i)
    print("Explained variance per component: \n" +
          '\n'.join(['Component ' + str(i) + ': ' +
                     str(component) for (i, component) in enumerate(pca.explained_variance_ratio_)]))
    print("Total explained variance: " + str(pca.explained_variance_ratio_.sum()))
    return pd.DataFrame(data_components).set_index(data2.index), pca.components_