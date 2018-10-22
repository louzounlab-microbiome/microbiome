import numpy as np
import pandas as pd
import matplotlib as plt
import plotly
import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls

def visualize_pca(new_matrix):
    cov_mat = np.cov(new_matrix.T)
    eig_vals = np.linalg.eigvals(cov_mat)
    eig_vals = eig_vals.real
    tot = sum(eig_vals)
    var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
    cum_var_exp = np.cumsum(var_exp)
    plotly.tools.set_credentials_file(username='anna.bel_', api_key=API_KEY)
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
