import seaborn as sns; sns.set(color_codes=True)
def plot_clustergram(data, columns, method = 'single'):
    new_set3 = data.groupby(columns).mean().reset_index()
    new_set3['Group'] = ''
    for i in range(len(columns)):
        new_set3["Group"] += new_set3[columns[i]].map(str) + '_'
    new_set3['Group']= new_set3['Group'].str[:-1]
    new_set3 = new_set3.drop(columns, axis=1)
    new_set3 = new_set3.set_index(['Group'])
    new_set3 = new_set3.transpose()
    sns.set(font_scale=1)
    g = sns.clustermap(new_set3, cmap="RdYlGn", vmin=-1, vmax=1, col_cluster=False, yticklabels=False,
                       method=method)

    ax = g.ax_heatmap
    ax.set_ylabel('')
    ax.set_xlabel('')
