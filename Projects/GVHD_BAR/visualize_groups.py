import seaborn as sns
import matplotlib.pyplot as plt

def plot_bars(data, groups, x, rwidth=0.9, fig_subtitle='Train', ax=None, axis_title=None):
    colors = ['red', 'blue', 'green', 'yellow'] # todo: need to add a generic function of n colors
    x_data = data[x].to_frame()
    x_data['class'] = groups
    data_grouped = x_data.groupby('class')

    fig=None
    if ax is None:
        fig, ax = plt.subplots(1, len(data_grouped))
        fig.suptitle(fig_subtitle, fontsize=16)

    for current_ax, (idx, (subject_id, subject_data)) in zip(ax.flatten(), enumerate(data_grouped)):
        current_ax.hist(subject_data[x].values.astype(float), color=colors[idx], label=subject_id, alpha=0.5, rwidth=rwidth)
        current_ax.set_xlabel(x)
        current_ax.set_ylabel('Count')
        current_ax.set_title(f'{axis_title} Class: {subject_id}')

    return fig, ax