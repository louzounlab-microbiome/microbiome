import matplotlib.pyplot as plt
from LearningMethods.utilis import set_default_parameters

def plot_real_and_shuffled_hist(ax: plt.Axes, real_values, shuffled_values, real_hist_dict=None,
                                shuffled_hist_dict=None,title = 'Histogram of real and shuffled spearman correlation'):
    if real_hist_dict is None:
        real_hist_dict = {}
    if shuffled_hist_dict is None:
        shuffled_hist_dict = {}

    default_real_dict = {'bins': 10, 'color': 'g', 'label': 'Real values', 'density': True}
    default_shuffled_dict = {'bins': 10, 'color': 'b', 'label': 'Shuffled values', 'density': True}

    real_hist_dict = set_default_parameters(real_hist_dict, default_real_dict)
    shuffled_hist_dict = set_default_parameters(shuffled_hist_dict, default_shuffled_dict)

    ax.hist(real_values, **real_hist_dict)
    ax.hist(shuffled_values, **shuffled_hist_dict)
    ax.set_title(title)

    ax.legend()
    return ax


