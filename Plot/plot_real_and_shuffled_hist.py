import matplotlib.pyplot as plt


def plot_real_and_shuffled_hist(ax: plt.Axes, real_values, shuffled_values, real_bins: int = 10,
                                shuffled_bins: int = 10, real_color: str = 'g', shuffled_color: str = 'b',
                                real_label='Real values', shuffled_label='Shuffled values'):
    # real histogram
    ax.hist(real_values, bins=real_bins, color=real_color, label=real_label)
    ax.hist(shuffled_values, bins=shuffled_bins, color=shuffled_color, label=shuffled_label)
    ax.legend()
    return ax