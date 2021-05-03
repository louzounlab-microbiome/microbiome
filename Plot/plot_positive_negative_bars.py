import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_positive_negative_bars(ax: plt.Axes, values: pd.Series, positive_color='green', negative_color='red'):
    sorted_values = values.sort_values()
    y_position = np.arange(len(sorted_values))
    positive_values = sorted_values.apply(lambda x: x if x >= 0 else 0)
    ax.barh(y_position, positive_values, color=positive_color)
    negative_values = sorted_values.apply(lambda x: x if x < 0 else 0)
    ax.barh(y_position, negative_values, color=negative_color)
    ax.set_yticks(y_position)
    ax.set_yticklabels(sorted_values.index)
    return ax
