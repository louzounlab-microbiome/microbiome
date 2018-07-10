import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing


def preprocess_data(data, preform_z_scoring=True, preform_log=True, eps_for_zeros=0.01, visualize_data=True):
    as_data_frame = pd.DataFrame(data).astype(float)

    if visualize_data:
        data_frame_flatten = as_data_frame.values.flatten()
        indexes_of_non_zeros = data_frame_flatten != 0
        visualize_preproccess(as_data_frame, indexes_of_non_zeros, 'Before', [221, 223])

    if preform_log:
        as_data_frame += eps_for_zeros
        as_data_frame = np.log10(as_data_frame)

    # if preform_z_scoring:
    #     as_data_frame[:] = preprocessing.scale(as_data_frame)

    if visualize_data:
        visualize_preproccess(as_data_frame, indexes_of_non_zeros, 'After', [222, 224])
        plt.subplots_adjust(hspace=0.5, wspace =0.5)
        plt.show()
    return as_data_frame


def visualize_preproccess(as_data_frame, indexes_of_non_zeros, name, subplot_idx):
    plt.subplot(subplot_idx[0])
    data_frame_flatten = as_data_frame.values.flatten()
    plot_preprocess_stage(data_frame_flatten, name)
    result = data_frame_flatten[indexes_of_non_zeros]
    plt.subplot(subplot_idx[1])
    plot_preprocess_stage(result, name + ' without zeros')


def plot_preprocess_stage(result, name):
    plt.hist(result, 1000, facecolor='green', alpha=0.75)
    plt.title('Distribution ' + name + ' preprocess')
    plt.xlabel('BINS')
    plt.ylabel('Count')
