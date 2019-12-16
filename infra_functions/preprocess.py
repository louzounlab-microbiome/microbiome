import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns


def preprocess_data(data, preform_z_scoring=True, preform_log=True, preform_taxnomy_group=True, taxnomy_level=6,
                    eps_for_zeros=0.1, visualize_data=True, taxonomy_col='taxonomy', std_to_delete=0):
    as_data_frame = pd.DataFrame(data.T).apply(pd.to_numeric, errors='ignore').copy()
    if as_data_frame.columns[0].startswith("Unnamed"):
        as_data_frame = as_data_frame.drop(as_data_frame.columns[0], axis=1)
    if visualize_data:
        folder = "preprocess_plots"
        if not os.path.exists(folder):
            os.mkdir(folder)
        plt.figure('Preprocess')
        data_frame_for_vis = as_data_frame.copy()
        try:
            data_frame_for_vis = data_frame_for_vis.drop(taxonomy_col, axis=1)
        except:
            pass
        data_frame_flatten = data_frame_for_vis.values.flatten()
        indexes_of_non_zeros = data_frame_flatten != 0
        visualize_preproccess(data_frame_for_vis, indexes_of_non_zeros, 'Before Taxonomy group', [321, 322])

    if preform_taxnomy_group:
        as_data_frame = as_data_frame.drop(as_data_frame.std()[as_data_frame.std() < std_to_delete].index.values, axis=1)
        # union taxonomy level by group level
        taxonomy_reduced = as_data_frame[taxonomy_col].map(lambda x: x.split(';'))
        taxonomy_reduced = taxonomy_reduced.map(lambda x: ';'.join(x[:taxnomy_level]))
        as_data_frame[taxonomy_col] = taxonomy_reduced
        as_data_frame = as_data_frame.groupby(as_data_frame[taxonomy_col]).mean()
        as_data_frame = as_data_frame.T
    else:
        try:
            as_data_frame = as_data_frame.drop(taxonomy_col, axis=1).T
        except:
            pass

    if visualize_data:
        data_frame_flatten = as_data_frame.values.flatten()
        indexes_of_non_zeros = data_frame_flatten != 0
        visualize_preproccess(as_data_frame, indexes_of_non_zeros, 'After-Taxonomy - Before', [323, 324])
        samples_density = as_data_frame.apply(np.sum, axis=1)
        plt.figure('Density of samples')
        samples_density.hist(bins=100)
        plt.title(f'Density of samples')
        plt.savefig(os.path.join(folder, "density_of_samples.svg"), bbox_inches='tight', format='svg')


    if preform_log:
        as_data_frame += eps_for_zeros
        as_data_frame = np.log10(as_data_frame)

    if visualize_data:
        # plot histogrm of variance
        samples_variance = as_data_frame.apply(np.var, axis=1)
        plt.figure('Variance of samples')
        samples_variance.hist(bins=100)
        plt.title(
            f'Histogram of samples variance before z-scoring\nmean={samples_variance.values.mean()},'
            f' std={samples_variance.values.std()}')
        plt.savefig(os.path.join(folder, "samples_variance.svg"), bbox_inches='tight', format='svg')


    if preform_z_scoring:
        as_data_frame[:] = preprocessing.scale(as_data_frame, axis=1)

    if visualize_data:
        plt.figure('Preprocess')
        visualize_preproccess(as_data_frame, indexes_of_non_zeros, 'After-Taxonomy - After', [325, 326])
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        plt.savefig(os.path.join(folder, "preprocess.svg"), bbox_inches='tight', format='svg')

    if visualize_data:
        plt.figure('standart heatmap')
        sns.heatmap(as_data_frame, cmap="Blues")
        plt.title('Heatmap after standartization and taxonomy group level ' + str(taxnomy_level))
        plt.savefig(os.path.join(folder, "standart_heatmap.svg"), bbox_inches='tight', format='svg')
        corr_method = 'pearson'
        # if smaples on both axis needed, specify the vmin, vmax and mathod
        plt.figure('correlation heatmap patient')
        sns.heatmap(as_data_frame.T.corr(method=corr_method), cmap='RdBu', vmin=-1, vmax=1)
        plt.title(corr_method + ' correlation patient with taxonomy level ' + str(taxnomy_level))
        # plt.savefig(os.path.join(folder, "correlation_heatmap_patient.svg"), bbox_inches='tight', format='svg')
        plt.savefig(os.path.join(folder, "correlation_heatmap_patient.png"))

        plt.figure('correlation heatmap bacteria')
        sns.heatmap(as_data_frame.corr(method=corr_method), cmap='RdBu', vmin=-1, vmax=1)
        plt.title(corr_method + ' correlation bacteria with taxonomy level ' + str(taxnomy_level))
        # plt.savefig(os.path.join(folder, "correlation_heatmap_bacteria.svg"), bbox_inches='tight', format='svg')
        plt.savefig(os.path.join(folder, "correlation_heatmap_bacteria.png"))

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
