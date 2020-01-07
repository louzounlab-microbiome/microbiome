# created by Yoel Jasner
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
from anna.microbiome.distance_learning_func import distance_learning
from infra_functions.general import apply_pca


def preprocess_data(data, dict_params, map_file, visualize_data=False):
    taxnomy_level = int(dict_params['taxonomy_level'])
    preform_taxnomy_group = dict_params['taxnomy_group']
    eps_for_zeros = float(dict_params['epsilon'])
    preform_norm = dict_params['normalization']
    preform_z_scoring = dict_params['z_scoring']
    relative_z = dict_params['norm_after_rel']
    var_th_delete = float(dict_params['std_to_delete'])
    pca = int(dict_params['pca'])

    taxonomy_col = 'taxonomy'

    as_data_frame = pd.DataFrame(data.T).apply(pd.to_numeric, errors='ignore').copy()

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

    if preform_taxnomy_group != '':
        print('Perform taxonomy grouping...')
        # union taxonomy level by group level
        # spliting the taxonomy level column
        taxonomy_reduced = as_data_frame[taxonomy_col].map(lambda x: x.split(';'))
        # insert new taxonomy level
        taxonomy_reduced = taxonomy_reduced.map(lambda x: ';'.join(x[:taxnomy_level]))
        as_data_frame[taxonomy_col] = taxonomy_reduced
        # group by mean
        if preform_taxnomy_group == 'mean':
            print('mean')
            as_data_frame = as_data_frame.groupby(as_data_frame[taxonomy_col]).mean()
        # group by sum
        elif preform_taxnomy_group == 'sum':
            print('sum')
            as_data_frame = as_data_frame.groupby(as_data_frame[taxonomy_col]).sum()
        # group by anna PCA
        elif preform_taxnomy_group == 'pca_anna':
            print('PCA')
            as_data_frame, map_file = distance_learning(perform_distance=True, level=taxnomy_level,
                                                        preprocessed_data=as_data_frame, mapping_file=map_file)
        as_data_frame = as_data_frame.T
        # here the samples are columns
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

    if preform_norm == 'log':
        print('Perform log normalization...')
        as_data_frame = log_normalization(as_data_frame, eps_for_zeros)

        # delete column with var give
        as_data_frame = drop_low_var(as_data_frame.T, var_th_delete)

        if visualize_data:
            # plot histogrm of variance
            samples_variance = as_data_frame.apply(np.var, axis=1)
            plt.figure('Variance of samples')
            samples_variance.hist(bins=100)
            plt.title(
                f'Histogram of samples variance before z-scoring\nmean={samples_variance.values.mean()},'
                f' std={samples_variance.values.std()}')
            plt.savefig(os.path.join(folder, "samples_variance.svg"), bbox_inches='tight', format='svg')

        if preform_z_scoring != 'No':
            as_data_frame = z_score(as_data_frame, preform_z_scoring)
    elif preform_norm == 'relative':
        print('Perform relative normalization...')
        as_data_frame = row_normalization(as_data_frame)
        if relative_z == "z_after_relative":
            as_data_frame = z_score(as_data_frame, 'col')

    if visualize_data:
        data_frame_flatten = as_data_frame.values.flatten()
        indexes_of_non_zeros = data_frame_flatten != 0
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
        #plt.show()
    as_data_frame_b_pca = as_data_frame.copy()
    bacteria = as_data_frame.columns

    if pca > 0:
        print('perform pca...')
        as_data_frame, pca_obj, pca_str = apply_pca(as_data_frame, n_components=pca)
    else:
        pca_obj = None

    return as_data_frame, as_data_frame_b_pca, pca_obj, bacteria


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


def row_normalization(as_data_frame):
    as_data_frame = as_data_frame.T
    for col in as_data_frame.columns:
        as_data_frame[col] /= as_data_frame[col].sum
    return as_data_frame.T


def drop_low_var(as_data_frame, threshold):
    drop_list = [col for col in as_data_frame.columns if col != 'taxonomy' and threshold > np.var(as_data_frame[col])]
    return as_data_frame.drop(columns=drop_list).T


def log_normalization(as_data_frame, eps_for_zeros):
    as_data_frame += eps_for_zeros
    as_data_frame = np.log10(as_data_frame)
    return as_data_frame


def z_score(as_data_frame, preform_z_scoring):
    if preform_z_scoring == 'col':
        print('perform z-core on columns...')
        # z-score on columns
        as_data_frame[:] = preprocessing.scale(as_data_frame, axis=1)
    elif preform_z_scoring == 'row':
        print('perform z-core on rows...')
        # z-score on rows
        as_data_frame[:] = preprocessing.scale(as_data_frame, axis=0)
    elif preform_z_scoring == 'both':
        print('perform z-core on rows and columns...')
        as_data_frame[:] = preprocessing.scale(as_data_frame, axis=0)
        as_data_frame[:] = preprocessing.scale(as_data_frame, axis=1)
    return as_data_frame