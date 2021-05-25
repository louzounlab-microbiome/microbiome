# created by Yoel Jasner
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
from collections import Counter

from Preprocess.distance_learning_func import distance_learning
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA


def preprocess_data(data, dict_params: dict, map_file, visualize_data=False):
    taxnomy_level = int(dict_params['taxonomy_level'])
    preform_taxnomy_group = dict_params['taxnomy_group']
    eps_for_zeros = float(dict_params['epsilon'])
    preform_norm = dict_params['normalization']
    preform_z_scoring = dict_params['z_scoring']
    relative_z = dict_params['norm_after_rel']
    correlation_removal_threshold = dict_params.get('correlation_threshold', None)
    rare_bacteria_threshold = dict_params.get('rare_bacteria_threshold', None)
    var_th_delete = float(dict_params['std_to_delete'])
    pca = dict_params['pca']

    taxonomy_col = 'taxonomy'

    as_data_frame = pd.DataFrame(data.T).apply(pd.to_numeric, errors='ignore').copy()  # data frame of OTUs

    # fill all taxonomy level with default values
    as_data_frame = fill_taxonomy(as_data_frame, tax_col=taxonomy_col)

    if visualize_data:  # prepare folder for visualization
        folder = "static"
        if not os.path.exists(folder):
            os.mkdir(folder)
        plt.figure('Preprocess')  # make preprocessing figure
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
        if preform_taxnomy_group == 'sub PCA':
            taxonomy_reduced = taxonomy_reduced.map(lambda x: ';'.join(x[:]))
        else:
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
        elif preform_taxnomy_group == 'sub PCA':
            print('PCA')
            as_data_frame = as_data_frame.groupby(as_data_frame[taxonomy_col]).mean()
            # as_data_frame = as_data_frame.T
            # as_data_frame.columns = as_data_frame.iloc[[-1]].values[0]
            # as_data_frame, _ = distance_learning(perform_distance=True, level=taxnomy_level, preproccessed_data=as_data_frame.iloc[:-1], mapping_file=map_file).T
            # as_data_frame_b_pca = as_data_frame

        as_data_frame = as_data_frame.T
        # here the samples are columns
    else:
        try:
            as_data_frame = as_data_frame.drop(taxonomy_col, axis=1).T
        except:
            pass

    # remove highly correlated bacteria
    if correlation_removal_threshold is not None:
        as_data_frame = dropHighCorr(as_data_frame, correlation_removal_threshold)

    if visualize_data:
        data_frame_flatten = as_data_frame.values.flatten()
        indexes_of_non_zeros = data_frame_flatten != 0
        visualize_preproccess(as_data_frame, indexes_of_non_zeros, 'After-Taxonomy - Before', [323, 324])
        samples_density = as_data_frame.apply(np.sum, axis=1)
        plt.figure('Density of samples')
        samples_density.hist(bins=100, facecolor='Blue')
        plt.title(f'Density of samples')
        plt.savefig(os.path.join(folder, "density_of_samples.svg"), bbox_inches='tight', format='svg')

    # drop bacterias with single values
    if rare_bacteria_threshold is not None:
        as_data_frame = drop_rare_bacteria(as_data_frame, rare_bacteria_threshold)

    if preform_norm == 'log':
        print('Perform log normalization...')
        as_data_frame = log_normalization(as_data_frame, eps_for_zeros)

        # delete column with var give
        # as_data_frame = drop_low_var(as_data_frame.T, var_th_delete)

        if visualize_data:
            # plot histogrm of variance
            samples_variance = as_data_frame.apply(np.var, axis=1)
            plt.figure('Variance of samples')
            samples_variance.hist(bins=100, facecolor='Blue')
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
        plt.figure('standard heatmap')
        sns.heatmap(as_data_frame, cmap="Blues", xticklabels=False, yticklabels=False)
        plt.title('Heatmap after standardization and taxonomy group level ' + str(taxnomy_level))
        plt.savefig(os.path.join(folder, "standard_heatmap.png"))
        corr_method = 'pearson'
        corr_name = 'Pearson'
        # if samples on both axis needed, specify the vmin, vmax and mathod
        plt.figure('correlation heatmap patient')
        sns.heatmap(as_data_frame.T.corr(method=corr_method), cmap='Blues', vmin=-1, vmax=1, xticklabels=False,
                    yticklabels=False)
        plt.title(corr_name + ' correlation patient with taxonomy level ' + str(taxnomy_level))
        # plt.savefig(os.path.join(folder, "correlation_heatmap_patient.svg"), bbox_inches='tight', format='svg')
        plt.savefig(os.path.join(folder, "correlation_heatmap_patient.png"))

        plt.figure('correlation heatmap bacteria')
        sns.heatmap(as_data_frame.corr(method=corr_method), cmap='Blues', vmin=-1, vmax=1, xticklabels=False,
                    yticklabels=False)
        plt.title(corr_name + ' correlation bacteria with taxonomy level ' + str(taxnomy_level))
        # plt.savefig(os.path.join(folder, "correlation_heatmap_bacteria.svg"), bbox_inches='tight', format='svg')
        plt.savefig(os.path.join(folder, "correlation_heatmap_bacteria.png"))
        # plt.show()
        plt.close()
        plt.clf()

    as_data_frame_b_pca = as_data_frame.copy()
    bacteria = as_data_frame.columns

    if preform_taxnomy_group == 'sub PCA':
        as_data_frame, _ = distance_learning(perform_distance=True, level=taxnomy_level,
                                             preproccessed_data=as_data_frame, mapping_file=map_file)
        as_data_frame_b_pca = as_data_frame

    if visualize_data:
        draw_component_rhos_calculation_figure(as_data_frame, map_file, save_folder=folder)

    if pca[0] != 0:
        print('perform ' + pca[1] + ' ...')
        as_data_frame, pca_obj, pca = apply_pca(as_data_frame, n_components=pca[0], dim_red_type=pca[1])
    else:
        pca_obj = None

    return as_data_frame, as_data_frame_b_pca, pca_obj, bacteria, pca


def visualize_preproccess(as_data_frame, indexes_of_non_zeros, name, subplot_idx):
    plt.subplot(subplot_idx[0])
    data_frame_flatten = as_data_frame.values.flatten()
    plot_preprocess_stage(data_frame_flatten, name)
    result = data_frame_flatten[indexes_of_non_zeros]
    plt.subplot(subplot_idx[1])
    plot_preprocess_stage(result, name + ' without zeros')


def plot_preprocess_stage(result, name, write_title=False, write_axis=True):
    plt.hist(result, 1000, facecolor='Blue', alpha=0.75)
    if write_title:
        plt.title('Distribution ' + name + ' preprocess')
    if write_axis:
        plt.xlabel('BINS')
        plt.ylabel('Count')


def row_normalization(as_data_frame):
    as_data_frame = as_data_frame.div(as_data_frame.sum(axis=1), axis=0).fillna(0)
    return as_data_frame


def drop_low_var(as_data_frame, threshold):
    drop_list = [col for col in as_data_frame.columns if col != 'taxonomy' and threshold > np.var(as_data_frame[col])]
    return as_data_frame.drop(columns=drop_list).T


def log_normalization(as_data_frame, eps_for_zeros):
    as_data_frame += eps_for_zeros
    as_data_frame = np.log10(as_data_frame)
    return as_data_frame


def z_score(as_data_frame, preform_z_scoring):
    if preform_z_scoring == 'row':
        print('perform z-score on samples...')
        # z-score on columns
        as_data_frame[:] = preprocessing.scale(as_data_frame, axis=1)
    elif preform_z_scoring == 'col':
        print('perform z-score on features...')
        # z-score on rows
        as_data_frame[:] = preprocessing.scale(as_data_frame, axis=0)
    elif preform_z_scoring == 'both':
        print('perform z-score on samples and features...')
        as_data_frame[:] = preprocessing.scale(as_data_frame, axis=1)
        as_data_frame[:] = preprocessing.scale(as_data_frame, axis=0)

    return as_data_frame


def drop_bacteria(as_data_frame):
    bacterias = as_data_frame.columns
    bacterias_to_dump = []
    for i, bact in enumerate(bacterias):
        f = as_data_frame[bact]
        num_of_different_values = set(f)
        if len(num_of_different_values) < 2:
            bacterias_to_dump.append(bact)
    if len(bacterias_to_dump) != 0:
        print("number of bacterias to dump before intersection: " + str(len(bacterias_to_dump)))
        print("percent of bacterias to dump before intersection: " + str(
            len(bacterias_to_dump) / len(bacterias) * 100) + "%")
    else:
        print("No bacteria with single value")
    return as_data_frame.drop(columns=bacterias_to_dump)


def dropHighCorr(data, threshold):
    corr = data.corr()
    df_not_correlated = ~(corr.mask(np.tril(np.ones([len(corr)] * 2, dtype=bool))).abs() > threshold).any()
    un_corr_idx = df_not_correlated.loc[df_not_correlated[df_not_correlated.index] == True].index
    df_out = data[un_corr_idx]
    number_of_bacteria_dropped = len(data.columns) - len(df_out.columns)
    print('{} bacteria were dropped due to high correlation with other columns'.format(number_of_bacteria_dropped))
    return df_out


def drop_rare_bacteria(as_data_frame, threshold):
    bact_to_num_of_non_zeros_values_map = {}
    bacteria = as_data_frame.columns
    num_of_samples = len(as_data_frame.index) - 1
    for bact in bacteria:
        values = as_data_frame[bact]
        count_map = Counter(values)
        zeros = 0
        if 0 in count_map.keys():
            zeros += count_map[0]
        if '0' in count_map.keys():
            zeros += count_map['0']

        bact_to_num_of_non_zeros_values_map[bact] = num_of_samples - zeros

    rare_bacteria = []
    for key, val in bact_to_num_of_non_zeros_values_map.items():
        if val < threshold:
            rare_bacteria.append(key)
    as_data_frame.drop(columns=rare_bacteria, inplace=True)
    print("{} bacteria with less then {} non-zero value: ".format(len(rare_bacteria), threshold))
    return as_data_frame


def apply_pca(data, n_components=15, dim_red_type='PCA', visualize=False):
    if n_components == -1:
        pca = PCA(n_components=min(len(data.index), len(data.columns)))
        pca.fit(data)
        data_components = pca.fit_transform(data)
        for accu_var, (i, component) in zip(pca.explained_variance_ratio_.cumsum(),
                                            enumerate(pca.explained_variance_ratio_)):
            if accu_var > 0.7:
                components = i + 1
                break
    else:
        components = n_components
    if dim_red_type == 'PCA':
        pca = PCA(n_components=components)
        pca.fit(data)
        data_components = pca.fit_transform(data)

        str_to_print = str("Explained variance per component: \n" +
                           '\n'.join(['Component ' + str(i) + ': ' +
                                      str(component) + ', Accumalative variance: ' + str(accu_var) for
                                      accu_var, (i, component) in zip(pca.explained_variance_ratio_.cumsum(),
                                                                      enumerate(pca.explained_variance_ratio_))]))

        str_to_print += str("\nTotal explained variance: " + str(pca.explained_variance_ratio_.sum()))

        print(str_to_print)
        if visualize:
            plt.figure()
            plt.plot(pca.explained_variance_ratio_.cumsum())
            plt.bar(np.arange(0, components), height=pca.explained_variance_ratio_)
            plt.title(
                f'PCA - Explained variance using {n_components} components: {pca.explained_variance_ratio_.sum()}')
            plt.xlabel('PCA #')
            plt.xticks(list(range(0, components)), list(range(1, components + 1)))

            plt.ylabel('Explained Variance')
            plt.show()
    else:
        pca = FastICA(n_components=components)
        data_components = pca.fit_transform(data)
    return pd.DataFrame(data_components).set_index(data.index), pca, components


def fill_taxonomy(as_data_frame, tax_col):
    df_tax = as_data_frame[tax_col].str.split(';', expand=True)
    df_tax[6] = df_tax[6].fillna('s__')
    df_tax[5] = df_tax[5].fillna('g__')
    df_tax[4] = df_tax[4].fillna('f__')
    df_tax[3] = df_tax[3].fillna('o__')
    df_tax[2] = df_tax[2].fillna('c__')
    df_tax[1] = df_tax[1].fillna('p__')
    df_tax[0] = df_tax[0].fillna('s__')

    as_data_frame[tax_col] = df_tax[0] + ';' + df_tax[1] + ';' + df_tax[2] + ';' + df_tax[3] + ';' + df_tax[4] + ';' + \
                             df_tax[5] + ';' + df_tax[6]

    return as_data_frame


def from_biom(biom_file_path, taxonomy_file_path, otu_dest_path, **kwargs):
    # Load the biom table and rename index.
    from biom import load_table
    otu_table = load_table(biom_file_path).to_dataframe(True)
    # Load the taxonomy file and extract the taxonomy column.
    taxonomy = pd.read_csv(taxonomy_file_path, index_col=0, sep=None, **kwargs).drop('Confidence', axis=1,
                                                                                     errors='ignore')
    otu_table = pd.merge(otu_table, taxonomy, right_index=True, left_index=True)
    otu_table.rename({'Taxon': 'taxonomy'}, inplace=True, axis=1)
    otu_table = otu_table.transpose()
    otu_table.index.name = 'ID'
    otu_table.to_csv(otu_dest_path)
