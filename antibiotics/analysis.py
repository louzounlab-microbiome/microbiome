import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import preprocessing
from infra_functions.filtering import remove_outliers
from infra_functions.statistical_tests import hottelings_t2_test

def load_data(mapping_csv_file, otu_csv_file, use_taxonomy = False):
    mapping_file = pd.read_csv(mapping_csv_file)
    mapping_file = mapping_file.set_index('#SampleID').sort_index()
    mapping_file['treatment_class'] = mapping_file.apply(lambda row: 0 if row['Treat'] == 'Control' else 1, axis=1)
    otu_file = pd.read_csv(otu_csv_file).set_index('#OTU ID')
    if use_taxonomy:
        otu_file['taxonomy'] = otu_file.apply(lambda row: 1 / len(row['taxonomy'].split(';')), axis=1)
    otu_file = otu_file.transpose().sort_index()
    if use_taxonomy:
        otu_file['taxonomy'] = otu_file.apply(lambda row: np.dot(row, otu_file[-1:].T.values[:, 0]), axis=1)
    otu_file = otu_file[:-1]
    return mapping_file, otu_file

def get_data_by_sample_month(data, months):
    a=data.loc[data['SampleMonth'].apply(lambda x: x in months)]
    return a

def apply_pca(data, n_components=8):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    data_components = pca.fit_transform(data)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_ratio_.sum())
    return pca, pd.DataFrame(data_components)


def visualize_pca(data, v_i, v_j, draw_outliers):
    inliers, outliers = remove_outliers(data, [v_i, v_j])
    class_0 = inliers.join(data['class']).loc[(data['class'] == 0)]
    class_1 = inliers.join(data['class']).loc[(data['class'] == 1)]

    fig, ax = plt.subplots()
    ax.scatter(class_0[v_i], class_0[v_j], c='r', label='No Treatment')
    ax.scatter(class_1[v_i], class_1[v_j], c='b', label='Treatment')
    if draw_outliers:
        ax.scatter(outliers[v_i], outliers[v_j], c='black', label='Outliers')
    plt.title(str(v_i) + ' vs ' + str(v_j))
    plt.xlabel("PCA component " + str(v_i))
    plt.ylabel("PCA component " + str(v_j))
    ax.legend()
    plt.show()


def visualize_all_data(data, explained_var, min_var_explained=0.05, draw_outliers=True):
    variance_to_plot = explained_var[explained_var > min_var_explained]
    for i in range(len(variance_to_plot)):
        for j in range(i+1, len(variance_to_plot)):
            visualize_pca(data, i, j, draw_outliers)
            plt.show()


def preprocessData(data, sampleMonth, eps=0.01, visualize_data=True):
    as_data_frame = pd.DataFrame(data).astype(float)
    indexes_of_non_zeros = data.flatten() != 0

    # data_before_preprocess = data.sum(axis=0)
    if visualize_data:
        visualize_preproccess(data, 'Before')
        result = data.flatten()[indexes_of_non_zeros]
        visualize_preproccess(result, 'Before without zeros')

    as_data_frame += eps
    as_data_frame = np.log(as_data_frame)
    as_data_frame['sampleMonth'] = sampleMonth
    mean_by_sample = as_data_frame.groupby(['sampleMonth']).mean().mean(axis=1)
    for sample in mean_by_sample.index:
        as_data_frame.loc[as_data_frame['sampleMonth'] == sample] = as_data_frame.loc[as_data_frame['sampleMonth'] == sample] - mean_by_sample[sample]
    as_data_frame.drop(['sampleMonth'], axis=1, inplace=True)
    data = as_data_frame.values

    if visualize_data:
        visualize_preproccess(data, 'After')
        result = data.flatten()[indexes_of_non_zeros]
        visualize_preproccess(result, 'After without zeros')
    return data


def visualize_preproccess(data, name):
    fig, ax = plt.subplots()
    plt.hist(data.flatten(), 1000, facecolor='green', alpha=0.75)
    plt.title('Distribution ' + name + ' preprocess')
    plt.xlabel('BINS')
    plt.ylabel('Count')
    plt.show()



if __name__ == "__main__":
    mapping_file, otu_file = load_data('mapping_file_abx.csv', 'otu_abx_new_new.csv')
    merged_data = otu_file.join(mapping_file)

    months_to_test = [1,6,12,24]

    merged_data = get_data_by_sample_month(merged_data, months_to_test)
    headers = merged_data.columns.values.tolist()
    original_names = merged_data[:].index.astype(str)
    classes = merged_data['treatment_class'].values[:]
    sampleMonth = merged_data['SampleMonth'].values[:]

    # drop last row for PCA
    data_for_pca = merged_data.ix[:, :otu_file.shape[1]]
    data_for_pca = data_for_pca[:].values.astype(int)


    # preprocess the data
    data_for_pca = preprocessData(data_for_pca, sampleMonth, visualize_data=False)
    merged_data_after_preproccess = merged_data.copy()
    merged_data_after_preproccess.ix[:, :otu_file.shape[1]] = data_for_pca

    # PCA
    pca_obj, pca_data = apply_pca(data_for_pca)


    pca_data['class'] = classes
    pca_data['sampleMonth'] = sampleMonth
    pca_data = pca_data.set_index(original_names)
    pca_data = pca_data.loc[pca_data['sampleMonth'] == 24].drop(['sampleMonth'], axis=1)
    # pca_data = pca_data.sort_values(by=['class'])

    # Visualize the data
    # visualize_pca(pca_data, 0, 1)
    # visualize_all_data(pca_data, pca_obj.explained_variance_ratio_, min_var_explained=0.04, draw_outliers = False)

    visualize_all_data(pca_data, pca_obj.explained_variance_ratio_, min_var_explained=0.0, draw_outliers = True)

    inliers, outliers = remove_outliers(pca_data.drop(['class'], axis=1), filter_sample_vectors=None, min_samples=2)

    class_0 = inliers.join(pca_data['class']).loc[(pca_data['class'] == 0)]
    #   class_1 = inliers.join(pca_data['class']).loc[(pca_data['class'] == 1)]

    #  class_0 = class_0.loc[:, class_0.columns != 'class']
    # class_1 = class_1.loc[:, class_1.columns != 'class']


    # hottelings_t2_test(class_0.drop(['class'], axis=1), class_1.drop(['class'], axis=1))

    # class_0 = np.random.normal(0, 1, (1000,8))
    # class_1 = np.random.normal(0, 1, (1000,8))
    #hottelings_t2_test(class_0, class_1)