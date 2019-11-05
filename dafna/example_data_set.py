from sys import stdout
from anna.microbiome.distance_learning_func import distance_learning
from dafna.abstract_data_set import AbstractDataLoader
from infra_functions.load_merge_otu_mf import OtuMfHandler
from infra_functions.preprocess import preprocess_data
import os
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
import numpy as np
from infra_functions.general import apply_pca

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

n_components = 20


class ExampleDataLoader(AbstractDataLoader):
    """
    Fill _read_file function according to specific data set in order to achieve mutual behavior
    create all of the following members:
        self._preproccessed_data = preproccessed_data
        self._pca_obj = pca_obj
        self._index_to_id_map = index_to_id_map
        self._id_to_features_map = id_to_features_map
        self._ids_list = ids_list
        make your own -> self._id_to_tag_map = _id_to_tag_map
        self._weight_map = classes_ratio
        self._feature_list = feature_list
    """
    def __init__(self, title, bactria_as_feature_file, samples_data_file, taxnomy_level, allow_printing, perform_anna_preprocess):
        super().__init__(title, bactria_as_feature_file, samples_data_file, taxnomy_level, allow_printing, perform_anna_preprocess)

    def _read_file(self, title, bactria_as_feature_file, samples_data_file, allow_printing, perform_anna_preprocess):
        features = pd.read_csv(bactria_as_feature_file, header=1)
        cols = list(features.columns)
        # remove non-numeric values
        cols.remove('Feature ID')
        cols.remove('Taxonomy')

        OtuMf = OtuMfHandler(os.path.join(SCRIPT_DIR, bactria_as_feature_file),
                             os.path.join(SCRIPT_DIR, samples_data_file),
                             from_QIIME=True, id_col='Feature ID', taxonomy_col='Taxonomy')

        if perform_anna_preprocess:
            preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxonomy_col='Taxonomy',
                                                 taxnomy_level=6)
            mapping_file = OtuMf.mapping_file['XXXXX']
            mapping_disease = {'a': 0,
                               'b': 1,  # 'Cashew' + 'Hazelnut' + 'Walnut'
                               'c': 2,
                               'd': 3}
            mapping_file = mapping_file.map(mapping_disease)
            preproccessed_data, mapping_file = distance_learning(perform_distance=True, level=self._taxnomy_level,
                                                                 preproccessed_data=preproccessed_data,
                                                                 mapping_file=mapping_file)
            self._preproccessed_data = preproccessed_data
        else:
            preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxnomy_level=self._taxnomy_level,
                                                 taxonomy_col='Taxonomy',
                                                 preform_taxnomy_group=True)

            self._preproccessed_data = preproccessed_data
            # drow_data(preproccessed_data)
            # otu_after_pca_wo_taxonomy, _, _ = apply_pca(data_after_log_zcore, n_components=40, visualize=False)

        otu_after_pca_wo_taxonomy, pca_obj, _ = apply_pca(preproccessed_data, n_components=n_components,
                                                          visualize=False)
        self._pca_obj = pca_obj

        index_to_id_map = {}
        id_to_features_map = {}
        for i, row in enumerate(otu_after_pca_wo_taxonomy.values):
            id_to_features_map[otu_after_pca_wo_taxonomy.index[i]] = row
            index_to_id_map[i] = otu_after_pca_wo_taxonomy.index[i]

        self._index_to_id_map = index_to_id_map
        self._id_to_features_map = id_to_features_map
        ids_list = otu_after_pca_wo_taxonomy.index.tolist()
        self._ids_list = ids_list

        XXXXX_column = 'SuccessDescription'
        id_to_tag_map = {}
        for sample in ids_list:
            t = OtuMf.mapping_file.loc[sample, XXXXX_column]
            id_to_tag_map[sample] = t
            if t == 'A1':
                id_to_tag_map[sample] = 1
            else:
                id_to_tag_map[sample] = 0
        self._id_to_tag_map = id_to_tag_map

        # -------------------------------------------- weights !--------------------------------------------
        # calculate weights
        y = list(id_to_tag_map.values())
        classes_sum = [np.sum(np.array(y) == unique_class) for unique_class in
                       np.unique(np.array(y))]
        classes_ratio = [1 - (a / sum(classes_sum)) for a in classes_sum]
        weights = [classes_ratio[a] for a in np.array(y)]
        self._weight_map = classes_ratio

        # return the list of features and the list of ids in the same order
        feature_list = [id_to_features_map[id] for id in ids_list]
        self._feature_list = feature_list

    def get_confusin_matrix_names(self):
        names = ['A', 'B']
        return names

if __name__ == "__main__":
    # need to run this file from the folder of the data!!!!
    # data_set_folder = "allergy"
    # cwd = os.getcwd()
    # last_folder = cwd.split("/")[-1]
    # cwd = cwd[:len(cwd) - len(last_folder)] + data_set_folder
    # os.chdir(cwd)

    task = 'success task'
    bactria_as_feature_file = 'feature-table_Allergy_cleaned_taxa_290119_updated_in_140219.csv'
    samples_data_file = 'mf_merge_ok84_ok93_ok66_69_merged_by_RestoredSampleCode_as_ID_290119.csv'
    tax = 6

    allergy_dataset = ExampleDataLoader(title=task, bactria_as_feature_file=bactria_as_feature_file,
                                         samples_data_file=samples_data_file,  taxnomy_level=tax,
                                         allow_printing=True, perform_anna_preprocess=False)
    print(allergy_dataset)
