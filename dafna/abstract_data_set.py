from sys import stdout
from anna.microbiome.distance_learning_func import distance_learning
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


class AbstractDataLoader:
    def __init__(self, title, bactria_as_feature_file, samples_data_file, taxnomy_level, allow_printing, perform_anna_preprocess):
        self._task = title
        self._taxnomy_level = taxnomy_level
        self._read_file(title, bactria_as_feature_file, samples_data_file, allow_printing, perform_anna_preprocess)

    @property
    def get_index_to_id_map(self):
        return self._index_to_id_map

    @property
    def get_id_to_features_map(self):
        return self._id_to_features_map

    def __len__(self):
        return len(self._id_list)

    def __getitem__(self, index):  # x, y
            return self._id_to_features_map[self._ids_list[index]], self._id_to_tag_map[self._ids_list[index]]

    @property
    def get_ids_list(self):
        return self._ids_list

    @property
    def get_id_to_tag_map(self):
        return self._id_to_tag_map

    @property
    def get_preproccessed_data(self):
        return self._preproccessed_data

    @property
    def get_pca_obj(self):
        return self._pca_obj

    def drow_data(self, preproccessed_data):
        nan_vals = preproccessed_data.isnull().sum().sum()
        plt.imshow(preproccessed_data)
        plt.imshow(preproccessed_data.apply(zscore))
        data_after_log = (preproccessed_data + 0.1).apply(np.log10)
        plt.imshow(data_after_log)
        plt.imshow(data_after_log.apply(zscore))
        data_after_log_zcore = data_after_log.apply(zscore)
        # plt.imsave('preproccessed_data____.png', (data))

        for i, x in enumerate(preproccessed_data.iterrows()):
            plt.hist(np.array(x[1]), normed=True, bins=30)
            plt.title("sample " + str(i) + ": " + x[0])
            plt.xlabel('Bacteria value')
            plt.ylabel('Probability')
            plt.show()
            plt.savefig('preproccessed_data_hist_' + str(i) + '.png')

    def _read_file(self, title, bactria_as_feature_file, samples_data_file, allow_printing, perform_anna_preprocess):
        raise NotImplemented
        """
        self._preproccessed_data = preproccessed_data
        self._pca_obj = pca_obj
        self._index_to_id_map = index_to_id_map
        self._id_to_features_map = id_to_features_map
        self._ids_list = ids_list
        make your own -> self._id_to_tag_map = _id_to_tag_map
        self._weight_map = classes_ratio
        self._feature_list = feature_list
        """

    def get_confusin_matrix_names(self):
        raise NotImplemented
    """
    write the suitable names for the data set
    """


    def get_learning_data(self,title):
        return self.get_ids_list, self.get_id_to_tag_map, self._task

    def get_weights(self):
        return self._weight_map



    def get_X_y_for_nni(self):
        ids, tag_map, task_name = self.get_learning_data()
        id_to_features_map = self.get_id_to_features_map
        X = [id_to_features_map[id] for id in ids]
        y = [tag_map[id] for id in ids]
        return np.array(X), np.array(y)


if __name__ == "__main__":
    task = 'success task'
    bactria_as_feature_file = 'feature-table_Allergy_cleaned_taxa_290119_updated_in_140219.csv'
    samples_data_file = 'mf_merge_ok84_ok93_ok66_69_merged_by_RestoredSampleCode_as_ID_290119.csv'
    tax = 6

    allergy_dataset = AbstractDataLoader(title=task, bactria_as_feature_file=bactria_as_feature_file,
                                         samples_data_file=samples_data_file,  taxnomy_level=tax,
                                         allow_printing=True, perform_anna_preprocess=False)
