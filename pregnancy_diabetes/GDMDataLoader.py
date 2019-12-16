import sys
from collections import Counter
import os

from anna.microbiome.distance_learning_func import distance_learning

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LearningMethods.abstract_data_set import AbstractDataLoader
from infra_functions.load_merge_otu_mf import OtuMfHandler
from infra_functions.preprocess import preprocess_data
import pandas as pd
import numpy as np
from infra_functions.general import apply_pca

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

n_components = 5


class GDMDataLoader(AbstractDataLoader):
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

    def __init__(self, title, bactria_as_feature_file, samples_data_file, taxnomy_level, allow_printing,
                 perform_anna_preprocess, visualize=False, re_arange=0):
        super().__init__(title, bactria_as_feature_file, samples_data_file, taxnomy_level, allow_printing,
                         perform_anna_preprocess, visualize, re_arange)
        

    def _read_file(self, title, bactria_as_feature_file, samples_data_file, allow_printing, perform_anna_preprocess, visualize_pre, re_arange):

        features = pd.read_csv(bactria_as_feature_file, header=0)
        cols = list(features.columns)
        # remove non-numeric values
        #cols.remove('Feature ID')
        #cols.remove('Taxonomy')
        ids = list(features["#OTU ID"])
        ids.remove('taxonomy')
        sample = "STOOL"

        '''OtuMf = OtuMfHandler(os.path.join(SCRIPT_DIR, bactria_as_feature_file),
                             os.path.join(SCRIPT_DIR, samples_data_file),
                             from_QIIME=True, id_col='Feature ID', taxonomy_col='Taxonomy')
        '''

        OtuMf = OtuMfHandler(os.path.join(SCRIPT_DIR, bactria_as_feature_file),
                             os.path.join(SCRIPT_DIR, samples_data_file),
                             from_QIIME=False, id_col='#OTU ID',
                             taxonomy_col='taxonomy')

        rare_bacteria = self.find_rare_bacteria(OtuMf)
        OtuMf = self.drop_rare_bacteria(rare_bacteria, OtuMf)

        if perform_anna_preprocess:
            preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxonomy_col='taxonomy',
                                                 taxnomy_level=5)
            mapping_file = OtuMf.mapping_file['Control/GDM']
            mapping_disease = {'Control': 0,
                               'GDM': 1}
            mapping_file = mapping_file.map(mapping_disease)
            preproccessed_data, mapping_file = distance_learning(perform_distance=True, level=3,
                                                                 preproccessed_data=preproccessed_data,
                                                                 mapping_file=mapping_file)
            self._preproccessed_data = preproccessed_data

        else:
            if re_arange != 0:
                OtuMf = self.rearange_data(OtuMf, re_arange)
            preproccessed_data = preprocess_data(OtuMf.otu_file.T, visualize_data=visualize_pre, taxnomy_level=self._taxnomy_level,
                                                 taxonomy_col='taxonomy', preform_taxnomy_group=True, std_to_delete = 0.25)
        self.OtuMf = OtuMf
        self._preproccessed_data = preproccessed_data
        
        otu_after_pca_wo_taxonomy, pca_obj, _ = apply_pca(preproccessed_data, n_components=n_components,
                                                          visualize=False)
        self._pca_obj = pca_obj
        
        #This line ignore the PCA made above disable the line if PCA is needed 
        otu_after_pca_wo_taxonomy = self._preproccessed_data
        
        
        
        index_to_id_map = {}
        id_to_features_map = {}
        for i, row in enumerate(otu_after_pca_wo_taxonomy.values):
            id_to_features_map[otu_after_pca_wo_taxonomy.index[i]] = row
            index_to_id_map[i] = otu_after_pca_wo_taxonomy.index[i]
        self._index_to_id_map = index_to_id_map
        self._id_to_features_map = id_to_features_map

        ids_whole_list = otu_after_pca_wo_taxonomy.index.tolist()

        # ------------------------------------ each TASK creates different tag map --------------------------------
        id_to_tag_map = {}

        tag_map = {'Control': 0, 'GDM': 1}
        if sample == "both":
            T1_ids = [id for id in ids_whole_list if OtuMf.mapping_file["trimester"][id] == '1']
        else:
            T1_ids = [id for id in ids_whole_list if OtuMf.mapping_file["trimester"][id] == '1' and OtuMf.mapping_file["body_site"][id] == sample]
        counter_GDM = 0
        counter_Control = 0
        for id in T1_ids:
            id_to_tag_map[id] = tag_map[OtuMf.mapping_file["Control/GDM"][id]]
        self._ids_list = T1_ids
        self._id_to_tag_map = id_to_tag_map

        # -------------------------------------------- weights !--------------------------------------------
        # calculate weights
        y = list(self._id_to_tag_map.values())
        classes_sum = [np.sum(np.array(y) == unique_class) for unique_class in
                       np.unique(np.array(y))]
        classes_ratio = [1 - (a / sum(classes_sum)) for a in classes_sum]
        weights = [classes_ratio[a] for a in np.array(y)]
        self._weight_map = {i: classes_ratio[i] for i in range(len(classes_ratio))}

        # return the list of features and the list of ids in the same order
        self._feature_list = [self._id_to_features_map[id] for id in self._ids_list]

    def get_confusin_matrix_names(self):
        names = ['Control', 'GDM']
        return names

    def find_rare_bacteria(self, OtuMf):
        bact_to_num_of_non_zeros_values_map = {}
        otu = OtuMf.otu_file.T
        bacteria = otu.columns[1:]
        num_of_samples = len(otu.index) - 1
        for bact in bacteria:
            values = otu[bact]
            count_map = Counter(values)
            zeros = 0
            if 0 in count_map.keys():
                zeros += count_map[0]
            if '0' in count_map.keys():
                zeros += count_map['0']

            bact_to_num_of_non_zeros_values_map[bact] = num_of_samples - zeros

        rare_bacteria = []
        for key, val in bact_to_num_of_non_zeros_values_map.items():
            if val < 5:
                rare_bacteria.append(key)
        return rare_bacteria

    def drop_rare_bacteria(self, rare_bacteria_list, OtuMf):
        OtuMf.otu_file.drop(rare_bacteria_list, inplace=True)
        return OtuMf
    
    def rearange_data(self,OtuMf, flag):
        df_mapping_file = pd.DataFrame(OtuMf.mapping_file)
        df_otu = pd.DataFrame(OtuMf.otu_file.T)
        if flag ==1:
            dict_arange = {'GDM':{'SALIVA':[], 'STOOL':[]}, 'Control':{'SALIVA':[], 'STOOL':[]}}
            for index, row in df_mapping_file.iterrows():
                if index != '#q2:types' and row['body_site'] != 'Meconium' and not pd.isna(row['Control/GDM']):
                    if index in OtuMf.otu_file:
                        dict_arange[row['Control/GDM']][row['body_site']].append(index)
            idx_list = dict_arange['GDM']['SALIVA'] + dict_arange['GDM']['STOOL'] + dict_arange['Control']['SALIVA'] + dict_arange['Control']['STOOL']
        elif flag==2:
            dict_arange = {'SALIVA':{'GDM':[], 'Control':[]}, 'STOOL':{'GDM':[], 'Control':[]}}
            for index, row in df_mapping_file.iterrows():
                if index != '#q2:types' and row['body_site'] != 'Meconium' and not pd.isna(row['Control/GDM']):
                    if index in OtuMf.otu_file:
                        dict_arange[row['body_site']][row['Control/GDM']].append(index)
            idx_list = dict_arange['SALIVA']['GDM'] + dict_arange['SALIVA']['Control'] + dict_arange['STOOL']['GDM'] + dict_arange['STOOL']['Control']
        df = df_otu.reindex(idx_list).T
        df['taxonomy'] = OtuMf.otu_file['taxonomy'].values
        OtuMf.otu_file = df
        return OtuMf
    
if __name__ == "__main__":
    task = 'prognostic_diabetes_task'
    bactria_as_feature_file = 'merged_GDM_tables_w_tax.csv'
    samples_data_file = 'ok.csv'
    tax = int(sys.argv[1])
    if len(sys.argv) == 3:
        if_rearange = int(sys.argv[2])
    else:
        if_rearange = 0
    GDM_dataset = GDMDataLoader(title=task, bactria_as_feature_file=bactria_as_feature_file,
                                         samples_data_file=samples_data_file,  taxnomy_level=tax,
                                         allow_printing=True, perform_anna_preprocess=False, visualize = True, re_arange = if_rearange)