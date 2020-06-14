from sys import stdout
import sys; import pprint
from collections import Counter
import os
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anna.microbiome.distance_learning_func import distance_learning
from LearningMethods.abstract_data_set import AbstractDataLoader
from Projects.GVHD_BAR.load_merge_otu_mf import OtuMfHandler
from Preprocess.preprocess import preprocess_data
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
import numpy as np
from Preprocess.general import apply_pca
import seaborn as sns

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
                 perform_anna_preprocess, visualize = False, re_arange = 0):
        super().__init__(title, bactria_as_feature_file, samples_data_file, taxnomy_level, allow_printing,
                         perform_anna_preprocess, visualize, re_arange)
        

    def _read_file(self, title, bactria_as_feature_file, samples_data_file, allow_printing, perform_anna_preprocess, visualize_pre, re_arange):
        
        sample = "SALIVA"


        OtuMf = OtuMfHandler(os.path.join(SCRIPT_DIR, bactria_as_feature_file),
                             os.path.join(SCRIPT_DIR, samples_data_file),
                             from_QIIME=False, id_col='ID',
                             taxonomy_col='taxonomy')
        #rare_bacteria = self.find_rare_bacteria(OtuMf)
        #OtuMf = self.drop_rare_bacteria(rare_bacteria, OtuMf)
        OtuMf = self.remove_duplicate(OtuMf)
        OtuMf = self.rearange_data(OtuMf, re_arange)
        #OtuMf.otu_file.T.to_csv("GDM_OTU_rmv_dup_arrange.csv")
        OtuMf.mapping_file.to_csv("GDM_tag_rmv_dup.csv")
        #returnmapping_file
        return
        
        if perform_anna_preprocess:
            preproccessed_data = preprocess_data(OtuMf.otu_file.T, visualize_data=False, taxonomy_col='taxonomy',
                                                 taxnomy_level=8)
            mapping_file = OtuMf.mapping_file['Control_GDM']
            mapping_disease = {'Control': 0,
                               'GDM': 1}
            mapping_file = mapping_file.map(mapping_disease)
            preproccessed_data, mapping_file = distance_learning(perform_distance=True, level=4,
                                                                 preproccessed_data=preproccessed_data,
                                                                 mapping_file=mapping_file)
            self._preproccessed_data = preproccessed_data
            self._preproccessed_data.to_csv('anna_pca_old_loader.csv')
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
            T1_ids = [id for id in ids_whole_list if int(OtuMf.mapping_file["trimester"][id]) == 1 and OtuMf.mapping_file["body_site"][id] == sample]
        counter_GDM = 0
        counter_Control = 0
        for id in T1_ids:
            id_to_tag_map[id] = tag_map[OtuMf.mapping_file["Control_GDM"][id]]
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
                if row['body_site'] != 'Meconium' and not pd.isna(row['Control_GDM']):
                    if index in OtuMf.otu_file:
                        dict_arange[row['Control_GDM']][row['body_site']].append(index)
            idx_list = dict_arange['GDM']['SALIVA'] + dict_arange['GDM']['STOOL'] + dict_arange['Control']['SALIVA'] + dict_arange['Control']['STOOL']
        elif flag==2:
            dict_arange = {'SALIVA':{'GDM':[], 'Control':[]}, 'STOOL':{'GDM':[], 'Control':[]}}
            for index, row in df_mapping_file.iterrows():
                if row['body_site'] != 'Meconium' and not pd.isna(row['Control_GDM']):
                    if index in OtuMf.otu_file:
                        dict_arange[row['body_site']][row['Control_GDM']].append(index)
            idx_list = dict_arange['SALIVA']['GDM'] + dict_arange['SALIVA']['Control'] + dict_arange['STOOL']['GDM'] + dict_arange['STOOL']['Control']
        df = df_otu.reindex(idx_list).T
        df['taxonomy'] = OtuMf.otu_file['taxonomy'].values
        OtuMf.otu_file = df
        OtuMf.mapping_file = df_mapping_file.reindex(idx_list)
        return OtuMf
        
    def remove_duplicate(self,OtuMf):
        df_mapping_file = pd.DataFrame(OtuMf.mapping_file)
        df_otu = pd.DataFrame(OtuMf.otu_file.T)
        list_id  = df_otu.index.values.tolist()
        list_id.remove('taxonomy')
        ids_dict = {}
        ids_to_drop = []
        for sample in list_id:
            if sample[:-1] not in ids_dict:
                ids_dict[sample[:-1]] = (sample, (df_otu.loc[[sample]].values == 0).astype(int).sum(axis=1)[0])
            else:
                if (df_otu.loc[[sample]].values == 0).astype(int).sum(axis=1)[0] < int(ids_dict[sample[:-1]][1]):
                    ids_to_drop.append(ids_dict[sample[:-1]][0])
                    ids_dict[sample[:-1]] = (sample, (df_otu.loc[[sample]].values == 0).astype(int).sum(axis=1))[0]
                else:
                    ids_to_drop.append(sample)
        OtuMf.otu_file = df_otu.drop(index = ids_to_drop).T
        OtuMf.mapping_file = df_mapping_file.drop(index = ids_to_drop)
        return OtuMf
    
    
if __name__ == "__main__":
    task = 'prognostic_diabetes_task'
    bactria_as_feature_file = 'DB/GDM_taxonomy.csv'
    samples_data_file = 'DB/samples_metadata.csv'
    tax = 5
    if len(sys.argv) == 3:
        if_rearange = int(sys.argv[2])
    else:
        if_rearange = 1
    GDM_dataset = GDMDataLoader(title=task, bactria_as_feature_file=bactria_as_feature_file,
                                         samples_data_file=samples_data_file,  taxnomy_level=tax,
                                         allow_printing=True, perform_anna_preprocess=True, visualize = True, re_arange = if_rearange)