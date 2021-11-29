import pickle
import sys
import os
import numpy as np
# sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from Preprocess.preprocess_grid import preprocess_data
from pathlib import Path

class CreateOtuAndMappingFiles(object):
    # get two relative path of csv files
    def __init__(self, otu_file_path, tags_file_path,group_col_name = None):
        self.otu_path = otu_file_path
        self.tags_path = tags_file_path
        print('read tag file...')
        mapping_table = pd.read_csv(self.tags_path)
        if 'Tag' not in mapping_table.columns:
            raise ('A column named tag must appear in the extra features table')

        self.extra_features_df = mapping_table.set_index('ID').copy()
        self.tags_df = self.extra_features_df['Tag'].to_frame().copy()
        self.tags_df.index = self.tags_df.index.astype(str)
        self.extra_features_df.drop(['Tag'],axis=1,inplace=True)
        self.group_col_name = group_col_name

        # subset of ids according to the tags data frame
        self.ids = self.tags_df.index.tolist()
        self.ids.append('taxonomy')
        print('read otu file...')
        self.otu_features_df = pd.read_csv(self.otu_path).drop('Unnamed: 0', axis=1, errors='ignore')
        self.otu_features_df = self.otu_features_df.set_index('ID')
        self.otu_features_df.index = self.otu_features_df.index.astype(str)
        self.pca_ocj = None
        self.pca_comp = None
        self.preprocess_flag = False

    def export_to_learning_files(self,results_folder,otu_name = 'otu_dataset.csv',tag_name= 'tag.csv'
                                 , group_name = 'group.csv',to_correspond = False, to_correspond_kwargs = None):
        if to_correspond_kwargs is None:
            to_correspond_kwargs = {}

        if to_correspond:
            self.to_correspond(**to_correspond_kwargs)

        self.otu_features_df.to_csv(Path(results_folder).joinpath(otu_name))
        self.tags_df['Tag'].to_csv(Path(results_folder).joinpath(tag_name))
        if self.group_col_name is not None:
            self.extra_features_df[self.group_col_name].to_csv(Path(results_folder).joinpath(group_name))




    def preprocess(self, preprocess_params, visualize):
        # print('preprocess...')
        taxnomy_level = int(preprocess_params['taxonomy_level'])

        self.otu_features_df, self.otu_features_df_b_pca, self.pca_ocj, self.bacteria, self.pca_comp = preprocess_data(
            self.otu_features_df, preprocess_params, self.tags_df, visualize_data=visualize)
        # otu_features_df is the processed data, before pca
        if int(preprocess_params['pca'][0]) == 0:
            self.otu_features_df = self.otu_features_df_b_pca

        self.preprocess_flag = True

    def remove_duplicates(self, keys, filtering_fn=None):
        """
        Written by Sharon Komissarov.
        The function removes duplicates from the mapping table based on the keys inserted i.e, the function will
        group the mapping table based on the keys list inserted and filter each group using the filtering_fn.
        finally, the merged and filtered dataframe will be returned. keys: A list of column names that according to
        them the groupby will be applied.
        filtering_fn: A groupby function that will filter the groups,
        default first(), i.e all rows in the group excluding the first will be filtered.
        """
        if filtering_fn is None:

            no_duplicates_mapping_table = self.extra_features_df.reset_index().groupby(
                keys).first().reset_index().set_index('ID')
        else:
            no_duplicates_mapping_table = filtering_fn(self.extra_features_df)
        merged_table = pd.merge(no_duplicates_mapping_table, self.tags_df, left_index=True, right_index=True,
                                how='left')

        self.extra_features_df = merged_table.drop(['Tag'], axis=1).copy()
        self.tags_df = merged_table[['Tag']].copy()

    def conditional_identification(self, dic, not_flag=False,to_correspond = False,to_correspond_kwargs = None):
        """
        Written by Sharon Komissarov.
        The function facilitate in removing undesired rows by filtering them out.
        dic: the keys are the names of the columns which according to them the filtering will be applied.
        the filtering will be applied using the corresponding dic values.
        for example if you would like to keep only the normal rows, dic should look as follows:
        dic={'Group':'normal'}
        if you would like to keep only the normal rows and their saliva samples, dic should look as follows:
        dic={'Group':'normal','body_site':'saliva'}
        # Please notice that the function only modifies the mapping table and the tag!.
        """
        if to_correspond_kwargs is None:
            to_correspond_kwargs = {}
        if not not_flag:
            mask = pd.DataFrame([self.extra_features_df[key] == val for key, val in dic.items()]).T.all(axis=1)
        else:
            mask = ~pd.DataFrame([self.extra_features_df[key] == val for key, val in dic.items()]).T.all(axis=1)


        merged_table = pd.merge(self.extra_features_df[mask].copy(), self.tags_df, left_index=True, right_index=True,
                                how='left')

        self.tags_df = merged_table['Tag'].to_frame()
        self.extra_features_df = merged_table.drop(['Tag'], axis=1,errors='ignore').copy()
        if to_correspond:
            self.to_correspond(**to_correspond_kwargs)

    def to_correspond(self, **kwargs):

        """Written by Sharon Komissarov.
            the function merges and separate the otu, mapping table and tag in order to make them correspond.
            kwargs are controlling the merging additional attributes.
            Currently the function can only be used before the preprocess"""
        otu_columns_len, mapping_table_columns_len = self.otu_features_df.shape[1], self.extra_features_df.shape[1]

        if not self.preprocess_flag:
            taxonomy = self.otu_features_df.iloc[-1].copy()

        else:
            taxonomy = None



        full_mapping_table = pd.merge(self.extra_features_df,self.tags_df,left_index=True,right_index=True)


        merged_table = pd.merge(full_mapping_table, self.otu_features_df, **kwargs)

        self.tags_df = merged_table['Tag'].to_frame().copy()
        self.otu_features_df = merged_table[self.otu_features_df.columns].copy()
        if taxonomy is not None:
            self.otu_features_df = self.otu_features_df.append(taxonomy)

        self.extra_features_df = merged_table[self.extra_features_df.columns].copy()

        assert otu_columns_len == self.otu_features_df.shape[1] and \
               mapping_table_columns_len == self.extra_features_df.shape[1]



if __name__ == "__main__":
    task = 'prognostic_VItamin_A_task'
    bactria_as_feature_file = '../pregnancy_diabetes/merged_microbiome_taxonomy.csv'
    samples_data_file = '../pregnancy_diabetes/mapping_table.csv'
    rhos_folder = os.path.join('..', 'pregnancy_diabetes', 'rhos')
    pca_folder = os.path.join('..', 'pregnancy_diabetes', 'pca')

    # parameters for preprocess
    tax = 6
    preprocess_prms = {'taxonomy_level': tax, 'taxnomy_group': 'sub PCA', 'epsilon': 0.1, 'normalization': 'log',
                       'z_scoring': 'No', 'norm_after_rel': '', 'std_to_delete': 0, 'pca': (0, 'PCA')}

    mapping_file = CreateOtuAndMappingFiles(bactria_as_feature_file, samples_data_file)
    mapping_file.preprocess(preprocess_params=preprocess_prms, visualize=False)
    # mapping_file.rhos_and_pca_calculation(task, preprocess_prms['taxonomy_level'], preprocess_prms['pca'],
    #                                      rhos_folder, pca_folder)

    otu_path, tag_path, pca_path = mapping_file.csv_to_learn('VItamin_A_task', os.path.join('..', 'pregnancy_diabetes'),
                                                             tax=tax)

    print(otu_path)
    print(tag_path)
    print(pca_path)
