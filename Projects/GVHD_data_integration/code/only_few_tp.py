from LearningMethods.create_otu_and_mapping_files import CreateOtuAndMappingFiles
from pathlib import Path
from integration_tools.utils.data.data_classes import DualDataset
from integration_tools.utils.transforms.transforms_classes import ToTensor
from torchvision import transforms
import pickle
import os
import pandas as pd
from datetime import datetime

time_differ = 10


def matching_fn(series0, series1):
    if series0['subjid'] == series1['subjid'] and abs((datetime.strptime(series0['DATE'],
                                                                         '%Y-%m-%d') - datetime.strptime(
            series1['DATE'], '%Y-%m-%d')).days) <= time_differ:
        return True
    return False


otu_paths = [Path('../data/data_used/basic_tables/stool_otu.csv'),
             Path('../data/data_used/basic_tables/saliva_otu.csv')]
mapping_paths = [Path('../data/data_used/basic_tables/stool_mapping_table.csv'),
                 Path('../data/data_used/basic_tables/saliva_mapping_table.csv')]
otumf_names_list = ['TP12_stool_otumf', 'TP12_saliva_otumf']
otumf_list = [CreateOtuAndMappingFiles(otu_path, mapping_path) for otu_path, mapping_path in
              zip(otu_paths, mapping_paths)]

tax = 6
preprocess_prms = {'taxonomy_level': tax, 'taxnomy_group': 'sub PCA', 'epsilon': 0.1, 'normalization': 'log',
                   'z_scoring': 'row', 'norm_after_rel': '', 'std_to_delete': 0, 'pca': (0, 'PCA')}

for otumf, otumf_name in zip(otumf_list, otumf_names_list):
    otumf.extra_features_df = otumf.extra_features_df[
        (otumf.extra_features_df['TimePoint'] == 1) | (otumf.extra_features_df['TimePoint'] == 2)]
    otumf.remove_duplicates(['subjid', 'DATE'])
    otumf.to_correspond(left_index=True, right_index=True, how='inner')
    otumf.preprocess(preprocess_prms, False)
    with open(os.path.join('../data/data_used/otumf_objects/', otumf_name), 'wb') as otu_file:
        pickle.dump(otumf, otu_file)

entities_dataset = DualDataset.from_sources(otumf_list[0].otu_features_df, otumf_list[1].otu_features_df,
                                            matching_info_source0=otumf_list[0].extra_features_df[['subjid', 'DATE']],
                                            matching_info_source1=otumf_list[1].extra_features_df[['subjid', 'DATE']],
                                            matching_fn=matching_fn,
                                            transform=transforms.Compose([ToTensor()]))

with open(Path('../data/data_used/entities_datasets/entities_TP12'), 'wb') as entities_file:
    pickle.dump(entities_dataset, entities_file)
