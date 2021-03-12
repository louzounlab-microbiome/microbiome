import pandas as pd
import pickle
from LearningMethods.create_otu_and_mapping_files import CreateOtuAndMappingFiles
from pathlib import Path
import argparse
parser = argparse.ArgumentParser(description='CRC in  mice analysis')

parser.add_argument('--CRC_only', '--c',
                    dest="crc_only_flag", action='store_true',
                    help='only crc mice should be considered')

args = parser.parse_args()
crc_only_flag = args.crc_only_flag
if not crc_only_flag:
    otu_path=Path('../data/used_data/CRC_AND_NORMAL/basic_data/otu.csv')
    mapping_path=Path('../data/used_data/CRC_AND_NORMAL/basic_data/mapping_table.csv')
    otumf_path = Path('../data/used_data/CRC_AND_NORMAL/otumf_data/decomposed_otumf')
    decomposition_path = Path('../data/used_data/CRC_AND_NORMAL/basic_data/decomposed_table.csv')
else:
    otu_path = Path('../data/used_data/CRC_ONLY/basic_data/otu.csv')
    mapping_path = Path('../data/used_data/CRC_ONLY/basic_data/mapping_table.csv')
    otumf_path = Path('../data/used_data/CRC_ONLY/otumf_data/decomposed_otumf')
    decomposition_path = Path('../data/used_data/CRC_ONLY/basic_data/decomposed_table.csv')

otumf=CreateOtuAndMappingFiles(otu_path,mapping_path)
otumf.to_correspond(left_index=True,right_index=True)
tax=6

# Preprocess the data and decompose it using pca to 2 dimensions
preprocess_prms = {'taxonomy_level': tax, 'taxnomy_group': 'mean', 'epsilon': 1, 'normalization': 'log',
                   'z_scoring': 'row', 'norm_after_rel': '', 'std_to_delete': 0, 'pca': (5, 'PCA'),
                   'correlation_threshold': 0.8,'rare_bacteria_threshold':1}
otumf.preprocess(preprocess_prms,visualize=False)
otumf.otu_features_df.to_csv()

with open(otumf_path,'wb') as otumf_file:
    pickle.dump(otumf,otumf_file)
