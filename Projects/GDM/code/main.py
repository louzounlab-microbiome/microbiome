import  pandas as pd
from LearningMethods.create_otu_and_mapping_files import CreateOtuAndMappingFiles
from Preprocess import preprocess_grid
import os
bactria_as_feature_file = '../data/data_used/table-with-taxonomy.csv'
samples_data_file = '../data/data_used/samples_metadata.csv'
tax = 6

preprocess_prms = {'taxonomy_level': tax, 'taxnomy_group': 'sub PCA', 'epsilon': 0.1, 'normalization': 'log',
                       'z_scoring': 'row', 'norm_after_rel': '', 'std_to_delete': 0, 'pca': (0, 'PCA')}
otuMf = CreateOtuAndMappingFiles(bactria_as_feature_file,samples_data_file)
otuMf.preprocess(preprocess_prms,visualize=True)
print(otuMf.otu_features_df)
