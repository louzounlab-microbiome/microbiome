import pandas as pd
import pickle
from LearningMethods.create_otu_and_mapping_files import CreateOtuAndMappingFiles
from pathlib import Path
otu_path=Path('../data/exp1/used_data/basic_data/otu.csv')
mapping_path=Path('../data/exp1/used_data/basic_data/mapping_table.csv')

otumf=CreateOtuAndMappingFiles(otu_path,mapping_path)
otumf.to_correspond(left_index=True,right_index=True)
tax=6

# Preprocess the data and decompose it using pca to 2 dimensions
preprocess_prms = {'taxonomy_level': tax, 'taxnomy_group': 'sub PCA', 'epsilon': 0.1, 'normalization': 'log',
                   'z_scoring': 'row', 'norm_after_rel': '', 'std_to_delete': 0, 'pca': (2, 'PCA')}
otumf.preprocess(preprocess_prms,visualize=False)
otumf.otu_features_df.to_csv(Path('../data/exp1/used_data/basic_data/decomposed_table.csv'))
with open(Path('../data/exp1/used_data/otumf_data/decomposed_otumf'),'wb') as otumf_file:
    pickle.dump(otumf,otumf_file)
