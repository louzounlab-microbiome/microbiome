import pickle
from LearningMethods.create_otu_and_mapping_files import CreateOtuAndMappingFiles
from pathlib import Path
otu_path=Path('../data/used_data/basic_data/otu.csv')
mapping_path=Path('../data/used_data/basic_data/mapping_table.csv')

otumf=CreateOtuAndMappingFiles(otu_path,mapping_path)
otumf.to_correspond(left_index=True,right_index=True)
tax=6

# Preprocess the data and decompose it using pca to 2 dimensions
preprocess_prms = {'taxonomy_level': tax, 'taxnomy_group': 'sub PCA', 'epsilon': 0.1, 'normalization': 'log',
                   'z_scoring': 'row', 'norm_after_rel': '', 'std_to_delete': 0,'pca': (0, 'PCA')}
otumf.preprocess(preprocess_prms,visualize=False)

with open(Path('../data/used_data/otumf_data/otumf'),'wb') as otumf_file:
    pickle.dump(otumf,otumf_file)
