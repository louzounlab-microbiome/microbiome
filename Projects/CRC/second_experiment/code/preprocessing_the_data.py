from LearningMethods.create_otu_and_mapping_files import CreateOtuAndMappingFiles
from pathlib import Path
import pickle
otu_path=Path('../data/data_used/otu.csv')
mapping_table_path=Path('../data/data_used/mapping_table.csv')
otuMf = CreateOtuAndMappingFiles(otu_path, mapping_table_path)
tax = 6
otuMf.to_correspond(left_index=True, right_index=True, how='inner')
otuMf.conditional_identification({'DayOfSam':'POOL'},not_flag=True)
preprocess_prms = {'taxonomy_level': tax, 'taxnomy_group': 'mean', 'epsilon': 0.1, 'normalization': 'log',
                   'z_scoring': 'row', 'norm_after_rel': '', 'std_to_delete': 0, 'pca': (0, 'PCA')}
otuMf.preprocess(preprocess_prms, visualize=False)

with open(Path('../data/data_used/otuMF'), 'wb') as otu_file:
    pickle.dump(otuMf, otu_file)

