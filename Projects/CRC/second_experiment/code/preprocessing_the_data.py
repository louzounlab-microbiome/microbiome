from LearningMethods.create_otu_and_mapping_files import CreateOtuAndMappingFiles
from pathlib import Path
import pickle
otu_path=Path('../data/data_used/otu.csv')
mapping_table_path=Path('../data/data_used/mapping_table.csv')
otuMf = CreateOtuAndMappingFiles(otu_path, mapping_table_path)
tax = 6
otuMf.to_correspond(left_index=True, right_index=True, how='inner')
preprocess_prms = {'taxonomy_level': tax, 'taxnomy_group': 'mean', 'epsilon': 1, 'normalization': 'log',
                   'z_scoring': 'row', 'norm_after_rel': '', 'std_to_delete': 0, 'pca': (3, 'PCA')}

otuMf.preprocess(preprocess_prms, visualize=False)
# The axis number should start from 1
otuMf.otu_features_df.columns = list(map(lambda x: '{} {}'.format('Axis' , str(x+1)),otuMf.otu_features_df.columns))

with open(Path(f'../data/data_used/otuMF_{tax}'), 'wb') as otu_file:
    pickle.dump(otuMf, otu_file)

otuMf.otu_features_df_b_pca.to_csv(Path('../data/data_used/preprocessed_otu_before_pca.csv'))
otuMf.extra_features_df.to_csv(Path('../data/data_used/extra_features_df.csv'))

