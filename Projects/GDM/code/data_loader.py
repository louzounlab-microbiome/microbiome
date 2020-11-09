from LearningMethods.create_otu_and_mapping_files import CreateOtuAndMappingFiles
import pickle
bactria_as_feature_file = '../data/data_used/table-with-taxonomy.csv'
samples_data_file = '../data/data_used/samples_metadata.csv'
tax = 6

preprocess_prms = {'taxonomy_level': tax, 'taxnomy_group': 'sub PCA', 'epsilon': 0.1, 'normalization': 'log',
                   'z_scoring': 'row', 'norm_after_rel': '', 'std_to_delete': 0, 'pca': (0, 'PCA')}
otuMf = CreateOtuAndMappingFiles(bactria_as_feature_file, samples_data_file)
otuMf.remove_duplicates(['womanno.', 'trimester', 'body_site'])
otuMf.conditional_identification({'Type': 'Mother'})
otuMf.to_correspond(left_index=True, right_index=True, how='inner')
otuMf.preprocess(preprocess_prms, visualize=False)

with open('../data/data_used/otuMF', 'wb') as otu_file:
    pickle.dump(otuMf, otu_file)
