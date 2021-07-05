import sys, os
from LearningMethods.create_otu_and_mapping_files import CreateOtuAndMappingFiles
from pathlib import Path
preprocess_prms = {'taxonomy_level': 6, 'taxnomy_group': 'sub PCA', 'epsilon': 0.1,
                     'normalization': 'log', 'z_scoring': 'row', 'norm_after_rel': 'No',
                     'std_to_delete': 0, 'pca': (0, 'PCA')}

otu_file = Path('Example_datasets/OTU.csv')
tag_file = Path('Example_datasets/Tag.csv')
task_name = 'Example_datasets'

mapping_file = CreateOtuAndMappingFiles(otu_file, tag_file)
mapping_file.preprocess(preprocess_params=preprocess_prms, visualize=True)
# mapping_file.rhos_and_pca_calculation(main_task, preprocess_prms['taxonomy_level'], preprocess_prms['pca'], rhos_folder, pca_folder)
otu_path, mapping_path, pca_path = mapping_file.csv_to_learn(task_name, os.path.join(os.getcwd(), task_name), preprocess_prms['taxonomy_level'])
print('CSV files are ready after MIPMLP')
print('OTU file', otu_path)
print('mapping file', mapping_path)
print('PCA object file', pca_path)