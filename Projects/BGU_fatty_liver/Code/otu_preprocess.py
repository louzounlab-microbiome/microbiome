from integration_tools.utils.data.data_classes import DualDataset
from integration_tools.utils.transforms.transforms_classes import ToTensor
import pickle
from pathlib import Path
from torchvision import transforms
from LearningMethods.create_otu_and_mapping_files import CreateOtuAndMappingFiles
import pandas as pd

otu_path = Path('../data/data_used/basic_data/otu_features.csv')
tag_path = Path('../data/data_used/basic_data/tag_df.csv')
metabolomics_path = Path('../data/data_used/basic_data/metabolomics_table.csv')
# Create the Otu object
metabolomics_table = pd.read_csv(metabolomics_path, index_col=0)

otumf = CreateOtuAndMappingFiles(otu_path, tag_path)
tax = 5

preprocess_prms = {'taxonomy_level': tax, 'taxnomy_group': 'sub PCA', 'epsilon': 0.5, 'normalization': 'log',
                   'z_scoring': 'row', 'norm_after_rel': '', 'std_to_delete': 0, 'pca': (0, 'PCA')}
otumf.preprocess(preprocess_prms, visualize=False)
otumf.otu_features_df.to_csv(Path('../data/data_used/basic_data/preprocessed_otu.csv'))
with open(Path('../data/data_used/otumf_data/otumf'), 'wb') as otumf_file:
    pickle.dump(otumf, otumf_file)

dual_dataset = DualDataset.from_sources(otumf.otu_features_df,metabolomics_table,transform=transforms.Compose([ToTensor()]))

with open(Path('../data/data_used/entities_datasets/entities'), 'wb') as entities_file:
    pickle.dump(dual_dataset, entities_file)
