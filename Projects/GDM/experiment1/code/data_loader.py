from LearningMethods.create_otu_and_mapping_files import CreateOtuAndMappingFiles
import pickle
from pathlib import Path
from integration_tools.utils.data.data_classes import DualDataset
from integration_tools.utils.transforms.transforms_classes import ToTensor
from torchvision import transforms

"""This file was created in order to prepare the data towards integration. the initial mapping file included many 
duplicated samples, therefore they were removed. in addition, the mapping table consisted of few babies samples, 
which wasn't necessary to my analysis hence, thy were removed. 
 
"""
# load the data (original mapping file and otu table)
bactria_as_feature_file = Path('../data/data_used/basic_data/table-with-taxonomy.csv')
samples_data_file = Path('../data/data_used/basic_data/samples_metadata.csv')
tax = 6

preprocess_prms = {'taxonomy_level': tax, 'taxnomy_group': 'sub PCA', 'epsilon': 0.1, 'normalization': 'log',
                   'z_scoring': 'row', 'norm_after_rel': '', 'std_to_delete': 0, 'pca': (0, 'PCA')}
# create the object remove duplicates and baby samples.
otuMf = CreateOtuAndMappingFiles(bactria_as_feature_file, samples_data_file)
otuMf.remove_duplicates(['womanno.', 'trimester', 'body_site'])
otuMf.conditional_identification({'Type': 'Mother'})

# After the modifications in the mapping table, make sure that the otu and mapping table correspond.
otuMf.to_correspond(left_index=True, right_index=True, how='inner')
otuMf.preprocess(preprocess_prms, visualize=False)

entities_dataset = DualDataset.from_sources(otuMf.otu_features_df,
                                            matching_info_source0=otuMf.extra_features_df[['womanno.', 'trimester']],
                                            separator=otuMf.extra_features_df['body_site'],
                                            transform=transforms.Compose([ToTensor()]))

with open(Path('../data/data_used/entities_datasets/entities'), 'wb') as entities_file:
    pickle.dump(entities_dataset, entities_file)

# save the object after the preprocess
with open(Path('../data/data_used/otumf_data/otuMF'), 'wb') as otu_file:
    pickle.dump(otuMf, otu_file)
