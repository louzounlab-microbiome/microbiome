import pickle
import pandas as pd
from pathlib import Path

"""The purpose of this file is to transform the given data-sets into a shape that can be used by the learning models i.e
 , data, labels form. all transformed datasets are saved in exported_data folder."""


def pick_index(id0, id1):
    if id0 != 'NO_ID0' and id1 != 'NO_ID1':
        return id0
    elif id0 != 'NO_ID0':
        return id0
    return id1


with open(Path('../data/data_used/otumf_data/otuMF'), 'rb') as otu_file:
    otuMf = pickle.load(otu_file)

latent_representation_samples = pd.read_csv(Path('../data/exported_data/latent_representation_data'
                                                 '/latent_representation.csv'), index_col=0)

# transform the target column into a numeric binary column
otuMf.tags_df['Tag'] = otuMf.tags_df['Tag'].apply(lambda tag: 0 if tag == 'Control' else 1)

# extract the saliva samples and their targets from the tables
saliva_samples = otuMf.otu_features_df[otuMf.extra_features_df['body_site'] == 'SALIVA'].copy()
saliva_tag = otuMf.tags_df.loc[saliva_samples.index]['Tag']

saliva_samples.to_csv(Path('../data/exported_data/data_for_learning/saliva_dataset.csv'))
saliva_tag.to_csv(Path('../data/exported_data/data_for_learning/saliva_tag.csv'))
# extract the stool samples and their targets from the tables

stool_samples = otuMf.otu_features_df[otuMf.extra_features_df['body_site'] == 'STOOL'].copy()
stool_tag = otuMf.tags_df.loc[stool_samples.index]['Tag']

stool_samples.to_csv(Path('../data/exported_data/data_for_learning/stool_dataset.csv'))
stool_tag.to_csv(Path('../data/exported_data/data_for_learning/stool_tag.csv'))
# Create a label for each patient
full_mapping_table = pd.merge(otuMf.extra_features_df, otuMf.tags_df, left_index=True, right_index=True)

new_index = pd.Index([pick_index(id0, id1) for id0, id1 in
                      zip(latent_representation_samples['id0'], latent_representation_samples['id1'])])

latent_representation_samples.index=new_index
latent_representation_samples.drop(['id0','id1'],axis=1 ,inplace=True)

# Add a latent representation for each patient.
latent_representation_tag = \
    pd.merge(latent_representation_samples, full_mapping_table, left_index=True, right_index=True)['Tag']


latent_representation_samples.to_csv(Path('../data/exported_data/data_for_learning/latent_representation_dataset.csv'))
latent_representation_tag.to_csv(Path('../data/exported_data/data_for_learning/latent_representation_tag.csv'))

