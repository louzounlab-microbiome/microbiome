import pickle
import pandas as pd
from pathlib import Path

# Load the data-sets we would like to compare.
from sklearn.neighbors import KNeighborsClassifier

"""The purpose of this file is to transform the given data-sets into a shape that can be used by the learning models i.e
 , data, labels form. all transformed datasets are saved in exported_data folder."""


def pick_index(id0, id1):
    if id0 != 'NO_ID0' and id1 != 'NO_ID1':
        return id0
    elif id0 != 'NO_ID0':
        return id0
    return id1


with open(Path('../data/data_used/otumf_objects/TP12_saliva_otumf'), 'rb') as otu_file:
    saliva_otuMf = pickle.load(otu_file)

with open(Path('../data/data_used/otumf_objects/TP12_stool_otumf'), 'rb') as otu_file:
    stool_otuMf = pickle.load(otu_file)

latent_representation_samples = pd.read_csv(
    Path('../data/exported_data/latent_representation_data/TP12/latent_representation.csv'), index_col=0)
saliva_otu = saliva_otuMf.otu_features_df.copy()

saliva_tag = saliva_otuMf.tags_df['Tag'].copy().dropna()

saliva_otu = \
    pd.merge(saliva_otu, saliva_tag, how='inner', left_index=True, right_index=True)

saliva_tag = saliva_otu['Tag']
saliva_otu = saliva_otu.drop('Tag', axis=1)

saliva_otu.to_csv(Path('../data/exported_data/data_for_learning/few_tp/saliva_otu.csv'))
saliva_tag.to_csv(Path('../data/exported_data/data_for_learning/few_tp/saliva_tag.csv'))

stool_otu = stool_otuMf.otu_features_df.copy()
stool_tag = stool_otuMf.tags_df['Tag'].copy()

stool_otu.to_csv(Path('../data/exported_data/data_for_learning/few_tp/stool_otu.csv'))
stool_tag.to_csv(Path('../data/exported_data/data_for_learning/few_tp/stool_tag.csv'))

new_index = pd.Index([pick_index(id0, id1) for id0, id1 in
                      zip(latent_representation_samples['id0'], latent_representation_samples['id1'])])

latent_representation_samples.index = new_index
latent_representation_samples.drop(['id0', 'id1'], axis=1, inplace=True)

all_tags = pd.concat([stool_tag, saliva_tag])
latent_representation_samples_and_tag = pd.merge(latent_representation_samples, all_tags, how='inner', left_index=True,
                                                 right_index=True)

latent_representation_tag = latent_representation_samples_and_tag['Tag']
latent_representation_samples = latent_representation_samples_and_tag.drop('Tag',axis=1)


latent_representation_samples.to_csv(Path('../data/exported_data/data_for_learning/few_tp/latent_representation_dataset.csv'))
latent_representation_tag.to_csv(Path('../data/exported_data/data_for_learning/few_tp/latent_representation_tag.csv'))

