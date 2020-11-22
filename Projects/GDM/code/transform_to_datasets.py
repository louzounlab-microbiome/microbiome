import pickle
import pandas as pd
from pathlib import Path

from sklearn.model_selection import cross_validate
# Load the data-sets we would like to compare.
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

with open(Path('../data/exported_data/otuMF'), 'rb') as otu_file:
    otuMf = pickle.load(otu_file)

latent_representation_samples = pd.read_csv(Path('../data/exported_data/latent_representation.csv'), index_col=0)
print(latent_representation_samples.shape)
# transform the target column into a numeric binary column
otuMf.tags_df['Tag'] = otuMf.tags_df['Tag'].apply(lambda tag: 0 if tag == 'Control' else 1)

# extract the saliva samples and their targets from the tables
saliva_samples = otuMf.otu_features_df[otuMf.extra_features_df['body_site'] == 'SALIVA'].copy()
saliva_tag = otuMf.tags_df.loc[saliva_samples.index]['Tag']

saliva_samples.to_csv(Path('../data/exported_data/saliva_dataset.csv'))
saliva_tag.to_csv(Path('../data/exported_data/saliva_tag.csv'))
# extract the stool samples and their targets from the tables

stool_samples = otuMf.otu_features_df[otuMf.extra_features_df['body_site'] == 'STOOL'].copy()
stool_tag = otuMf.tags_df.loc[stool_samples.index]['Tag']

stool_samples.to_csv(Path('../data/exported_data/stool_dataset.csv'))
stool_tag.to_csv(Path('../data/exported_data/stool_tag.csv'))
# Create a label for each patient
full_mapping_table = pd.merge(otuMf.extra_features_df, otuMf.tags_df, left_index=True, right_index=True,
                              how='left')
patient_df = full_mapping_table.groupby(
    by=['womanno.']).first().reset_index().set_index('womanno.')

# Add a latent representation for each patient.
latent_representation_tag = \
    pd.merge(latent_representation_samples, patient_df, how='left', left_index=True, right_index=True)['Tag']


latent_representation_samples.to_csv(Path('../data/exported_data/latent_representation_dataset.csv'))
latent_representation_tag.to_csv(Path('../data/exported_data/latent_representation_tag.csv'))

