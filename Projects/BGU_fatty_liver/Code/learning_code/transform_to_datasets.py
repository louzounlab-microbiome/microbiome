import pickle
import pandas as pd
from pathlib import Path

# Load the data-sets we would like to compare.

"""The purpose of this file is to transform the given data-sets into a shape that can be used by the learning models i.e
 , data, labels form. all transformed datasets are saved in exported_data folder."""

otu_features_path = Path('../../data/data_used/basic_data/preprocessed_otu.csv')
metabolomics_features_path = Path('../../data/data_used/basic_data/metabolomics_table.csv')

otu_features=pd.read_csv(otu_features_path,index_col=0)
metabolomics_features=pd.read_csv(metabolomics_features_path,index_col=0)

latent_representation_samples = pd.read_csv(Path('../../data/exported_data/latent_representation_data'
                                                 '/latent_representation.csv'), index_col=0)
latent_representation_samples=latent_representation_samples.set_index('id1').drop('id0',axis=1)
tag_df=pd.read_csv(Path('../../data/data_used/basic_data/tag_df.csv'),index_col=0)
tag_column_name='Tag'
full_otu_table=pd.merge(otu_features,tag_df,left_index=True,right_index=True)
otu_tag=full_otu_table[tag_column_name].copy()
otu_features=full_otu_table.drop(tag_column_name,axis=1).copy()
otu_features.to_csv(Path('../../data/exported_data/data_for_learning/GAN_integration/otu_dataset.csv'))
otu_tag.to_csv(Path('../../data/exported_data/data_for_learning/GAN_integration/otu_tag.csv'))

full_metabolomics_table=pd.merge(metabolomics_features,tag_df,left_index=True,right_index=True)
metabolomics_tag=full_metabolomics_table[tag_column_name].copy()
metabolomics_features=full_metabolomics_table.drop(tag_column_name,axis=1).copy()
metabolomics_features.to_csv(Path('../../data/exported_data/data_for_learning/GAN_integration/metabolomics_dataset.csv'))
metabolomics_tag.to_csv(Path('../../data/exported_data/data_for_learning/GAN_integration/metabolomics_tag.csv'))

full_latent_representation_table=pd.merge(latent_representation_samples,tag_df,left_index=True,right_index=True)
latent_representation_tag=full_latent_representation_table[tag_column_name].copy()
latent_representation_samples=full_latent_representation_table.drop(tag_column_name,axis=1).copy()
latent_representation_samples.to_csv(Path('../../data/exported_data/data_for_learning/GAN_integration/latent_representation_dataset.csv'))
latent_representation_tag.to_csv(Path('../../data/exported_data/data_for_learning/GAN_integration/latent_representation_tag.csv'))