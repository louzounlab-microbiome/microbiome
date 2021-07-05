from pathlib import Path
import pandas as pd

"""
A Data loader for Oshrit's BGU project.
Oshrit provided me with the following files.
ALL_features_after_scale.csv - which is a table with otu and extra features combined, the table consists of 686 records.
metabolomics_scaled.csv - metabolomics table which provides more info upon the patients in the experiment.
tag.csv - a target table which determines whether a patient has a fatty liver. 

"""
path_to_all_features = Path('../data/original_data/ALL_features_after_scale.csv')
metabolomics_table_path = Path('../data/original_data/metabolomics_scaled.csv')
tag_path = Path('../data/original_data/tag.csv')

all_features_table = pd.read_csv(path_to_all_features)
# Some modifications towards Yoel's preprocess.
all_features_table.set_index('ID', inplace=True)
# Only a sub group of the columns is referencing to the otu data therefore, columns selection is made.
otu_features = all_features_table.loc[:, ' ;p__;c__;o__;f__;g__;s__':]

taxonomy = list(otu_features.columns)
otu_features.columns=range(otu_features.shape[1])
otu_features.loc['taxonomy'] = taxonomy
metabolomics_table = pd.read_csv(metabolomics_table_path, index_col=0)
tag_df = pd.read_csv(tag_path, index_col=0)
# More modifications towards Yoel's preprocess.
tag_df.rename(columns={'LI_B': 'Tag'},inplace=True)

otu_features.to_csv(Path('../data/data_used/basic_data/otu_features.csv'))
metabolomics_table.to_csv(Path('../data/data_used/basic_data/metabolomics_table.csv'))
tag_df.to_csv(Path('../data/data_used/basic_data/tag_df.csv'))
