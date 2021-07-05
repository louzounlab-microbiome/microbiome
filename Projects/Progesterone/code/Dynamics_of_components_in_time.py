""" The progestrone data is composed of samples with few Timepoints, after the pca decomposition we would like to
observe the dynamics of the center of mass of each principal component in time. """

import pickle
from pathlib import Path
from Plot.plot_time_series_analysis import progress_in_time_of_column_attribute_mean
import matplotlib.pyplot as plt

with open(Path('../data/exp1/used_data/otumf_data/decomposed_otumf'), 'rb') as otumf_file:
    otumf = pickle.load(otumf_file)

otu_features = otumf.otu_features_df
mapping_table = otumf.extra_features_df
male_rows = mapping_table['gender'] == 'male'
female_rows=~male_rows
progress_in_males = progress_in_time_of_column_attribute_mean(otu_features[male_rows], mapping_table[male_rows]['day'],
                                                              mapping_table[male_rows]['treatment'])

progress_in_females = progress_in_time_of_column_attribute_mean(otu_features[female_rows], mapping_table[female_rows]['day'],
                                                              mapping_table[female_rows]['treatment'])
male_ax = progress_in_males.plot()
male_ax.set_title('Males')
male_ax.set_ylabel('Principal Components')
male_ax.set_xlabel('Time')


female_ax = progress_in_females.plot()
female_ax.set_title('Females')
female_ax.set_ylabel('Principal Components')
female_ax.set_xlabel('Time')
plt.show()
