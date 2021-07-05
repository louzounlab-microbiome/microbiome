from pathlib import Path
import pickle
from Plot.plot_time_series_analysis import progress_in_time_of_column_attribute_mean
from Plot.plot_relationship_between_features import relationship_between_features
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser(description='CRC in  mice analysis')

parser.add_argument('--CRC_only', '--c',
                    dest="crc_only_flag", action='store_true',
                    help='only crc mice should be considered')

args = parser.parse_args()
crc_only_flag = args.crc_only_flag

if not crc_only_flag:
    with open(Path('../data/used_data/CRC_AND_NORMAL/otumf_data/decomposed_otumf'), 'rb') as otumf_file:
        decomposed_otumf = pickle.load(otumf_file)
    labels_dict = {1: 'Tumor', 0: 'No Tumor'}
    color_dict = {1: 'r', 0: 'k'}
    figure_dest_path =Path('../visualizations/CRC_AND_NORMAL')
else:
    with open(Path('../data/used_data/CRC_ONLY/otumf_data/decomposed_otumf'), 'rb') as otumf_file:
        decomposed_otumf = pickle.load(otumf_file)
    labels_dict = {1: 'More Tumors', 0: 'Less Tumors'}
    color_dict = {1: 'r', 0: 'k'}
    figure_dest_path =Path('../visualizations/CRC_ONLY')

# Plot the components progress in time
decomposed_otu = decomposed_otumf.otu_features_df
mapping_table = decomposed_otumf.extra_features_df
tag = decomposed_otumf.tags_df['Tag']

p1 = progress_in_time_of_column_attribute_mean(decomposed_otumf.otu_features_df, mapping_table['TimePointNum'],
                                               attribute_series=tag, margin=0.1)
new_plot = p1.plot()
new_plot.set_title('Progress in time of column attribute mean PCA')
new_plot.set_xlabel('Time')
new_plot.set_ylabel('Mean Value')
plt.show()

# plot the  relationship between the components
relationship_between_features(decomposed_otu, folder=figure_dest_path,
                              separator=tag, labels_dict=labels_dict,color_dict=color_dict)
