from pathlib import Path
import pickle
from Plot.plot_time_series_analysis import progress_in_time_of_column_attribute_mean
from Plot.plot_relationship_between_features import relationship_between_features
import matplotlib.pyplot as plt
import argparse
plt.rcParams["font.weight"] = "bold"

parser = argparse.ArgumentParser(description='CRC in  mice analysis')

parser.add_argument('--CRC_only', '--c',
                    dest="crc_only_flag", action='store_true',
                    help='only crc mice should be considered')

args = parser.parse_args()
crc_only_flag = args.crc_only_flag

if not crc_only_flag:
    with open(Path('../data/used_data/CRC_AND_NORMAL/otumf_data/decomposed_otumf'), 'rb') as otumf_file:
        decomposed_otumf = pickle.load(otumf_file)
    labels_dict = {1: 'CAC', 0: 'Control'}
    color_dict = {1: 'r', 0: 'k'}
    components_to_keep = [1,2]

    figure_dest_path = Path('../visualizations/CRC_AND_NORMAL')
    title = 'CAC and Control'

else:
    with open(Path('../data/used_data/CRC_ONLY/otumf_data/decomposed_otumf'), 'rb') as otumf_file:
        decomposed_otumf = pickle.load(otumf_file)
    labels_dict = {1: 'More Tumors', 0: 'Less Tumors'}
    color_dict = {1: 'r', 0: 'k'}
    components_to_keep = [0,1]
    figure_dest_path =Path('../visualizations/CRC_ONLY')
    title = 'CAC only'

# Plot the components progress in time
decomposed_otu = decomposed_otumf.otu_features_df
mapping_table = decomposed_otumf.extra_features_df
tag = decomposed_otumf.tags_df['Tag']

p1 = progress_in_time_of_column_attribute_mean(decomposed_otumf.otu_features_df.iloc[:,components_to_keep], mapping_table['TimePoint'],
                                               attribute_series=tag, margin=0.1,labels_dict = labels_dict,fontsize =12,colors=['r', 'k'])

new_plot = p1.plot()
new_plot.set_title(title,size = 20)
new_plot.set_xlabel('Time Point',size = 20)
new_plot.set_ylabel('Mean Value',size = 20)
plt.show()

# plot the  relationship between the components
relationship_between_features(decomposed_otu, folder=figure_dest_path,
                              separator=tag, labels_dict=labels_dict,color_dict=color_dict,axis_labels_size=25)
