import pandas as pd
from pathlib import Path
from LearningMethods.correlation_evaluation import SignificantCorrelation
import pickle
from Plot.plot_positive_negative_bars import plot_positive_negative_bars
from Plot.plot_real_and_shuffled_hist import plot_real_and_shuffled_hist
import matplotlib.pyplot as plt
from LearningMethods.CorrelationFramework import CorrelationFramework

plt.rcParams["font.weight"] = "bold"
with open(Path('../data/data_used/otuMF_6'), 'rb') as otumf_file:
    otumf = pickle.load(otumf_file)
crc_only_treatment = otumf.extra_features_df['Treatment'] == 'CRC'
for tp in otumf.extra_features_df['DayOfSam'].unique():
    otu_specific_tp = otumf.otu_features_df_b_pca[(otumf.extra_features_df['DayOfSam'] == tp) & crc_only_treatment]
    tag_specific_tp = otumf.tags_df['Tag'][(otumf.extra_features_df['DayOfSam'] == tp) & crc_only_treatment]
    cf = CorrelationFramework(otu_specific_tp, tag_specific_tp, random_seed=1)
    corr_ax = plt.subplot()
    cf.plot.plot_positive_negative_bars(corr_ax, percentile=1, positive_dict={'color': 'k'})
    corr_ax.set_xlabel('Correlation', fontsize=20)
    corr_ax.set_title(f'Correlation between microbiome day {tp} and tumor_load ', fontsize=16)
    corr_ax.tick_params(axis='y', which='major', labelsize=20)
    plt.show()
