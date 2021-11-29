import argparse
from pathlib import Path
import pickle
import pandas as pd

from LearningMethods.CorrelationFramework import CorrelationFramework
from LearningMethods.correlation_evaluation import SignificantCorrelation
from Plot.plot_positive_negative_bars import plot_positive_negative_bars
from Plot.plot_real_and_shuffled_hist import plot_real_and_shuffled_hist
import matplotlib.pyplot as plt
plt.rcParams["font.weight"] = "bold"
from utils import conditional_identification

parser = argparse.ArgumentParser(description='CRC in  mice analysis')

parser.add_argument('--CRC_only', '--c',
                    dest="crc_only_flag", action='store_true',
                    help='only crc mice should be considered')

args = parser.parse_args()
crc_only_flag = args.crc_only_flag

immune_columns = ['spleen_weight','cell_spleen','MDSC_GR1_spleen','MFI_zeta_spleen','cell_BM','MDSC_GR1_bm']

if not crc_only_flag:
    with open(Path('../data/used_data/CRC_AND_NORMAL/otumf_data/decomposed_otumf'), 'rb') as otumf_file:
        otumf = pickle.load(otumf_file)
        save_folder=Path('../visualizations/CRC_AND_NORMAL/correlation')
else:
    with open(Path('../data/used_data/CRC_ONLY/otumf_data/decomposed_otumf'), 'rb') as otumf_file:
        otumf = pickle.load(otumf_file)
        save_folder=Path('../visualizations/CRC_ONLY/correlation')

mapping_table = otumf.extra_features_df
target_time = 4
immune_system_list = []
for idx, cage, mice, exp,group in zip(mapping_table.index, mapping_table['CageNum'], mapping_table['MiceNum'],
                                mapping_table['Experiment'],mapping_table['Group']):

    dic = {'CageNum': cage, 'MiceNum': mice, 'Experiment': exp, 'TimePointNum': target_time}
    relevant_row = conditional_identification(mapping_table, dic)
    if relevant_row.empty:
        # If the mouse is Normal we already know it has zero tumors.
        pass
    else:
        item = relevant_row[immune_columns].iloc[0]
        immune_system_list.append(item)

immune_system_parameters_df = pd.DataFrame(immune_system_list,index=mapping_table.index)
first_tp_otu = otumf.otu_features_df_b_pca[mapping_table['TimePointNum'] == 0]
tag = otumf.tags_df[mapping_table['TimePointNum'] == 0]['Tag']
last_tp_immune_system_parameters = immune_system_parameters_df.loc[first_tp_otu.index]

for immune_column_name in last_tp_immune_system_parameters.columns:
    corr = CorrelationFramework(first_tp_otu,last_tp_immune_system_parameters[immune_column_name], random_seed=1)
    fig, (corr_ax, hist_ax) = plt.subplots(nrows=2, ncols=1)
    corr.plot.plot_positive_negative_bars(corr_ax,percentile=1, positive_dict = {'color':'k'},title = None,x_label = None)
    corr.plot.plot_real_and_shuffled_hist(hist_ax,title = None,x_label =None)

    corr_ax.set_xlabel('Correlation',fontsize = 20)
    hist_ax.set_xlabel('Correlation',fontsize = 20)
    fig.suptitle('Correlation between initial microbiome and {} '.format(immune_column_name), fontsize=16)
    plt.tight_layout()
    plt.show()

"""Correlation between otu  and tumor_load"""
corr = CorrelationFramework(first_tp_otu,tag, random_seed=1)
fig, (corr_ax, hist_ax) = plt.subplots(nrows=2, ncols=1)
corr.plot.plot_positive_negative_bars(corr_ax,percentile=1, positive_dict = {'color':'k'},title = None,x_label = None)
corr.plot.plot_real_and_shuffled_hist(hist_ax,title = None,x_label =None)

corr_ax.set_xlabel('Correlation',fontsize = 20)
hist_ax.set_xlabel('Correlation',fontsize = 20)
fig.suptitle('Correlation between initial microbiome and Tumor_load', fontsize=16)
plt.show()

"""Correlation between Immune system parameters and tumor_load"""

corr = CorrelationFramework(last_tp_immune_system_parameters,tag, random_seed=1)
fig, (corr_ax, hist_ax) = plt.subplots(nrows=2, ncols=1)
corr.plot.plot_positive_negative_bars(corr_ax,percentile=1, positive_dict = {'color':'k'},title = None,x_label = None)
corr.plot.plot_real_and_shuffled_hist(hist_ax,title = None,x_label =None)

corr_ax.set_xlabel('Correlation',fontsize = 20)
hist_ax.set_xlabel('Correlation',fontsize = 20)
fig.suptitle('Correlation between immune system parameters and Tumor_load', fontsize=16)
plt.show()


