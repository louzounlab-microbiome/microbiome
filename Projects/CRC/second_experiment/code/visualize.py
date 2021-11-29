from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from Plot.plot_3D import plot_data_3d
import argparse

plt.rcParams["font.weight"] = "bold"
parser=argparse.ArgumentParser()
parser.add_argument('--transplant_only',action='store_true')
args = parser.parse_args()
with open(Path('../data/data_used/otuMF_6'), 'rb') as otumf_file:
    decomposed_otumf = pickle.load(otumf_file)
decomposed_otu = decomposed_otumf.otu_features_df
if args.transplant_only:
    color_dict = {'normal': '#e6130b', 'CRC': '#30379c',}

    label_dict = {'normal': 'Control Transplant ', 'CRC': 'CAC Transplant'}
    sep = decomposed_otumf.extra_features_df['Transplant']
else:
    color_dict = {'normal.CRC':'#e6130b','normal.normal':'#30379c',
                  'CRC.CRC':'#12a615','CRC.normal':'#7d237d'}

    label_dict = {'normal.CRC':'Control Transplant CAC Treatment','normal.normal':'Control Transplant Control Treatment',
                  'CRC.CRC':'CAC Transplant CAC Treatment','CRC.normal':'CAC Transplant Control Treatment'}
    sep = decomposed_otumf.extra_features_df['TransplamtTreatment']

# Hadas requested a set of plots  considering only specific tp each time, tp = 15 should also include the pool
for dayofsam in decomposed_otumf.extra_features_df['DayOfSam'].unique():
    print(f'The day is:{dayofsam}')
    if dayofsam == 'POOL':
        pass
    else:
        decomposed_otu_specific_tp = decomposed_otu[decomposed_otumf.extra_features_df['DayOfSam'] == dayofsam]
        sep_specific_tp = sep[decomposed_otumf.extra_features_df['DayOfSam'] == dayofsam]
        fig, ax = plot_data_3d(decomposed_otu_specific_tp, sep_specific_tp
                               ,labels_dict=label_dict,color_dict=color_dict)
        if dayofsam == '15':
            control_pool = decomposed_otu[decomposed_otumf.extra_features_df['DayTransplant'] == 'POOL.normal']
            cac_pool = decomposed_otu[decomposed_otumf.extra_features_df['DayTransplant'] == 'POOL.crc']

            ax.scatter(control_pool.iloc[:, 0], control_pool.iloc[:, 1], control_pool.iloc[:, 2], depthshade=False,
                       label = 'POOL CAC', c= '#f2eb0a' )
            ax.scatter(cac_pool.iloc[:, 0], cac_pool.iloc[:, 1], cac_pool.iloc[:, 2], depthshade=False,
                       label='POOL Control', c='#ffb300')

            fig.legend()

        fig.set_facecolor('white')
        ax.set_facecolor('white')

        ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        for axis in [ax.w_xaxis, ax.w_yaxis, ax.w_zaxis]:
            axis.line.set_linewidth(3)
        plt.show()
