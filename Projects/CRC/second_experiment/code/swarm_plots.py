import pickle
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
sns.set(font_scale=1)
plt.rcParams["font.weight"] = "bold"

parser = argparse.ArgumentParser()
parser.add_argument('--all_in_one', action="store_true")
args = parser.parse_args()
with open(Path('../data/data_used/otuMF_6'), 'rb') as otumf_file:
    decomposed_otumf = pickle.load(otumf_file)
decomposed_otu = decomposed_otumf.otu_features_df
mapping_table = decomposed_otumf.extra_features_df
mapping_table['Transplant'] = mapping_table['Transplant'].replace({'CRC':'CAC','normal':'Control'})
mapping_table['Treatment'] = mapping_table['Treatment'].replace({'CRC':'CAC','normal':'Control'})


decomposed_otu_with_tp = pd.merge(decomposed_otu,mapping_table[['DayOfSam','Transplant','Treatment']],left_index=True,right_index=True)
decomposed_otu_with_tp = decomposed_otu_with_tp[decomposed_otu_with_tp['DayOfSam'] != 'POOL']
if not args.all_in_one:
    for dayofsam in mapping_table['DayOfSam'].unique():
        print(f'The day is:{dayofsam}')
        if dayofsam == 'POOL':
            pass
        else:
            decomposed_otu_specific_tp = decomposed_otu_with_tp[decomposed_otu_with_tp['DayOfSam'] == dayofsam]
            for col_name in decomposed_otu.columns:
                ax = sns.swarmplot(x='Transplant',y = col_name,data=decomposed_otu_specific_tp,hue = 'Treatment')
                ax.set_title(f'Swarm plot of {col_name} in Time Point {dayofsam}',size = 20)
                plt.show()
else:
    for col_name in decomposed_otu.columns:
        ax = sns.swarmplot(x='Transplant', y=col_name, data=decomposed_otu_with_tp,hue = 'DayOfSam')
        ax.set_title(f'Swarm plot of {col_name} all Time Points',size = 20)
        plt.show()
