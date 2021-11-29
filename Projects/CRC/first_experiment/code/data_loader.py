from pathlib import Path
import pandas as pd
from utils import conditional_identification
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='CRC in  mice analysis')

parser.add_argument('--CRC_only', '--c',
                    dest="crc_only_flag", action='store_true',
                    help='only crc mice should be considered')

args = parser.parse_args()
crc_only_flag = args.crc_only_flag
# The data time domain varies from 0-5 but doesn't include tp 4, therefore, tp 5 will be treated as tp 4
old_target_time_point = 5
new_target_time = 4

otu_path = Path('../data/original_data/otu.txt')
mapping_table_path = Path('../data/original_data/mapping_table.csv')
taxonomy_path = Path('../data/original_data/taxonomy.tsv')
# Load the data
otu_table = pd.read_csv(otu_path, delimiter='\t', index_col=0)
mapping_table = pd.read_csv(mapping_table_path, index_col=0)
taxonomy = pd.read_csv(taxonomy_path, index_col=0, delimiter='\t').drop('Confidence', axis=1)
otu_table = pd.merge(otu_table, taxonomy, right_index=True, left_index=True)

# rename columns before preprocess
otu_table.rename({'Taxon': 'taxonomy'}, inplace=True, axis=1)
# replace tp 5 to tp 4
mapping_table['TimePointNum'] = mapping_table['TimePointNum'].replace(old_target_time_point,new_target_time)
mapping_table['TimePoint'] = mapping_table['TimePointNum'].apply(lambda x: '{}{}'.format('T',str(x)))
otu_table = otu_table.transpose()
otu_table.rename_axis('ID', inplace=True)


if crc_only_flag:
    mapping_table = mapping_table[mapping_table['Group'] == 'CRC']

tag_list = []
# Find the number of tumors for each mouse based on it's sample in tp 5
for idx, cage, mice, exp,group in zip(mapping_table.index, mapping_table['CageNum'], mapping_table['MiceNum'],
                                mapping_table['Experiment'],mapping_table['Group']):

    dic = {'CageNum': cage, 'MiceNum': mice, 'Experiment': exp, 'TimePointNum': new_target_time}
    relevant_row = conditional_identification(mapping_table, dic)
    if relevant_row.empty:
        # If the mouse is Normal we already know it has zero tumors.
        if group == 'NORMAL':
            tag_list.append((idx, 0))
        else: 
            pass
    else:
        item = relevant_row['tumor_load'].iloc[0]
        if np.isnan(item):
            # If the mouse is Normal we already know it has zero tumors.
            if group == 'NORMAL':
                tag_list.append((idx, 0))
            else:
                pass
        else:
            tag_list.append((idx, item))

tag_series = pd.DataFrame(tag_list).set_index(0)[1].rename('Tag')
if crc_only_flag:
    median = np.median(tag_series)
    tag_series = tag_series.apply(lambda x: 1 if x>median else 0).rename('Tag')
else:
    tag_series = tag_series.apply(lambda x: 1 if x>0 else 0).rename('Tag')


mapping_table = mapping_table.merge(tag_series, left_index=True, right_index=True)
mapping_table.rename_axis('ID', inplace=True)

if not crc_only_flag:
    otu_table.to_csv(Path('../data/used_data/CRC_AND_NORMAL/basic_data/otu.csv'))
    mapping_table.to_csv(Path('../data/used_data/CRC_AND_NORMAL/basic_data/mapping_table.csv'))
else:
    otu_table.to_csv(Path('../data/used_data/CRC_ONLY/basic_data/otu.csv'))
    mapping_table.to_csv(Path('../data/used_data/CRC_ONLY/basic_data/mapping_table.csv'))