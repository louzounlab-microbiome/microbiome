from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
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

else:
    with open(Path('../data/used_data/CRC_ONLY/otumf_data/decomposed_otumf'), 'rb') as otumf_file:
        decomposed_otumf = pickle.load(otumf_file)

decomposed_otu = decomposed_otumf.otu_features_df
mapping_table = decomposed_otumf.extra_features_df
tag_series = decomposed_otumf.tags_df['Tag']

zero_tp_otu = decomposed_otu[mapping_table['TimePointNum'] == 0]

if not crc_only_flag:
    zero_tp_tag = tag_series[mapping_table['TimePointNum'] == 0].apply(lambda x: 1 if x > 0 else 0)

else:
    median_tumor = np.median(tag_series[mapping_table['TimePointNum'] == 0])
    zero_tp_tag = tag_series[mapping_table['TimePointNum'] == 0].apply(lambda x: 1 if x > median_tumor else 0)


score = 'roc_auc'
max_neigh = 10
cv = 5
test_scores = []
for neigh in range(1, max_neigh + 1):
    knn = KNeighborsClassifier(n_neighbors=neigh)
    cv_results = cross_validate(knn, zero_tp_otu, zero_tp_tag, cv=cv, scoring=score)
    test_scores.append(np.average(cv_results['test_score']))

plt.plot(range(1, max_neigh + 1), test_scores,c ='k',marker='o',markersize=15,markerfacecolor='red')
plt.show()
