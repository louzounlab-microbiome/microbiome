from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut

import numpy as np
import argparse
parser = argparse.ArgumentParser(description='CRC in  mice analysis')
plt.rcParams["font.weight"] = "bold"

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
    title = 'CAC and Control'

else:
    median_tumor = np.median(tag_series[mapping_table['TimePointNum'] == 0])
    zero_tp_tag = tag_series[mapping_table['TimePointNum'] == 0].apply(lambda x: 1 if x > median_tumor else 0)
    title = 'CAC only'
loo = LeaveOneOut()
sample_prob_list = []
sample_tag_list = []
for train_index, test_index in loo.split(zero_tp_otu):
    train_x , test_x = zero_tp_otu.iloc[train_index],zero_tp_otu.iloc[test_index]
    train_y,test_y = zero_tp_tag.iloc[train_index],zero_tp_tag.iloc[test_index]
    rfc = RandomForestClassifier(n_estimators=5, random_state=2)
    rfc.fit(train_x,train_y)
    sample_prob_list.append(rfc.predict_proba(test_x)[0][1])
    sample_tag_list.append(test_y.values[0])
fpr, tpr, thresholds = metrics.roc_curve(sample_tag_list, sample_prob_list)
ax = plt.gca()
ax.plot(fpr,tpr,c = 'r',label = f'RandomForestClassifier (AUC =  {round(roc_auc_score(sample_tag_list, sample_prob_list),2)})' )
ax.set_title(title,size = 20)
ax.set_xlabel('False Positive Rate',size = 20)
ax.set_ylabel('True Positive Rate',size = 20)
plt.legend()
plt.show()
"""
X_train, X_test, y_train, y_test = train_test_split(zero_tp_otu,zero_tp_tag,random_state=1,test_size = 0.3)
rfc = RandomForestClassifier(n_estimators=5, random_state=1)
rfc.fit(X_train, y_train)
ax = plt.gca()
rfc_disp = plot_roc_curve(rfc, X_test, y_test, ax=ax,drop_intermediate = False,color = 'r')
ax.set_title(title,size = 20)
ax.set_xlabel('False Positive Rate',size = 20)
ax.set_ylabel('True Positive Rate',size = 20)

plt.show()
"""

"""
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
"""