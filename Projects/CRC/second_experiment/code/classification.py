import pickle
from pathlib import Path

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, LeaveOneOut
import matplotlib.pyplot as plt
plt.rcParams["font.weight"] = "bold"

with open(Path('../data/data_used/otuMF_6'), 'rb') as otumf_file:
    decomposed_otumf = pickle.load(otumf_file)
decomposed_otu = decomposed_otumf.otu_features_df
mapping_table = decomposed_otumf.extra_features_df
tag_series = decomposed_otumf.tags_df['Tag']
zero_tp_otu = decomposed_otu[(mapping_table['DayOfSam'] == '15')&(mapping_table['Treatment'] == 'CRC')]
zero_tp_tag = tag_series[(mapping_table['DayOfSam'] == '15')&(mapping_table['Treatment'] == 'CRC')].apply(lambda x: 1 if x > 0 else 0)
title = 'Day 15 Tumor_load prediction'
loo = LeaveOneOut()
sample_prob_list = []
sample_tag_list = []
for train_index, test_index in loo.split(zero_tp_otu):
    train_x , test_x = zero_tp_otu.iloc[train_index],zero_tp_otu.iloc[test_index]
    train_y,test_y = zero_tp_tag.iloc[train_index],zero_tp_tag.iloc[test_index]
    rfc = RandomForestClassifier(n_estimators=7,random_state=0)
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