from load_merge_otu_mf import OtuMfHandler
from preprocess import preprocess_data
from pca import *
import scipy
from plot_confusion_matrix import *
import pandas as pd
import math
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
import math
import seaborn as sns; sns.set(color_codes=True)
import operator
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold,LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics, svm
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# otu = 'C:/Users/Anna/Desktop/docs/otu_psc2.csv'
# mapping = 'C:/Users/Anna/Desktop/docs/mapping_psc.csv'
# OtuMf = OtuMfHandler(otu, mapping, from_QIIME=False)
# preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxnomy_level=6)
#
# merged_data = preproccessed_data.join(OtuMf.mapping_file['DiagnosisGroup'])
#
# merged_data.fillna(0)
#
# mapping_disease_for_labels = {'Control':0,'Cirrhosis/HCC':1, 'PSC/PSC+IBD':2}
# mapping_disease = {'Control':0,'Cirrhosis ':1, 'HCC':1, 'PSC+IBD':2,'PSC':2}
# merged_data['DiagnosisGroup'] = merged_data['DiagnosisGroup'].map(mapping_disease)
#
#
# X = merged_data.loc[:, merged_data.columns != 'DiagnosisGroup']
# y = merged_data['DiagnosisGroup']


otu = 'C:/Users/Anna/Documents/otu_IBD3.csv'
mapping = 'C:/Users/Anna/Documents/mapping_IBD3.csv'
OtuMf = OtuMfHandler(otu, mapping, from_QIIME=False)
preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxnomy_level=6)
#visualize_pca(preproccessed_data)

#otu_after_pca, _ = apply_pca(preproccessed_data, n_components=30)
preproccessed_data = preproccessed_data.join(OtuMf.mapping_file[['CD_or_UC', 'preg_trimester', 'P-ID']], how ='inner')
preproccessed_data = preproccessed_data.loc[(preproccessed_data['CD_or_UC'] != 'control')]
preproccessed_data = preproccessed_data.groupby(['CD_or_UC', 'preg_trimester', 'P-ID'], as_index=False).mean()
#visualize_pca(preproccessed_data)
new_set2=preproccessed_data.groupby(['preg_trimester']).mean()
for i in range(0,len(preproccessed_data)):
    month = preproccessed_data['preg_trimester'][i]
    preproccessed_data.iloc[i:i+1,3:preproccessed_data.shape[1]] =  (preproccessed_data.iloc[i:i+1,3:preproccessed_data.shape[1]].values - new_set2.loc[month:month,:].values)
otu_after_pca, pca_components = apply_pca(preproccessed_data.drop(['CD_or_UC', 'preg_trimester', 'P-ID'], axis=1), n_components=50)
merged_data = otu_after_pca.join(preproccessed_data[['CD_or_UC', 'preg_trimester']], how ='inner')
#merged_data = preproccessed_data.drop(['P-ID'], axis=1)
merged_data = merged_data.fillna(0)
#merged_data = merged_data.loc[(merged_data['CD_or_UC'] != 'control')]
mapping_disease = {'CD':1,'UC':-1}

merged_data['CD_or_UC'] = merged_data['CD_or_UC'].map(mapping_disease)

merged_data= merged_data.reset_index()
try:
    merged_data=merged_data.drop('index',axis=1)
except:
    pass

X = merged_data.drop(['CD_or_UC', 'preg_trimester'], axis=1)

y = merged_data['CD_or_UC']

#
# otu = 'C:/Users/Anna/Documents/otu_saliva_GVHD.csv'
# mapping = 'C:/Users/Anna/Documents/mapping_saliva_GVHD.csv'
#
# OtuMf = OtuMfHandler(otu, mapping, from_QIIME=False)
# preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxnomy_level=6)
# mapping_file = OtuMf.mapping_file
# mapping_file['DATE'] = pd.to_datetime(OtuMf.mapping_file['DATE'])
# mapping_file['Date_Of_Transplantation'] = pd.to_datetime(OtuMf.mapping_file['Date_Of_Transplantation'])
# mapping_file['Date_of_engraftmen'] = pd.to_datetime(OtuMf.mapping_file['Date_of_engraftmen'])
# mapping_file['aGVHD1_Stat'] = pd.to_datetime(OtuMf.mapping_file['aGVHD1_Stat'])
# mapping_file['cGVHD_start'] = pd.to_datetime(OtuMf.mapping_file['cGVHD_start'])
# end = pd.to_datetime('2020-01-01')
# mapping_file['aGVHD1_Stat'] = mapping_file['aGVHD1_Stat'].fillna(end)
# mapping_file['cGVHD_start'] = mapping_file['cGVHD_start'].fillna(end)
# mapping_file = mapping_file[(mapping_file['DATE']>mapping_file['Date_Of_Transplantation']) & (mapping_file['DATE']<mapping_file['aGVHD1_Stat']) & (mapping_file['DATE']<mapping_file['cGVHD_start'])].sort_values(['Personal_ID', 'DATE'])
#
# mapping_file = mapping_file.reset_index()
# mapping_file = mapping_file.sort_values("DATE").groupby("Personal_ID", as_index=False).last().set_index('#SampleID')
# preproccessed_data = preproccessed_data.join(mapping_file[['MTX', 'Age', 'aGVHD1 ', 'cGVHD ']], how ='inner')
# preproccessed_data = preproccessed_data.fillna('No')
#
# mapping_yes_no = {'Yes':1,'No':0}
# preproccessed_data['aGVHD1 '] = preproccessed_data['aGVHD1 '].map(mapping_yes_no)
# preproccessed_data['cGVHD '] = preproccessed_data['cGVHD '].map(mapping_yes_no)
#
# preproccessed_data["disease"] = preproccessed_data["aGVHD1 "].map(str) + '_' +preproccessed_data["cGVHD "].map(str)
# mapping_diseases = {'0_0':1,'1_0':-1,'0_1':-1,'1_1':-1}
# preproccessed_data["disease"] = preproccessed_data["disease"].map(mapping_diseases)
#
#
# X = preproccessed_data.drop(['Age', 'MTX', 'aGVHD1 ', 'cGVHD ', 'disease'], axis=1)
# y = preproccessed_data['disease']

most_corelated_taxon = {}
for i in range(X.shape[1]):
    #if scipy.stats.spearmanr(predicted_data.iloc[:, i], predicted_data['pred'])[1]<0.05:  #/predicted_data.shape[1]:
    p_val = scipy.stats.spearmanr(X.iloc[:, i], y)[1]
    if math.isnan(p_val):
        most_corelated_taxon[X.columns[i]] = 1
    else:
        most_corelated_taxon[X.columns[i]] = p_val

sorted_taxon = sorted(most_corelated_taxon.items(), key=operator.itemgetter(1))
most_corelated_taxon = sorted_taxon[:round(X.shape[1]*0.2)]
#or
#most_corelated_taxon = [i for i in sorted_taxon if i[1]<=0.01]
bact = [i[0] for i in most_corelated_taxon]
new_data = X[bact]

#visualize_pca(new_data)
# def ibd_xgboost_with_dif_ncomp(n_components):
#     otu_after_pca, _ = apply_pca(new_data, n_components=n_components)
#     merged_data = otu_after_pca.join(merged_data[['CD_or_UC', 'preg_trimester']], how ='inner')
# #merged_data = preproccessed_data.drop(['P-ID'], axis=1)
#     merged_data = merged_data.fillna(0)
#
# # merged_data= merged_data.reset_index()
# # try:
# #     merged_data=merged_data.drop('index',axis=1)
# # except:
# #     pass
#
#     X = merged_data.drop(['CD_or_UC', 'preg_trimester'], axis=1)
#
#     y = merged_data['CD_or_UC']


regex = re.compile(r"\[|\]|<", re.IGNORECASE)
X.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X.columns.values]
loo = LeaveOneOut()
for md in range(3,5):
      for ne in range (50,300,50):
           for lr in range (5, 20, 5):
               for rg in range(250,400,25):
                    accuracy = []
                    y_pred_list = []
                    y_pred_list2 =[]
                    auc = []
                    auc_train = []
                    for train_index, test_index in loo.split(X):
                        train_index=list(train_index)
                        #print("%s %s" % (train_index, test_index))
                        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
                        y_train, y_test = y[train_index], y[test_index]
                        model = XGBClassifier(max_depth=md,n_estimators = ne ,learning_rate = lr/100,  #objective='multi:softmax' )
                                              objective= 'binary:logistic', scale_pos_weight = (np.sum(y_train==-1)/np.sum(y_train==1)),
                                              reg_lambda = rg)
                        model.fit(X_train, y_train)
                        y_pred2 = model.predict(X_test)
                        pred_train = model.predict_proba(X_train)[:, 1]
                        auc_train.append(metrics.roc_auc_score(y_train, pred_train))
                        y_pred = model.predict_proba(X_test)[:,1]
                        y_pred_list.append(y_pred[0])
                        y_pred_list2.append(y_pred2[0])
                    try:
                        auc = metrics.roc_auc_score(y, y_pred_list)
                    except:
                        pass
                    print('PCA components' + str(50), md, ne, lr, rg, round(auc, 2))


def gvhd_run_xgboost_with_pca(n_components):
    otu_after_pca, _ = apply_pca(new_data, n_components=n_components)

    merged_data = otu_after_pca.join(preproccessed_data[['MTX', 'disease']], how ='inner')
    #merged_data = preproccessed_data
    mapping_yes_no = {'Yes':1,'No':0}
    merged_data['MTX'] = merged_data['MTX'].map(mapping_yes_no)
    X = merged_data.drop(['disease', 'MTX'], axis=1)
    y = merged_data['disease']

    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    X.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X.columns.values]


    loo = LeaveOneOut()
    for md in range(3,4):
          for ne in range (250,300,50):
               for lr in range (10, 15, 5):
                   for rg in range(350,375,25):
                        accuracy = []
                        y_pred_list = []
                        y_pred_list2 =[]
                        auc = []
                        auc_train = []
                        for train_index, test_index in loo.split(X):
                            train_index=list(train_index)
                            #print("%s %s" % (train_index, test_index))
                            X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
                            y_train, y_test = y[train_index], y[test_index]
                            model = XGBClassifier(max_depth=md,n_estimators = ne ,learning_rate = lr/100,  #objective='multi:softmax' )
                                                  objective= 'binary:logistic', scale_pos_weight = (np.sum(y_train==-1)/np.sum(y_train==1)),
                                                  reg_lambda = rg)
                            model.fit(X_train, y_train)
                            y_pred2 = model.predict(X_test)
                            pred_train = model.predict_proba(X_train)[:, 1]
                            auc_train.append(metrics.roc_auc_score(y_train, pred_train))
                            y_pred = model.predict_proba(X_test)[:,1]
                            y_pred_list.append(y_pred[0])
                            y_pred_list2.append(y_pred2[0])
                        try:
                            auc = metrics.roc_auc_score(y, y_pred_list)
                        except:
                            pass
                        print('PCA components' + str(n_components), md, ne, lr, rg, round(auc, 2))

for i in range(5,25,5):
    gvhd_run_xgboost_with_pca(i)
##Clustergram
sns.set(font_scale=1)
g = sns.clustermap(new_data.T, cmap="RdYlGn", vmin=-1, vmax=1, col_cluster=False, yticklabels=False,
                       method='single')
ax = g.ax_heatmap
ax.set_ylabel('')
ax.set_xlabel('')
plt.show()

# p_values =[]
# for i in range(X.shape[1]):
#     p_values.append(scipy.stats.spearmanr(X.iloc[:, i], y)[1])
# print(p_values)
#
# p_values = [1.0 if math.isnan(x) else x for x in p_values]
# p_values = sorted([round(i,3) for i in p_values])
#
# p_values = pd.Series(p_values)
#
# plt.hist( p_values,  bins=100, weights=np.ones(len(p_values)) / len(p_values),
#                    color='#607c8e',cumulative=True,)
# plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
# plt.xticks(np.arange(0, 1, step=0.05), rotation='vertical')
# plt.xlabel('p-values')
# plt.show()
#
