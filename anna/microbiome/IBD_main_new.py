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

otu = 'C:/Users/Anna/Documents/otu_IBD3.csv'
mapping = 'C:/Users/Anna/Documents/mapping_IBD3.csv'
OtuMf = OtuMfHandler(otu, mapping, from_QIIME=False)
preproccessed_data = preprocess_data(OtuMf.otu_file, taxnomy_level=6) #, visualize_data=False)
preproccessed_data = preproccessed_data.join(OtuMf.mapping_file[['CD_or_UC', 'preg_trimester', 'P-ID']], how ='inner')
preproccessed_data = preproccessed_data.loc[(preproccessed_data['CD_or_UC'] != 'control')]
preproccessed_data = preproccessed_data.groupby(['CD_or_UC', 'preg_trimester', 'P-ID'], as_index=False).mean()
new_set2=preproccessed_data.groupby(['preg_trimester']).mean()
for i in range(0,len(preproccessed_data)):
    month = preproccessed_data['preg_trimester'][i]
    preproccessed_data.iloc[i:i+1,3:preproccessed_data.shape[1]] =  (preproccessed_data.iloc[i:i+1,3:preproccessed_data.shape[1]].values - new_set2.loc[month:month,:].values)

train_accuracy = []
test_accuracy = []

otu_after_pca, pca_components = apply_pca(preproccessed_data.drop(['CD_or_UC', 'preg_trimester', 'P-ID'], axis=1), n_components=19)
merged_data = otu_after_pca.join(preproccessed_data[['CD_or_UC', 'preg_trimester']], how='inner')
merged_data = merged_data.fillna(0)
mapping_disease = {'CD': 1, 'UC': -1}

merged_data['CD_or_UC'] = merged_data['CD_or_UC'].map(mapping_disease)

merged_data = merged_data.reset_index()
try:
    merged_data = merged_data.drop('index', axis=1)
except:
    pass

X = merged_data.drop(['CD_or_UC', 'preg_trimester'], axis=1)

y = merged_data['CD_or_UC']

loo = LeaveOneOut()


for md in range(1,7):
    for ne in range(100,300,50):
        for lr in range(5,20,5):
            y_pred_list = []
            for train_index, test_index in loo.split(X):
                train_index = list(train_index)
                # print("%s %s" % (train_index, test_index))
                X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
                y_train, y_test = y[train_index], y[test_index]
                model = XGBClassifier(max_depth=md, n_estimators=ne, learning_rate=lr / 100,
                                      # objective='multi:softmax' )
                                      objective='binary:logistic',
                                      scale_pos_weight=(np.sum(y_train == -1) / np.sum(y_train == 1)),
                                      reg_lambda=250)
                model.fit(X_train, y_train)
                pred_train = model.predict_proba(X_train)[:, 1]
                #auc_train.append(metrics.roc_auc_score(y_train, pred_train))
                y_pred = model.predict_proba(X_test)[:, 1]
                y_pred_list.append(y_pred[0])
            #try:
            auc = metrics.roc_auc_score(y, y_pred_list)
            #except:
            #    pass
            print(md,ne,lr, round(auc, 2))

