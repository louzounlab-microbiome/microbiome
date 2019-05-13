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
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold,LeaveOneOut, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics, svm
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

otu = 'C:/Users/Anna/Documents/otu_IBD3.csv'
mapping = 'C:/Users/Anna/Documents/mapping_IBD3.csv'
max_num_of_pcas = 30

OtuMf = OtuMfHandler(otu, mapping, from_QIIME=False)
preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxnomy_level=7)
preproccessed_data = preproccessed_data.join(OtuMf.mapping_file[['CD_or_UC', 'preg_trimester', 'P-ID']], how ='inner')
preproccessed_data = preproccessed_data.loc[(preproccessed_data['CD_or_UC'] != 'control')]
preproccessed_data = preproccessed_data.groupby(['CD_or_UC', 'preg_trimester', 'P-ID'], as_index=False).mean()
new_set2=preproccessed_data.groupby(['preg_trimester']).mean()
for i in range(0,len(preproccessed_data)):
    month = preproccessed_data['preg_trimester'][i]
    preproccessed_data.iloc[i:i+1,3:preproccessed_data.shape[1]] =  (preproccessed_data.iloc[i:i+1,3:preproccessed_data.shape[1]].values - new_set2.loc[month:month,:].values)

preproccessed_data = preproccessed_data.drop(['preg_trimester', 'P-ID'], axis =1)
cols = [col for col in preproccessed_data.columns if col not in ['CD_or_UC'] and len(preproccessed_data[col].unique()) !=1]
dict_bact ={'else':[]}
for col in preproccessed_data[cols]:
    col_name = preproccessed_data[col].name.split(';')
    # if 'c__' in col_name[-1]:
    #     if  col_name[-1] in dict_bact:
    #         dict_bact[col_name[-1]].append(preproccessed_data[col].name)
    #     else:
    #         dict_bact[col_name[-1]] = [preproccessed_data[col].name]
    # else:
    #     dict_bact['else'].append(preproccessed_data[col].name)
    if len(col_name)>4:
        if col_name[4] in dict_bact:
            dict_bact[col_name[4]].append(preproccessed_data[col].name)
        else:
            dict_bact[col_name[4]] = [preproccessed_data[col].name]
    else:
        dict_bact['else'].append(preproccessed_data[col].name)
    print(col_name[-1])

new_df = pd.DataFrame(index = preproccessed_data.index)
col=0
for key, values in dict_bact.items():
    new_data = preproccessed_data[values]
    pca = PCA(n_components=round(new_data.shape[1] / 2)+1)
    pca.fit(new_data)
    sum = 0
    num_comp = 0
    for (i, component) in enumerate(pca.explained_variance_ratio_):
        if sum <= 0.5:
            sum += component
        else:
            num_comp = i
            break
    if num_comp == 0:
        num_comp += 1
    otu_after_pca_new, pca_components = apply_pca(new_data, n_components=num_comp)
    for j in range(otu_after_pca_new.shape[1]):
        new_df[col+j] = otu_after_pca_new[j]
    col += num_comp

#visualize_pca(new_df)
pcas =[]
train_accuracy = []
test_accuracy = []
for n_comp in range(1, max_num_of_pcas):
    pcas.append(n_comp)
    otu_ap, _ = apply_pca(new_df, n_components=n_comp)
    new_df2 = otu_ap.join(preproccessed_data['CD_or_UC'], how='inner')
    new_df2 = new_df2.fillna(0)
    mapping_disease = {'CD': 1, 'UC': -1}

    new_df2['CD_or_UC'] = new_df2['CD_or_UC'].map(mapping_disease)

    X= new_df2.drop(['CD_or_UC'], axis =1)
    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    X.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X.columns.values]

    y = new_df2['CD_or_UC']

    loo = LeaveOneOut()
    y_pred_list = []
    auc = []
    auc_train = []
    for train_index, test_index in loo.split(X):
        train_index = list(train_index)
        # print("%s %s" % (train_index, test_index))
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y[train_index], y[test_index]
        model = XGBClassifier(max_depth=3, n_estimators=250, learning_rate=15 / 100,
                              #objective='multi:softmax',
                              objective='binary:logistic',
                              scale_pos_weight=(np.sum(y_train == -1) / np.sum(y_train == 1)),
                              reg_lambda=250)
        model.fit(X_train, y_train)
        pred_train = model.predict_proba(X_train)[:, 1]
        auc_train.append(metrics.roc_auc_score(y_train, pred_train))
        y_pred = model.predict_proba(X_test)[:, 1]
        y_pred_list.append(y_pred[0])
    try:
        auc = metrics.roc_auc_score(y, y_pred_list)
    except:
        pass
    scores = round(auc, 2)
    scores_train = round(np.array(auc_train).mean(), 2)
    train_accuracy.append(scores_train)
    test_accuracy.append(round(scores.mean(), 2))

train_accuracy_all = []
test_accuracy_all = []
def pca_graph(max_num_of_pcas = max_num_of_pcas):
    for i in range (1,max_num_of_pcas):
        otu_after_pca, pca_components = apply_pca(preproccessed_data.drop(['CD_or_UC'], axis=1), n_components=i)
        merged_data = otu_after_pca.join(preproccessed_data['CD_or_UC'], how='inner')
        merged_data = merged_data.fillna(0)
        mapping_disease = {'CD': 1, 'UC': -1}

        merged_data['CD_or_UC'] = merged_data['CD_or_UC'].map(mapping_disease)

        merged_data = merged_data.reset_index()
        try:
            merged_data = merged_data.drop('index', axis=1)
        except:
            pass

        X = merged_data.drop(['CD_or_UC'], axis=1)

        y = merged_data['CD_or_UC']

        loo = LeaveOneOut()

        y_pred_list = []
        auc = []
        auc_train = []
        for train_index, test_index in loo.split(X):
            train_index = list(train_index)
            # print("%s %s" % (train_index, test_index))
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y[train_index], y[test_index]
            model = XGBClassifier(max_depth=3, n_estimators=250, learning_rate=15 / 100,
                                  # objective='multi:softmax' )
                                  objective='binary:logistic',
                                  scale_pos_weight=(np.sum(y_train == -1) / np.sum(y_train == 1)),
                                  reg_lambda=250)
            model.fit(X_train, y_train)
            pred_train = model.predict_proba(X_train)[:, 1]
            auc_train.append(metrics.roc_auc_score(y_train, pred_train))
            y_pred = model.predict_proba(X_test)[:, 1]
            y_pred_list.append(y_pred[0])

        auc = metrics.roc_auc_score(y, y_pred_list)
        print('PCA components' + str(50), round(auc, 2))
        scores = round(auc, 2)
        scores_train = round(np.array(auc_train).mean(), 2)
        train_accuracy_all.append(scores_train)
        test_accuracy_all.append(round(scores.mean(), 2))

pca_graph(max_num_of_pcas = max_num_of_pcas)
def plot_graph(test_accuracy, train_accuracy, train_accuracy_all,  test_accuracy_all, pcas):
    plt.plot(pcas,test_accuracy, color ='red', label ='test_fs')
    plt.plot(pcas,train_accuracy, color='blue', label ='train_fs')
    plt.plot(pcas, test_accuracy_all, color='orange', label='test')
    plt.plot(pcas, train_accuracy_all, color='black', label='train')
    plt.legend( loc=1,ncol=1)
    plt.show()
plot_graph(test_accuracy,train_accuracy,train_accuracy_all,  test_accuracy_all, pcas)
print('done')
