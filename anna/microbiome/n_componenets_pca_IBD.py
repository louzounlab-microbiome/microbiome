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
pcas =[]
def pca_graph():
    for i in range (19,20):
        pcas.append(i)
        otu_after_pca, pca_components = apply_pca(preproccessed_data.drop(['CD_or_UC', 'preg_trimester', 'P-ID'], axis=1), n_components=i)
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
        try:
            auc = metrics.roc_auc_score(y, y_pred_list)
        except:
            pass
        print('PCA components' + str(50), round(auc, 2))
        scores = round(auc, 2)
        scores_train = round(np.array(auc_train).mean(), 2)
        train_accuracy.append(scores_train)
        test_accuracy.append(round(scores.mean(), 2))

def pca_graph_pvals_less_than():
    mapping_disease = {'CD': 1, 'UC': -1}
    preproccessed_data['CD_or_UC'] = preproccessed_data['CD_or_UC'].map(mapping_disease)

    X = preproccessed_data.drop(['P-ID', 'preg_trimester', 'CD_or_UC'], axis=1)

    y = preproccessed_data['CD_or_UC']

    for n_comp in range(2, 30):
            pcas.append(n_comp)

            loo = LeaveOneOut()

            y_pred_list = []
            auc = []
            auc_train = []
            for train_index, test_index in loo.split(X):
                train_index = list(train_index)
                # print("%s %s" % (train_index, test_index))
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y[train_index], y[test_index]
                most_corelated_taxon = {}
                for i in range(X_train.shape[1]):
                    p_val = scipy.stats.spearmanr(X_train.iloc[:, i], y_train)[1]
                    if math.isnan(p_val):
                        most_corelated_taxon[X_train.columns[i]] = 1
                    else:
                        most_corelated_taxon[X_train.columns[i]] = p_val
                sorted_taxon = sorted(most_corelated_taxon.items(), key=operator.itemgetter(1))
                most_corelated_taxon = sorted_taxon[:round(X_train.shape[1] * 0.2)]
                bact = [i[0] for i in most_corelated_taxon if i[0]!=1]
                new_data = X[bact]

                otu_after_pca, _ = apply_pca(new_data, n_components=n_comp)

                new_data = otu_after_pca.join(preproccessed_data['CD_or_UC'], how='inner')

                X_new = new_data.drop(['CD_or_UC'], axis=1)
                y_new = new_data['CD_or_UC']
                regex = re.compile(r"\[|\]|<", re.IGNORECASE)
                X_new.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in
                             X_new.columns.values]

                X_train, X_test = X_new.iloc[train_index], X_new.iloc[test_index]
                y_train, y_test = y_new[train_index], y_new[test_index]

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
            try:
                    auc =metrics.roc_auc_score(y, y_pred_list)
            except:
                    pass
            print('PCA components' + str(n_comp), round(auc,2))
            scores = round(auc,2)
            scores_train = round(np.array(auc_train).mean(), 2)
            train_accuracy.append(scores_train)
            test_accuracy.append(round(scores.mean(), 2))
pca_graph()
#pca_graph_pvals_less_than()
def plot_graph(test_accuracy, train_accuracy, pcas):
    plt.plot(pcas,test_accuracy, color ='red', label ='test')
    plt.plot(pcas,train_accuracy, color='blue', label ='train')
    plt.legend( loc=1,ncol=1)
    plt.show()
plot_graph(test_accuracy,train_accuracy,pcas)

print('scores_train')
print(train_accuracy)
print('scores_test')
print(test_accuracy)

