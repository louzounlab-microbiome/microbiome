from load_merge_otu_mf import OtuMfHandler
from Preprocess import preprocess_data
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

otu = 'C:/Users/Anna/Documents/otu_saliva_GVHD.csv'
mapping = 'C:/Users/Anna/Documents/mapping_saliva_GVHD.csv'

OtuMf = OtuMfHandler(otu, mapping, from_QIIME=False)
preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxnomy_level=6)
mapping_file = OtuMf.mapping_file
mapping_file['DATE'] = pd.to_datetime(OtuMf.mapping_file['DATE'])
mapping_file['Date_Of_Transplantation'] = pd.to_datetime(OtuMf.mapping_file['Date_Of_Transplantation'])
mapping_file['Date_of_engraftmen'] = pd.to_datetime(OtuMf.mapping_file['Date_of_engraftmen'])
mapping_file['aGVHD1_Stat'] = pd.to_datetime(OtuMf.mapping_file['aGVHD1_Stat'])
mapping_file['cGVHD_start'] = pd.to_datetime(OtuMf.mapping_file['cGVHD_start'])
end = pd.to_datetime('2020-01-01')
mapping_file['aGVHD1_Stat'] = mapping_file['aGVHD1_Stat'].fillna(end)
mapping_file['cGVHD_start'] = mapping_file['cGVHD_start'].fillna(end)
mapping_file = mapping_file[(mapping_file['DATE']>mapping_file['Date_Of_Transplantation']) & (mapping_file['DATE']<mapping_file['aGVHD1_Stat']) & (mapping_file['DATE']<mapping_file['cGVHD_start'])].sort_values(['Personal_ID', 'DATE'])

mapping_file = mapping_file.reset_index()
mapping_file = mapping_file.sort_values("DATE").groupby("Personal_ID", as_index=False).last().set_index('#SampleID')
preproccessed_data = preproccessed_data.join(mapping_file[['MTX', 'Age', 'aGVHD1 ', 'cGVHD ']], how ='inner')
preproccessed_data = preproccessed_data.fillna('No')

mapping_yes_no = {'Yes':1,'No':0}
preproccessed_data['aGVHD1 '] = preproccessed_data['aGVHD1 '].map(mapping_yes_no)
preproccessed_data['cGVHD '] = preproccessed_data['cGVHD '].map(mapping_yes_no)
preproccessed_data['MTX'] = preproccessed_data['MTX'].map(mapping_yes_no)
preproccessed_data["disease"] = preproccessed_data["aGVHD1 "].map(str) + '_' +preproccessed_data["cGVHD "].map(str)
mapping_diseases = {'0_0':1,'1_0':-1,'0_1':-1,'1_1':-1}
preproccessed_data["disease"] = preproccessed_data["disease"].map(mapping_diseases)


train_accuracy = []
test_accuracy = []
pcas =[]
def pca_graph(preproccessed_data = preproccessed_data):
    for i in range (1,35):
        pcas.append(i)
        otu_after_pca, pca_components = apply_pca(preproccessed_data.drop(['MTX', 'Age', 'aGVHD1 ', 'cGVHD ', 'disease'], axis=1), n_components=i)
        merged_data = otu_after_pca.join(preproccessed_data[['MTX', 'disease']], how='inner')
        merged_data = merged_data.fillna(0)

        X = merged_data.drop(['disease'], axis=1)
        y = merged_data['disease']
        loo = LeaveOneOut()
        for md in range(5,6):
            for ne in range(300,350,50):
                for lr in range(15,20,5):
                    for rg in range(450,500,50):
                        y_pred_list = []
                        auc = []
                        auc_train = []
                        for train_index, test_index in loo.split(X):
                            train_index = list(train_index)
                            # print("%s %s" % (train_index, test_index))
                            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
                            y_train, y_test = y[train_index], y[test_index]
                            model = XGBClassifier(max_depth=md, n_estimators=ne, learning_rate=lr / 100,
                                                  # objective='multi:softmax' )
                                                  objective='binary:logistic',
                                                  scale_pos_weight=(np.sum(y_train == -1) / np.sum(y_train == 1)),
                                                  reg_lambda=rg)
                            model.fit(X_train, y_train)
                            pred_train = model.predict_proba(X_train)[:, 1]
                            auc_train.append(metrics.roc_auc_score(y_train, pred_train))
                            y_pred = model.predict_proba(X_test)[:, 1]
                            y_pred_list.append(y_pred[0])

                        auc = metrics.roc_auc_score(y, y_pred_list)

                        print('PCA components' + str(i),md,ne,lr,rg, round(auc, 2))
        scores = round(auc, 2)
        scores_train = round(np.array(auc_train).mean(), 2)
        train_accuracy.append(scores_train)
        test_accuracy.append(round(scores.mean(), 2))

def pca_graph_pvals_less_than():

    X = preproccessed_data.drop(['MTX', 'Age', 'aGVHD1 ', 'cGVHD ', 'disease'], axis=1)

    y = preproccessed_data['disease']

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
                most_corelated_taxon = sorted_taxon[:round(X_train.shape[1] * 0.1)]
                bact = [i[0] for i in most_corelated_taxon if i[0]!=1]
                new_data = X[bact]

                otu_after_pca, _ = apply_pca(new_data, n_components=n_comp)

                new_data = otu_after_pca.join(preproccessed_data[['MTX', 'disease']], how='inner')

                X_new = new_data.drop(['disease'], axis=1)
                y_new = new_data['disease']
                regex = re.compile(r"\[|\]|<", re.IGNORECASE)
                X_new.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in
                             X_new.columns.values]

                X_train, X_test = X_new.iloc[train_index], X_new.iloc[test_index]
                y_train, y_test = y_new[train_index], y_new[test_index]

                model = XGBClassifier(max_depth=3, n_estimators=250, learning_rate=10 / 100,
                                      # objective='multi:softmax' )
                                      objective='binary:logistic',
                                      scale_pos_weight=(np.sum(y_train == -1) / np.sum(y_train == 1)),
                                      reg_lambda=350)
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
#pca_graph_pvals_less_than()
pca_graph()
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

