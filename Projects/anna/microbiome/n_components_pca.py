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
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold,LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics, svm
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

otu = 'C:/Users/Anna/Desktop/docs/otu_psc2.csv'
mapping = 'C:/Users/Anna/Desktop/docs/mapping_psc.csv'
OtuMf = OtuMfHandler(otu, mapping, from_QIIME=False)
preproccessed_data = preprocess_data(OtuMf.otu_file, taxnomy_level=6)
mapping_file = OtuMf.mapping_file

mapping_disease = {'Control':0,'Cirrhosis ':1, 'HCC':1, 'PSC+IBD':2,'PSC':2}
mapping_file['DiagnosisGroup'] = mapping_file['DiagnosisGroup'].map(mapping_disease)
mappin_boolean = {'yes' :1, 'no': 0, 'Control': 0, '0':0, '1':1}
mapping_file['FattyLiver'] = mapping_file['FattyLiver'].map(mappin_boolean)
mapping_file['RegularExercise'] = mapping_file['RegularExercise'].map(mappin_boolean)
mapping_file['Smoking'] = mapping_file['Smoking'].map(mappin_boolean)

train_accuracy = []
test_accuracy = []
pcas =[]
def pca_graph():
    for i in range (10,37):
        pcas.append(i)
        otu_after_pca, _ = apply_pca(preproccessed_data, n_components=i)
        merged_data = otu_after_pca.join(mapping_file[['Age', 'BMI', 'FattyLiver','RegularExercise', 'Smoking','DiagnosisGroup']])

        merged_data.fillna(0)

        # mapping_disease = {'Control':0,'Cirrhosis ':1, 'HCC':1, 'PSC+IBD':2,'PSC':2}
        # merged_data['DiagnosisGroup'] = merged_data['DiagnosisGroup'].map(mapping_disease)
        # merged_data = merged_data.join(OtuMf.mapping_file[['Age', 'BMI', 'FattyLiver','RegularExercise', 'Smoking']])
        # mappin_boolean = {'yes' :1, 'no': 0, 'Control': 0, '0':0, '1':1}
        # merged_data['FattyLiver'] = merged_data['FattyLiver'].map(mappin_boolean)
        # merged_data['RegularExercise'] = merged_data['RegularExercise'].map(mappin_boolean)
        # merged_data['Smoking'] = merged_data['Smoking'].map(mappin_boolean)

        X = merged_data.loc[:, merged_data.columns != 'DiagnosisGroup']
        y = merged_data['DiagnosisGroup']

        loo = LeaveOneOut()
        for md in range(4, 5):
            for ne in range(150, 200, 50):
                for lr in range(15, 20, 5):
                    y_pred_list = []
                    x_indx = []
                    y_pred_train =[]
                    for train_index, test_index in loo.split(X):
                        train_index = list(train_index)
                        # print("%s %s" % (train_index, test_index))
                        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
                        y_train, y_test = y[train_index], y[test_index]
                        model = XGBClassifier(max_depth=md, n_estimators=ne, learning_rate=lr / 100,
                                              objective='multi:softmax')#,  reg_lambda=550)
                        # # #                                           #objective= 'binary:logistic')
                        model.fit(X_train, y_train)
                        x_indx.append(X_test.index[0])
                        y_pred = model.predict(X_test)

                        y_pred_list.append(y_pred[0])
                        y_pre_tr = model.predict(X_train)
                        accuracy_train = metrics.accuracy_score(y_train,y_pre_tr)
                        y_pred_train.append(accuracy_train)

                    scores = np.array(metrics.accuracy_score(y, y_pred_list))
                    scores_train = round(np.array(y_pred_train).mean(), 2)

                    print(md, ne, lr, round(scores.mean(), 2), round(scores.std(), 2) * 2)
        train_accuracy.append(scores_train)
        test_accuracy.append(round(scores.mean(), 2))

def pca_graph_pvals_less_than():

    data = preproccessed_data.join(mapping_file[['Age', 'BMI', 'FattyLiver','RegularExercise', 'Smoking', 'DiagnosisGroup']])
    X = data.drop(['Age', 'BMI', 'FattyLiver','RegularExercise', 'Smoking', 'DiagnosisGroup'], axis=1)

    y = data['DiagnosisGroup']

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
                most_corelated_taxon = [i for i in sorted_taxon if i[1]<=0.01]
                bact = [i[0] for i in most_corelated_taxon if i[0]!=1]
                new_data = X[bact]

                otu_after_pca, _ = apply_pca(new_data, n_components=n_comp)

                new_data = otu_after_pca.join(data[['Age', 'BMI', 'FattyLiver','RegularExercise', 'Smoking', 'DiagnosisGroup']], how='inner')

                X_new = new_data.drop(['DiagnosisGroup'], axis=1)
                y_new = new_data['DiagnosisGroup']
                regex = re.compile(r"\[|\]|<", re.IGNORECASE)
                X_new.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in
                             X_new.columns.values]

                X_train, X_test = X_new.iloc[train_index], X_new.iloc[test_index]
                y_train, y_test = y_new[train_index], y_new[test_index]

                model = XGBClassifier(max_depth=4, n_estimators=150, learning_rate=15 / 100,
                                      objective='multi:softmax' )
                                      #objective='binary:logistic',
                                      #scale_pos_weight=(np.sum(y_train == -1) / np.sum(y_train == 1)))
                model.fit(X_train, y_train)
                pred_train = model.predict(X_train)
                auc_train.append(metrics.accuracy_score(y_train, pred_train))
                y_pred = model.predict(X_test)
                y_pred_list.append(y_pred[0])
            try:
                    auc =metrics.accuracy_score(y, y_pred_list)
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
print('end')