from load_merge_otu_mf import OtuMfHandler
from Preprocess import preprocess_data
from pca import *
from plot_confusion_matrix import *
from plot_roc_curve import *
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold,LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from xgboost import XGBClassifier

otu = 'C:/Users/Anna/Desktop/docs/otu_psc2.csv'
mapping = 'C:/Users/Anna/Desktop/docs/mapping_psc.csv'
OtuMf = OtuMfHandler(otu, mapping, from_QIIME=False)
preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxnomy_level=6)
#visualize_pca(preproccessed_data)

otu_after_pca = apply_pca(preproccessed_data, n_components=30)
merged_data = otu_after_pca.join(OtuMf.mapping_file['Diagnosis'])
merged_data.fillna(0)


# font = {'size': 22}
# max_pca = 5
# k =1
# for i in range(max_pca):
#     for j in range(max_pca):
#         if j>i:
#                 plt.subplot(2, 5, k)
#                 plt.scatter(x=merged_data[i][(merged_data['DiagnosisGroup']=='Control')],y=merged_data[j][(merged_data['DiagnosisGroup']=='Control')],
#                   marker='.', color='darkturquoise', label = 'Control', lw=0,s=10**2)
#                 plt.scatter(x=merged_data[i][(merged_data['DiagnosisGroup']=='Cirrhosis ')],y=merged_data[j][(merged_data['DiagnosisGroup']=='Cirrhosis ')],
#                   marker='.', color='red', label = 'Cirrhosis', lw=0,s=10**2)
#                 plt.scatter(x=merged_data[i][(merged_data['DiagnosisGroup'] == 'HCC')],y=merged_data[j][(merged_data['DiagnosisGroup'] == 'HCC')],
#                     marker='.', color='orange', label='HCC', lw=0,s=10**2)
#                 plt.scatter(x=merged_data[i][(merged_data['DiagnosisGroup'] == 'PSC')],y=merged_data[j][(merged_data['DiagnosisGroup'] == 'PSC')],
#                     marker='.', color='green', label='PSC', lw=0,s=10**2)
#                 plt.scatter(x=merged_data[i][(merged_data['DiagnosisGroup'] == 'PSC+IBD')],y=merged_data[j][(merged_data['DiagnosisGroup'] == 'PSC+IBD')],
#                     marker='.', color='black', label='PSC+IBD', lw=0,s=10**2)
#
#                 plt.ylabel('PCA %s' %j)
#                 plt.xlabel('PCA %s' %i)
#                 plt.grid(True)
#                 k+=1
# plt.legend( loc=1,ncol=1)
# plt.show()
mapping_disease_for_labels = {'Control':0,'Cirrhosis/+HCC':1, 'HCC':2, 'PSC/+IBD':3}
mapping_disease = {'Control':0,'Cirrhosis ':1,'CirrhosisHCC':1, 'HCC':1 , 'PSC+IBD':2,'PSC':2}
merged_data['Diagnosis'] = merged_data['Diagnosis'].map(mapping_disease)
merged_data = merged_data.join(OtuMf.mapping_file[['Age', 'BMI', 'FattyLiver','RegularExercise', 'Smoking']])
mappin_boolean = {'yes' :1, 'no': 0, 'Control': 0, '0':0, '1':1}
merged_data['FattyLiver'] = merged_data['FattyLiver'].map(mappin_boolean)
merged_data['RegularExercise'] = merged_data['RegularExercise'].map(mappin_boolean)
merged_data['Smoking'] = merged_data['Smoking'].map(mappin_boolean)
#merged_data = merged_data[np.isfinite(merged_data['DiagnosisGroup'])]

#plot ROC curve
# X_train, X_test, y_train, y_test = train_test_split(
#                            merged_data.loc[:, merged_data.columns != 'DiagnosisGroup'], merged_data['DiagnosisGroup'],
#                            test_size=0.25)
# model = XGBClassifier(max_depth=4,n_estimators = 100 ,learning_rate = 10/100,
#                                            objective= 'binary:logistic')
# model.fit(X_train, y_train)
# y_pred = model.predict_proba(X_test)[:,1]
# fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
# plot_roc_curve(fpr,tpr)

#Random Forest
# for j in range(6, 20, 2):
#     for k in range(6, 8, 2):
#         for l in range(100, 500, 50):
#             auc = []
#             accuracy=[]
#             auc_train = []
#             for i in range(0,20):
#                 X_train, X_test, y_train, y_test = train_test_split(
#                     merged_data.loc[:, merged_data.columns != 'DiagnosisGroup'], merged_data['DiagnosisGroup'],
#                     test_size=0.2)
#                 clf = RandomForestClassifier(max_depth=k, min_samples_split=j, n_estimators=l)
#                 clf.fit(X_train, y_train)
#                 y_train = np.array(y_train)
#                 pred_train = clf.predict_proba(X_train)[:,1]
#                 auc_train.append(metrics.roc_auc_score(y_train, pred_train))
#                 y = np.array(y_test)
#
#                 pred = clf.predict_proba(X_test)[:,1]
#
#                 try:
#                     auc.append(metrics.roc_auc_score(y, pred))
#                 except:
#                     continue
#                 accuracy.append(clf.score(X_test,y_test))
#             #if (sum(accuracy)/len(accuracy) > 0.7) and (sum(auc_train) / len(auc_train))<0.99:
#             print(k, j, l, sum(auc) / len(auc), sum(auc_train) / len(auc_train))
#                 #print(k, j, l,sum(accuracy)/len(accuracy) )


X = merged_data.loc[:, merged_data.columns != 'Diagnosis']
y = merged_data['Diagnosis']

loo = LeaveOneOut()
#skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
# XGBoost fit model no training


for md in range(3,4):
      for ne in range (100,150,50):
           for lr in range (10, 15, 5):
               accuracy = []
# # #             auc_train = []
# # #             auc = []
               #for i in range(0,40,2):
               #        X_train, X_test, y_train, y_test = train_test_split(
               #            merged_data.loc[:, merged_data.columns != 'DiagnosisGroup'], merged_data['DiagnosisGroup'],
               #            test_size=0.25, random_state=i)
               y_pred_list = []
               for train_index, test_index in loo.split(X):
                    train_index=list(train_index)
                    #print("%s %s" % (train_index, test_index))
                    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
                    y_train, y_test = y[train_index], y[test_index]
                    model = XGBClassifier(max_depth=md,n_estimators = ne ,learning_rate = lr/100,  objective='multi:softmax' )
# # #                                           #objective= 'binary:logistic')
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
# # #                     #pred_train = model.predict_proba(X_train)[:, 1]
# # #                     #auc_train.append(metrics.roc_auc_score(y_train, pred_train))
# # #                     #y_pred = model.predict_proba(X_test)[:,1]
# # #                     #try:
# # #                     #    auc.append(metrics.roc_auc_score(y_test, y_pred))
# # #                     #except:
# # #                     #    continue
                    y_pred_list.append(y_pred)
                    #accuracy.append(metrics.accuracy_score(y_test,y_pred))
               cnf_matrix = metrics.confusion_matrix(y,y_pred_list)
               class_names = mapping_disease_for_labels.keys()
               # # Plot non-normalized confusion matrix
               plt.figure()
               plot_confusion_matrix(cnf_matrix, classes=class_names,
                                        title='Confusion matrix, without normalization')

               # # Plot normalized confusion matrix
               plt.figure()
               plot_confusion_matrix(cnf_matrix, classes=list(class_names), normalize=True,
                                        title='Normalized confusion matrix')

               plt.show()
               scores = np.array(metrics.accuracy_score(y,y_pred_list))
               print(md, ne, lr,  round(scores.mean(),2), round(scores.std(),2) * 2)
#plot roc_curve #




#
# X_train, X_test, y_train, y_test = train_test_split(
#                           merged_data.loc[:, merged_data.columns != 'DiagnosisGroup'], merged_data['DiagnosisGroup'],
#                           test_size=0.2)
# model = XGBClassifier(max_depth=3,n_estimators = 300 ,learning_rate = 15/100,  objective='multi:softmax' ,random_state=2 )
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
# accuracy = metrics.accuracy_score(y_test,y_pred)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))
# print(cnf_matrix)
# class_names = mapping_disease_for_labels.keys()
# # # # Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names,
#                         title='Confusion matrix, without normalization')
#
# # # # Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=list(class_names), normalize=True,
#                         title='Normalized confusion matrix')
#
# plt.show()
