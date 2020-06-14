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
from sklearn import metrics, svm
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from plot_clustergram import *
import csv

# csvfile = 'C:/Users/Anna/Documents/xgboost_gvhd_fecal.csv'
# headers = ['ms', 'ne', 'learning rate', 'regularization', 'auc test', 'auc train']
# with open(csvfile, "w") as output:
#     writer = csv.writer(output, delimiter=',', lineterminator='\n')
#     writer.writerow(headers)

otu = 'C:/Users/Anna/Documents/otu_fecal_GVHD.csv'
mapping = 'C:/Users/Anna/Documents/mapping_fecal_GVHD.csv'
OtuMf = OtuMfHandler(otu, mapping, from_QIIME=False)
print(OtuMf.otu_file.shape)
preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxnomy_level=6)
print(preproccessed_data.shape)
mapping_file = OtuMf.mapping_file
mapping_file['DATE'] = pd.to_datetime(OtuMf.mapping_file['DATE'])
mapping_file['Date_Of_Transplantation'] = pd.to_datetime(OtuMf.mapping_file['Date_Of_Transplantation'])
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
#preproc_test1 = preproccessed_data.drop(['MTX', 'Age', 'aGVHD1 ', 'cGVHD '], axis=1)
#visualize_pca(preproc_test1)

otu_after_pca, pca_components = apply_pca(preproccessed_data.drop(['MTX', 'Age', 'aGVHD1 ', 'cGVHD '], axis=1) , n_components=24)
merged_data = otu_after_pca.join(preproccessed_data[['MTX', 'Age', 'aGVHD1 ', 'cGVHD ']], how ='inner')
#merged_data = preproccessed_data
mapping_yes_no = {'Yes':1,'No':0}
merged_data['MTX'] = merged_data['MTX'].map(mapping_yes_no)
merged_data['aGVHD1 '] = merged_data['aGVHD1 '].map(mapping_yes_no)
merged_data['cGVHD '] = merged_data['cGVHD '].map(mapping_yes_no)
merged_data["disease"] = merged_data["aGVHD1 "].map(str) + '_' +merged_data["cGVHD "].map(str)
mapping_diseases = {'0_0':1,'1_0':0,'0_1':0,'1_1':0}
merged_data["disease"] = merged_data["disease"].map(mapping_diseases)
print(merged_data.shape)
#
# font = {'size': 22}
# max_pca = 5
# k =1
# for i in range(max_pca):
#      for j in range(max_pca):
#         if j>i:
#             plt.subplot(2, 5, k)
#             plt.scatter(x=merged_data[i][(merged_data['disease']==0)],y=merged_data[j][(merged_data['disease']==0)],
#                    marker='.', color='darkturquoise', label = 'control', lw=0,s=10**2)
#             plt.scatter(x=merged_data[i][(merged_data['disease']==1)],y=merged_data[j][(merged_data['disease']==1)],
#                    marker='.', color='orange', label = 'GVHD acute', lw=0,s=10**2)
#             plt.scatter(x=merged_data[i][(merged_data['disease'] == 2)],y=merged_data[j][(merged_data['disease'] == 2)],
#                      marker='.', color='red', label='GVHD chronic', lw=0,s=10**2)
#             plt.scatter(x=merged_data[i][(merged_data['disease'] == 3)],y=merged_data[j][(merged_data['disease'] == 3)],
#                         marker='.', color='brown', label='GVHD both', lw=0, s=10 ** 2)
#             plt.ylabel('PCA %s' %j)
#             plt.xlabel('PCA %s' %i)
#             plt.grid(True)
#             plt.ylim(-10,10)
#             plt.xlim(-10,10)
#             k+=1
# #plt.title('mean reduced')
# plt.legend( loc=1,ncol=1)
# plt.show()
#
# print('done')

X = merged_data.drop(['Age', 'aGVHD1 ', 'cGVHD ', 'disease'], axis=1)
y = merged_data['disease']
loo = LeaveOneOut()
# for md in range(2,6):
#       for ne in range (50,350,50):
#            for lr in range (5, 20, 5):
#                for rg in range(250, 400, 25):
#                    accuracy = []
#                    y_pred_list = []
#                    auc = []
#                    auc_train = []
#                    for train_index, test_index in loo.split(X):
#                         train_index=list(train_index)
#                         #print("%s %s" % (train_index, test_index))
#                         X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
#                         y_train, y_test = y[train_index], y[test_index]
#                         model = XGBClassifier(max_depth=md,n_estimators = ne ,learning_rate = lr/100,  #objective='multi:softmax' )
#                                               objective= 'binary:logistic', scale_pos_weight = (np.sum(y_train==-1)/np.sum(y_train==1)),
#                                               reg_lambda = rg)
#                         model.fit(X_train, y_train)
#                         y_pred = model.predict(X_test)
#                         pred_train = model.predict_proba(X_train)[:, 1]
#                         auc_train.append(metrics.roc_auc_score(y_train, pred_train))
#                         #y_pred = model.predict_proba(X_test)[:,1]
#                         y_pred_list.append(y_pred)
#                    try:
#                        auc = metrics.roc_auc_score(y, y_pred_list)
#                    except:
#                        pass
#                    with open(csvfile, "a") as output:
#                        writer = csv.writer(output, delimiter=',', lineterminator='\n')
#                        writer.writerow([md, ne, lr, rg, round(auc, 2), round(sum(auc_train) / len(auc_train), 2)])
#                    print(md, ne, lr, round(auc, 2), round(sum(auc_train) / len(auc_train), 2))
#
clf = svm.SVC(kernel='linear', probability = True)
#clf =  LogisticRegression(penalty='l1', solver='liblinear', C= 0.55 )
auc = []
y_pred1 =[]
y_test1 = []
y_pred_train =[]
for train_index, test_index in loo.split(X):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict_proba(X_train)[:,1]
    y_pred = clf.predict_proba(X_test)[:,1]
    y_pred1.append(y_pred)
    y_test1.append(y_test.values[0])
    print(metrics.roc_auc_score(y_train,y_pred_train))
print(y_pred1)
print(y_test1)
print(metrics.roc_auc_score(y_test1, y_pred1))