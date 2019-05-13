from load_merge_otu_mf import OtuMfHandler
from preprocess import preprocess_data
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
import csv
import networkx as nx
import scipy
import re
from plot_clustergram import *
csvfile = 'C:/Users/Anna/Documents/xgboost_gvhd_saliva.csv'
otu = 'C:/Users/Anna/Documents/otu_saliva_GVHD.csv'
mapping = 'C:/Users/Anna/Documents/mapping_saliva_GVHD.csv'
headers = ['ms', 'ne', 'learning rate', 'regularization', 'auc test', 'auc train']
# with open(csvfile, "w") as output:
#     writer = csv.writer(output, delimiter=',', lineterminator='\n')
#     writer.writerow(headers)

OtuMf = OtuMfHandler(otu, mapping, from_QIIME=False)
print(OtuMf.otu_file.shape)
preproccessed_data = preprocess_data(OtuMf.otu_file, visualize_data=False, taxnomy_level=6)
print(preproccessed_data.shape)
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
#mapping_file = mapping_file[(mapping_file['DATE']>mapping_file['Date_of_engraftmen']) & (mapping_file['DATE']<mapping_file['aGVHD1_Stat']) & (mapping_file['DATE']<mapping_file['cGVHD_start'])].sort_values(['Personal_ID', 'DATE'])

mapping_file = mapping_file.reset_index()
mapping_file = mapping_file.sort_values("DATE").groupby("Personal_ID", as_index=False).last().set_index('#SampleID')
preproccessed_data = preproccessed_data.join(mapping_file[['MTX', 'Age', 'aGVHD1 ', 'cGVHD ']], how ='inner')
preproccessed_data = preproccessed_data.fillna('No')
#preproc_test = preproccessed_data.drop(['R008W03', 'R119W03', 'R145W03', 'R009W29', 'R072W02', 'R055W16', 'R023W01'])
#print(preproccessed_data.tail())
#preproc_test1 = preproccessed_data.drop(['MTX', 'Age', 'aGVHD1 ', 'cGVHD '], axis=1)
#visualize_pca(preproc_test1)
otu_after_pca, pca_components = apply_pca(preproccessed_data.drop(['MTX', 'Age', 'aGVHD1 ', 'cGVHD '], axis=1) , n_components=20)
merged_data = otu_after_pca.join(preproccessed_data[['MTX', 'Age', 'aGVHD1 ', 'cGVHD ']], how ='inner')
#merged_data = preproccessed_data
mapping_yes_no = {'Yes':1,'No':0}
merged_data['MTX'] = merged_data['MTX'].map(mapping_yes_no)
merged_data['aGVHD1 '] = merged_data['aGVHD1 '].map(mapping_yes_no)
merged_data['cGVHD '] = merged_data['cGVHD '].map(mapping_yes_no)
merged_data["disease"] = merged_data["aGVHD1 "].map(str) + '_' +merged_data["cGVHD "].map(str)
mapping_diseases = {'0_0':1,'1_0':-1,'0_1':-1,'1_1':-1}
merged_data["disease"] = merged_data["disease"].map(mapping_diseases)


X = merged_data.drop(['Age', 'aGVHD1 ', 'cGVHD ', 'disease'], axis=1)
y = merged_data['disease']
loo = LeaveOneOut()
for md in range(2,5):
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
                    print(md, ne, lr, rg, round(auc, 2))
predicted_data = preproccessed_data
predicted_data = predicted_data.drop(['Age', 'aGVHD1 ', 'cGVHD ', 'MTX'],axis =1)
predicted_data['pred'] = np.array(y_pred_list2)
most_corelated_taxon = {}
for i in range(predicted_data.shape[1] - 1):
    if scipy.stats.spearmanr(predicted_data.iloc[:, i], predicted_data['pred'])[1]<0.05:  #/predicted_data.shape[1]:
        most_corelated_taxon[predicted_data.columns[i]] = scipy.stats.spearmanr(predicted_data.iloc[:, i], predicted_data['pred'])[0]

sorted_taxon = sorted(most_corelated_taxon.items(), key=lambda x: abs(x[1]), reverse=True)
most_corelated_taxon = sorted_taxon[:50]

G=nx.Graph()
labeldict = {}
for i in range(len(most_corelated_taxon)):
    G.add_node(i+1, taxonomy = most_corelated_taxon[i][0])
    labeldict[i+1] = most_corelated_taxon[i][0]

for i in range(len(most_corelated_taxon)):
    for j in range(len(most_corelated_taxon)):
        if i!=j:
            if (scipy.stats.spearmanr(predicted_data.loc[:, most_corelated_taxon[i][0]], predicted_data.loc[:,most_corelated_taxon[j][0]])[1]) < 0.001/17 :
                #print(most_corelated_taxon[i][0], most_corelated_taxon[j][0])
                if not G.has_edge(i+1,j+1):
                    G.add_edge(i+1,j+1)

nx.draw(G,  with_labels = True)
print(nx.connected_components(G))
print(nx.degree(G))
print(sorted(i[1] for i in nx.degree(G)))
print(nx.clustering(G))
plt.show()

rel_dict = []
for i in nx.degree(G):
    #if i[1] != 0:
        print(i)
        print(labeldict[i[0]])
        rel_dict.append(labeldict[i[0]])

new_data = preproccessed_data[rel_dict]
visualize_pca(new_data)
otu_after_pca, _ = apply_pca(new_data, n_components=6)

merged_data = otu_after_pca.join(preproccessed_data[['MTX', 'Age', 'aGVHD1 ', 'cGVHD ']], how ='inner')
#merged_data = preproccessed_data
mapping_yes_no = {'Yes':1,'No':0}
merged_data['MTX'] = merged_data['MTX'].map(mapping_yes_no)
merged_data['aGVHD1 '] = merged_data['aGVHD1 '].map(mapping_yes_no)
merged_data['cGVHD '] = merged_data['cGVHD '].map(mapping_yes_no)
merged_data["disease"] = merged_data["aGVHD1 "].map(str) + '_' +merged_data["cGVHD "].map(str)
mapping_diseases = {'0_0':1,'1_0':-1,'0_1':-1,'1_1':-1}
merged_data["disease"] = merged_data["disease"].map(mapping_diseases)


X = merged_data.drop(['Age', 'aGVHD1 ', 'cGVHD ', 'disease'], axis=1)
y = merged_data['disease']
regex = re.compile(r"\[|\]|<", re.IGNORECASE)
X.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X.columns.values]

loo = LeaveOneOut()

for md in range(2,5):
      for ne in range (50,350,50):
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
                    print(md, ne, lr, rg, round(auc, 2))