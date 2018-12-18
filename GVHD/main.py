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
from plot_clustergram import *

otu = 'C:/Users/Anna/Documents/otu_saliva_GVHD.csv'
mapping = 'C:/Users/Anna/Documents/mapping_saliva_GVHD.csv'
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
mapping_file = mapping_file.sort_values("DATE").groupby("Personal_ID", as_index=False).first().set_index('#SampleID')
preproccessed_data = preproccessed_data.join(mapping_file[['MTX', 'Age', 'aGVHD1 ', 'cGVHD ']], how ='inner')
preproccessed_data = preproccessed_data.fillna('no')
print(preproccessed_data.tail())
preproccessed_data = preproccessed_data.drop(['MTX', 'Age', 'aGVHD1 ', 'cGVHD '], axis=1)
visualize_pca(preproccessed_data)
