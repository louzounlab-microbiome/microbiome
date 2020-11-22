import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit, cross_validate
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import configparser
from sklearn.svm import SVC
import xgboost as xgb

data_paths_list = [Path('../../data/exported_data/stool_dataset.csv'),
                   Path('../../data/exported_data/saliva_dataset.csv'),
                   Path('../../data/exported_data/latent_representation_dataset.csv')]
tag_paths_list = [Path('../../data/exported_data/stool_tag.csv'), Path('../../data/exported_data/saliva_tag.csv'),
                  Path('../../data/exported_data/latent_representation_tag.csv')]

legend_list = ['Stool microbiome', 'Saliva microbiome', 'Latent representation']

data_sets_list = []
tag_list = []

for data_set_path, tag_path in zip(data_paths_list, tag_paths_list):
    dataset=pd.read_csv(data_set_path, index_col=0)
    dataset.columns=range(0,len(dataset.columns))
    data_sets_list.append(dataset)
    tag_list.append(pd.read_csv(tag_path, index_col=0)['Tag'])

config = configparser.ConfigParser()
config.read('config.ini')
if 'SVM' in config:
    svm_dict = dict(config['SVM'])
    start, end, jump = map(lambda x: int(x), svm_dict['c'].split(','))
    iterator = list(map(lambda x: x / 10, list(range(start, end, jump))))
    iterator_name = 'C'
    param_dict_list = [
        {iterator_name: item, 'kernel': svm_dict.get('kernel', None), 'degree': int(svm_dict.get('degree', 3))} for item
        in iterator]
    model_list = [SVC(**param_dict) for param_dict in param_dict_list]


elif 'KNN' in config:
    knn_dict = dict(config['KNN'])
    start, end, jump = map(lambda x: int(x), knn_dict['n_neighbors'].split(','))
    iterator = range(start, end, jump)
    iterator_name = 'n_neighbors'
    param_dict_list = [{iterator_name: item} for item
                       in iterator]

    model_list = [KNeighborsClassifier(**param_dict) for param_dict in param_dict_list]

elif 'XGBOOST' in config:
    xgboost_dict = dict(config['XGBOOST'])
    param_dict = {'learning_rate': float(xgboost_dict['learning_rate']), 'max_depth': int(xgboost_dict['max_depth']),
                  'n_estimators': int(xgboost_dict['n_estimators']),'scale_pos_weight':5}
    model_list = [xgb.XGBClassifier(**param_dict)]
cv = 5
validation_results = []
for dataset, label, legend in zip(data_sets_list, tag_list, legend_list):
    for model in model_list:
        cv_results = cross_validate(model, dataset, label, cv=cv,scoring='f1_weighted')
        validation_results.append(cv_results['test_score'].mean())
    print('\n The best result achieved on {legend} is {result} \n'.format(legend=legend, result=max(validation_results)))
    validation_results = []
