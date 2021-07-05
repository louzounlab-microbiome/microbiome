import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit, cross_validate
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import configparser
from sklearn.svm import SVC
import xgboost as xgb
import math

"""
The purpose of this file is to compare between the  performances of the different projections in a classification 
task. 
The model and its parameters are extracted from a configuration file called config.ini"" 
The script plots the performance of each projection for all models based on the evaluation fn inserted.
"""

data_paths_list = [Path('../../data/exported_data/data_for_learning/GAN_integration/otu_dataset.csv'),
                   Path('../../data/exported_data/data_for_learning/GAN_integration/metabolomics_dataset.csv'),
                   Path('../../data/exported_data/data_for_learning/GAN_integration/latent_representation_dataset.csv')]
tag_paths_list = [Path('../../data/exported_data/data_for_learning/GAN_integration/otu_tag.csv'),
                  Path('../../data/exported_data/data_for_learning/GAN_integration/metabolomics_tag.csv'),
                  Path('../../data/exported_data/data_for_learning/GAN_integration/latent_representation_tag.csv')]

legend_list = ['Otu', 'metabolomics', 'Latent representation']

data_sets_list = []
tag_list = []
# Load the datasets and their labels according to the paths supplied above
for data_set_path, tag_path in zip(data_paths_list, tag_paths_list):
    dataset = pd.read_csv(data_set_path, index_col=0)
    dataset.columns = range(0, len(dataset.columns))
    data_sets_list.append(dataset)
    tag_list.append(pd.read_csv(tag_path, index_col=0)['Tag'])
# read the parameters from the configuration file.
config = configparser.ConfigParser()
config.read('config.ini')
# If SVM model is requested
if 'SVM' in config:
    model_name ='SVM'
    svm_dict = dict(config['SVM'])
    # the script provides the user the possibility to create multiple svm's with different C parameter
    start, end, factor = list(map(lambda x: float(x), svm_dict['c'].split(',')))
    iterator = [start*factor**i for i in range(0,int(math.log((end/start),factor)+1))]
    iterator_name = 'C'
    # create the parameters dictionary for the svm constructor.
    param_dict_list = [
        {iterator_name: item, 'kernel': svm_dict.get('kernel', None), 'degree': int(svm_dict.get('degree', 3))} for item
        in iterator]
    model_list = [SVC(**param_dict) for param_dict in param_dict_list]

# If KNN model is requested

elif 'KNN' in config:
    model_name = 'KNN'
    knn_dict = dict(config['KNN'])
    # the script provides the user the possibility to create multiple svm's with different n_neighbors parameter
    start, end, jump = map(lambda x: int(x), knn_dict['n_neighbors'].split(','))
    iterator = range(start, end, jump)
    iterator_name = 'n_neighbors'
    # create the parameters dictionary for the svm constructor.

    param_dict_list = [{iterator_name: item} for item
                       in iterator]

    model_list = [KNeighborsClassifier(**param_dict) for param_dict in param_dict_list]

# If XGBOOST model is requested

elif 'XGBOOST' in config:
    model_name = 'XGBOOST'
    xgboost_dict = dict(config['XGBOOST'])
    # create the parameters dictionary for the XGBOOST constructor.

    param_dict = {'learning_rate': float(xgboost_dict['learning_rate']), 'max_depth': int(xgboost_dict['max_depth']),
                  'n_estimators': int(xgboost_dict['n_estimators']), 'scale_pos_weight': 5}
    model_list = [xgb.XGBClassifier(**param_dict)]
    iterator_name='No iterator'
# The performance is calculated using cross validation.
cv = 5
validation_results = []
best_models_results = []
evaluation_methods = ['accuracy', 'f1']
fig, axes = plt.subplots(ncols=len(evaluation_methods))
# iterate over all projections
for dataset, label, legend in zip(data_sets_list, tag_list, legend_list):
    # for each model find the evaluation values of the evaluation functions
    # The next command is not comprehensible, but it returns a list of all evaluation values for each model.
    validation_results = [[cv_results['test_{method}'.format(method=evaluation_method)].mean() for evaluation_method in
                           evaluation_methods] for cv_results in
                          [cross_validate(model, dataset, label, cv=cv, scoring=evaluation_methods) for model in
                           model_list]]

# Plot the results of the projections, each evaluation in a separate ax.
    for i, ax in enumerate(axes):
        specific_evaluation_to_all_models = [validation_evaluations[i] for validation_evaluations in validation_results]
        ax.plot(iterator, specific_evaluation_to_all_models, label=legend, marker='.', markersize=10)
    validation_results = []
# Set the titles and labels to each ax.
for ax, evaluation_method in zip(axes, evaluation_methods):
    ax.set_xlabel(iterator_name,fontsize=15)
    ax.set_ylabel(evaluation_method,fontsize=15)
    ax.set_title('{evaluation} results of {model} with {cv} fold cross validation'.format(evaluation=evaluation_method,model=model_name,cv=cv),fontsize=15)
    ax.legend(fontsize=15)
plt.show()
