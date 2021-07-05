import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit, cross_validate
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import configparser
from sklearn.svm import SVC
import xgboost as xgb

"""
The purpose of this file is to compare between the  performances of the different projections in a classification 
task. 
The model and its parameters are extracted from a configuration file called config.ini"" 
The script plots the performance of each projection for all models based on the evaluation fn inserted.
"""

data_paths_list = [Path('../../data/exported_data/data_for_learning/few_tp/stool_otu.csv'),
                   Path('../../data/exported_data/data_for_learning/few_tp/saliva_otu.csv'),
                   Path('../../data/exported_data/data_for_learning/few_tp/latent_representation_dataset.csv')]
tag_paths_list = [Path('../../data/exported_data/data_for_learning/few_tp/stool_tag.csv'),
                  Path('../../data/exported_data/data_for_learning/few_tp/saliva_tag.csv'),
                  Path('../../data/exported_data/data_for_learning/few_tp/latent_representation_tag.csv')]

legend_list = ['Stool microbiome', 'Saliva microbiome', 'Latent representation']

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
    svm_dict = dict(config['SVM'])
    # the script provides the user the possibility to create multiple svm's with different C parameter
    start, end, jump = map(lambda x: int(x), svm_dict['c'].split(','))
    iterator = list(map(lambda x: x / 10, list(range(start, end, jump))))
    iterator_name = 'C'
    # create the parameters dictionary for the svm constructor.
    param_dict_list = [
        {iterator_name: item, 'kernel': svm_dict.get('kernel', None), 'degree': int(svm_dict.get('degree', 3))} for item
        in iterator]
    model_list = [SVC(**param_dict) for param_dict in param_dict_list]

# If KNN model is requested

elif 'KNN' in config:
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
    xgboost_dict = dict(config['XGBOOST'])
    # create the parameters dictionary for the XGBOOST constructor.

    param_dict = {'learning_rate': float(xgboost_dict['learning_rate']), 'max_depth': int(xgboost_dict['max_depth']),
                  'n_estimators': int(xgboost_dict['n_estimators']), 'scale_pos_weight': 5}
    model_list = [xgb.XGBClassifier(**param_dict)]

# The performance is calculated using cross validation.
cv = 4
validation_results = []
best_models_results = []
evaluation_methods = ['roc_auc', 'f1']
fig, axes = plt.subplots(ncols=len(evaluation_methods))
# iterate over all projections
for dataset, label, legend in zip(data_sets_list, tag_list, legend_list):
    # for each model find the evaluation values of evaluation function
    # The next command is not comprehensible, but it returns a list of all evaluation values for each model.
    validation_results = [[cv_results['test_{method}'.format(method=evaluation_method)].mean() for evaluation_method in
                           evaluation_methods] for cv_results in
                          [cross_validate(model, dataset, label, cv=cv, scoring=evaluation_methods) for model in
                           model_list]]
# Plot the results of the projections, each evaluation in a separate ax.
    for i, ax in enumerate(axes):
        specific_evaluation_to_all_models = [validation_evaluations[i] for validation_evaluations in validation_results]
        ax.plot(range(1,len(model_list)+1), specific_evaluation_to_all_models, label=legend, marker='.', markersize=10)
    validation_results = []
# Set the titles and labels to each ax.
for ax, evaluation_method in zip(axes, evaluation_methods):
    ax.set_xlabel(iterator_name,fontsize=15)
    ax.set_ylabel(evaluation_method,fontsize=15)
    ax.set_title('{evaluation} results of KNN with {cv} fold cross validation'.format(evaluation=evaluation_method,cv=cv),fontsize=15)
    ax.legend(fontsize=15)
plt.show()

"""
    for model in model_list:
        cv_results = cross_validate(model, dataset, label, cv=cv, scoring=evaluation_method)
        validation_results.append(cv_results['test_score'].mean())
    best_result = max(validation_results)
    best_models_results.append(best_result)
    plt.plot(iterator,validation_results, label=legend,marker='.', markersize=10)
    plt.xlabel(iterator_name)
    plt.ylabel(evaluation_method)
    plt.title('Accuracy results of KNN with {cv} fold cross validation'.format(cv=cv))
    print('\n The best result achieved on {legend} is {result} {method} \n'.format(legend=legend, result=best_result,
                                                                                method=evaluation_method))
    validation_results = []
"""
# results_df = pd.DataFrame(data=best_models_results, index=legend_list)

# save the best results for each projection.
# results_df.to_csv('learning_models_results.csv',header=['roc_auc'])
