import sys
import os
import nni
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import roc_curve, auc
import numpy as np
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Plot import calc_auc_on_flat_results
from LearningMethods.create_otu_and_mapping_files import CreateOtuAndMappingFiles


def print_auc_for_iter(y_test, y_score):
    fpr, tpr, thresholds = roc_curve(np.array(y_test), np.array(y_score))
    roc_auc = auc(fpr, tpr)
    print('ROC AUC = ' + str(round(roc_auc, 4)))

def main_preprocess(tax, site, trimester, preprocess_prms):
    main_task = 'GDM_taxonomy_level_' + str(tax)+'_'+site+'_trimester_'+trimester
    bactria_as_feature_file = 'GDM_OTU_rmv_dup.csv'

    samples_data_file = 'GDM_tag_rmv_dup_' +trimester + '_' + site +'.csv'
    rhos_folder = os.path.join('pregnancy_diabetes_'+trimester+'_'+site, 'rhos')
    pca_folder = os.path.join('pregnancy_diabetes_'+trimester+'_'+site, 'pca')

    mapping_file = CreateOtuAndMappingFiles(bactria_as_feature_file, samples_data_file)
    mapping_file.preprocess(preprocess_params=preprocess_prms, visualize=True)
    mapping_file.rhos_and_pca_calculation(main_task, preprocess_prms['taxonomy_level'], preprocess_prms['pca'],
                                         rhos_folder, pca_folder)
    otu_path, mapping_path, pca_path = mapping_file.csv_to_learn(main_task, os.path.join(os.getcwd(), 'pregnancy_diabetes_'+trimester+'_'+site), tax)
    return otu_path, mapping_path, pca_path


def learn(X_trains, X_tests,y_trains ,y_tests, k_fold, clf_params):
    #create classifier
    clf = XGBClassifier(max_depth=int(clf_params['max_depth']), learning_rate=clf_params['lr'],
              n_estimators=int(clf_params['estimators']), objective='binary:logistic',
              gamma=clf_params['gamma'], min_child_weight=int(clf_params['min_child_weight']), 
              reg_lambda=clf_params['lambda'], booster='gbtree', alpha=clf_params['alpha'])
    y_train_scores, y_test_scores, y_train_preds, y_test_preds = [], [], [], []
    for i in range(k_fold):
        print('------------------------------\niteration number ' + str(i))    
        X_train, X_test, y_train, y_test = X_trains[i], X_tests[i], y_trains[i], y_tests[i]
        clf.fit(X_train, y_train)
        clf.predict_proba(X_test)
        y_score = clf.predict_proba(X_test)
        y_pred = clf.predict(X_test)
        y_test_preds.append(y_pred)
        y_test_scores.append(y_score[:, 1])
        train_pred = clf.predict(X_train)
        train_score = clf.predict_proba(X_train)
        y_train_preds.append(train_pred)
        y_train_scores.append(train_score[:, 1])

        print_auc_for_iter(np.array(y_tests[i]['Tag'].values), np.array(y_score).T[1])

    all_y_train = []
    for i in range(k_fold):
        all_y_train.append(y_trains[i]['Tag'].values)
    all_y_train = np.array(all_y_train).flatten()

    all_y_test = []
    for i in range(k_fold):
        all_y_test.append(y_tests[i]['Tag'].values)
    all_y_test = np.array(all_y_test).flatten()

    y_train_scores = np.array(y_train_scores).flatten()
    y_test_scores = np.array(y_test_scores).flatten()

    #calc AUC on validation set
    _, test_auc, _, _ = calc_auc_on_flat_results(all_y_train, y_train_scores, all_y_test, y_test_scores)
    return test_auc


def split(X, y, test_size,k_fold):
    X_trains, X_tests, y_trains, y_tests= [], [], [], []
    for i in range(k_fold):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, stratify=y)
        X_trains.append(X_train)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_tests.append(y_test)
    return  X_trains, X_tests, y_trains, y_tests

def read_otu_and_mapping_files(otu_path, mapping_path):
    otu_file = pd.read_csv(otu_path)
    mapping_file = pd.read_csv(mapping_path)
    X = otu_file.set_index("ID")
    X = X.rename(columns={x: y for x, y in zip(X.columns, range(0, len(X.columns)))})
    y = mapping_file.set_index("ID")
    return X, y



if __name__ == '__main__':
    clf_params = {'site': 'STOOL', 'taxonomy': 5, 'lr': 0.2548340206430038, 'estimators': 280, 'max_depth': 6, 'min_child_weight': 1, 'gamma': 1.8799612115510362, 'lambda': 2.6763925297462245, 'alpha':4.510310540576153}
    #clf_params = nni.get_next_parameter()
    trimester = 'T1'
    site = 'STOOL'
    tax = 5
    k_fold = 15
    # parameters for Preprocess
    preprocess_prms = {'taxonomy_level': tax, 'taxnomy_group': 'mean', 'epsilon': 0.1,
                       'normalization': 'log', 'z_scoring': 'row', 'norm_after_rel': '',
                       'std_to_delete': 0.2, 'pca': 0}
    #create X,Y
    otu_path, mapping_path, _ = main_preprocess(tax, site, trimester, preprocess_prms)
    X, y = read_otu_and_mapping_files(otu_path, mapping_path)
    #split data
    X_trains, X_tests, y_trains, y_tests = split(X, y, 0.2, k_fold)
    #nni run
    auc = learn(X_trains, X_tests,y_trains ,y_tests, k_fold, clf_params)
    print(auc)
    #nni.report_final_result(auc)