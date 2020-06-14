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


def build_score_df(X_train, score_list):
    idx_list = [id for id in X_train.index.values]
    df = pd.DataFrame.from_dict({'ID':idx_list, 'Score':score_list})
    df =df.set_index('ID')
    return df


def learn_XGBOOST(X_trains, X_tests,y_trains ,y_tests, k_fold, clf_params, clf_ens_params, df_concat):

    #create classifier
    clf_score = XGBClassifier(max_depth=int(clf_params['max_depth']), learning_rate=clf_params['lr'],
                        n_estimators=int(clf_params['estimators']), objective='binary:logistic',
                        gamma=clf_params['gamma'], min_child_weight=int(clf_params['min_child_weight']),
                        reg_lambda=clf_params['lambda'], booster='dart', alpha=clf_params['alpha'])

    ens_clf = XGBClassifier(max_depth=int(clf_ens_params['max_depth']), learning_rate=clf_ens_params['lr'],
                        n_estimators=int(clf_ens_params['estimators']), objective='binary:logistic',
                        gamma=clf_ens_params['gamma'], min_child_weight=int(clf_ens_params['min_child_weight']),
                        reg_lambda=clf_ens_params['lambda'], booster='dart', alpha=clf_ens_params['alpha'])

    y_train_scores, y_test_scores, y_train_scores_ens, y_test_scores_ens = [], [], [], []
    all_y_train_ens, all_y_test_ens = [], []

    for i in range(k_fold):
        print('------------------------------\niteration number ' + str(i))
        X_train, X_test, y_train, y_test = X_trains[i], X_tests[i], y_trains[i], y_tests[i]
        #train XGBOOST model
        clf_score.fit(X_train, y_train)
        clf_score.predict_proba(X_test)
        y_score = clf_score.predict_proba(X_test)
        y_test_scores.append(y_score[:, 1])
        train_score = clf_score.predict_proba(X_train)
        y_train_scores.append(train_score[:, 1])

        #building new data frame for learning all model
        score_train_df = build_score_df(X_train, train_score[:, 1])
        X_train_ens, y_train_ens = create_concate_df_to_learn(score_train_df, df_concat)
        all_y_train_ens.append(y_train_ens.values)
        score_test_df = build_score_df(X_test, y_score[:, 1])
        X_test_ens, y_test_ens = create_concate_df_to_learn(score_test_df, df_concat)
        all_y_test_ens.append(y_test_ens.values)

        #train all model using score predictions of XGBOOST
        ens_clf.fit(X_train_ens, y_train_ens)
        ens_clf.predict_proba(X_test_ens)
        y_score_ens = ens_clf.predict_proba(X_test_ens)
        y_test_scores_ens.append(y_score_ens[:, 1])
        train_score_ens = ens_clf.predict_proba(X_train_ens)
        y_train_scores_ens.append(train_score_ens[:, 1])

        #print auc of each fold
        print('iner model')
        print_auc_for_iter(np.array(y_test), np.array(y_score).T[1])
        print('ensemble model')
        print_auc_for_iter(np.array(y_test_ens), np.array(y_score_ens).T[1])

    # calc AUC on validation set_inner model
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
    print('Inner Model')
    _, test_auc, _, _ = calc_auc_on_flat_results(all_y_train, y_train_scores, all_y_test, y_test_scores)

    # calc AUC on validation set ensemble model

    all_y_train_ens = np.array(all_y_train_ens).flatten()
    all_y_test_ens = np.array(all_y_test_ens).flatten()
    y_train_scores = np.array(y_train_scores_ens).flatten()
    y_test_scores = np.array(y_test_scores_ens).flatten()

    print('Ensemble Model')
    _, test_auc, _, _ = calc_auc_on_flat_results(all_y_train_ens, y_train_scores, all_y_test_ens, y_test_scores)
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


def create_concate_df_to_learn(score_df, df_concat, i):
    idx_list_otu = score_df.index.values.tolist()
    idx_list_extra = df_concat.index.values.tolist()
    dict_index_to_full_index = {full.split('-')[0]:full for full in idx_list_otu}
    otu_short_idx = dict_index_to_full_index.keys()
    intersection_idx = set(idx_list_extra).intersection(set(otu_short_idx))
    idx_to_drop = [dict_index_to_full_index[short_name] for short_name in set(otu_short_idx) - intersection_idx]
    score_df = score_df.drop(index=idx_to_drop)
    df_concat = df_concat.drop(index=list(set(idx_list_extra) - intersection_idx))
    idx_list_extra = df_concat.index.values.tolist()
    for short_name in idx_list_extra:
        idx = idx_list_extra.index(short_name)
        idx_list_extra[idx] = dict_index_to_full_index[short_name]
    df_concat.index = idx_list_extra
    result = pd.concat([score_df, df_concat], axis=1)
    if i==1:
      result.to_csv('check_csv_concatenate_train.csv')
    if i==2:
      result.to_csv('check_csv_concatenate_val.csv')
    X = result.iloc[:, :-1]
    y = result.iloc[:, -1]
    return X, y


if __name__ == '__main__':
    clf_params = {'site': 'STOOL', 'taxonomy': 5, 'lr': 0.34274808130945006, 'estimators': 180, 'max_depth': 10, 'min_child_weight': 2,
                  'gamma': 0.113898163565056, 'lambda': 3.102658911459862, 'alpha':0.34196194833422705}
    #clf_ens_params = {'lr': 0.1, 'estimators': 150, 'max_depth': 4, 'min_child_weight': 2,
    #              'gamma': 0.5, 'lambda': 0.1}
    clf_ens_params = nni.get_next_parameter()
    trimester = 'T1'
    site = str(clf_params['site'])
    tax = int(clf_params['taxonomy'])
    # parameters for Preprocess
    preprocess_prms = {'taxonomy_level': tax, 'taxnomy_group': 'mean', 'epsilon': 0.1,
                       'normalization': 'log', 'z_scoring': 'row', 'norm_after_rel': '',
                       'std_to_delete': 0.2, 'pca': 0}
    #read extra features data frame
    df_concat = pd.read_csv('DB/metadata.csv')
    df_concat = df_concat.set_index('ID')
    # create X,Y
    otu_path, mapping_path, _ = main_preprocess(tax, site, trimester, preprocess_prms)
    X, y = read_otu_and_mapping_files(otu_path, mapping_path)
    # split data
    X_trains, X_tests, y_trains, y_tests = split(X, y, 0.2, 5)
    # nni run
    auc = learn_XGBOOST(X_trains, X_tests, y_trains, y_tests, 5, clf_params, clf_ens_params, df_concat)
    nni.report_final_result(auc)