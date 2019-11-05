import math
import os
from pathlib import Path

import pandas as pd
import numpy as np
import pickle
from sklearn import svm
from sklearn.metrics import classification_report, roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from dafna.abstract_data_set import AbstractDataLoader
from dafna.nn import nn_main
from dafna.plot_3D_pca import plot_data_3d, plot_data_2d, PCA_t_test
from infra_functions.general import draw_rhos_calculation_figure
from dafna.plot_auc import roc_auc, calc_auc_on_joined_results, multi_class_roc_auc
from dafna.plot_coef import create_coeff_plots_by_alogorithm
from dafna.plot_confusion_mat import edit_confusion_matrix, print_confusion_matrix
from infra_functions.general import convert_pca_back_orig

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
n_components = 20

# ----------------------------------------------! learning methods ------------------------------------------------


def svm_calc_bacteria_coeff_average(data_loader, clf, svm_coefs, bacteria_coeff_average):
    preproccessed_data = data_loader.get_preproccessed_data
    pca_obj = data_loader.get_pca_obj
    num_of_classes = clf.coef_.shape[0]
    if num_of_classes > 1:  # multi-class
        c = clf.coef_.tolist()
        svm_coefs.append(c)
        bacterias = convert_pca_back_orig(pca_obj.components_, c[0],
                                          original_names=preproccessed_data.columns[:],
                                          visualize=False)['Taxonome'].tolist()

        coefficients = np.array([convert_pca_back_orig(pca_obj.components_, c[i],
                                                       original_names=preproccessed_data.columns[:],
                                                       visualize=False)['Coefficients'].tolist() for i in
                                 range(num_of_classes)])
        bacteria_coeff_average.append(coefficients)

    else:  # binary
        c = clf.coef_.tolist()[0]
        svm_coefs.append(c)
        bacteria_coeff = convert_pca_back_orig(pca_obj.components_, c,
                                               original_names=preproccessed_data.columns[:],
                                               visualize=False)

        bacterias = bacteria_coeff['Taxonome'].tolist()
        coefficients = bacteria_coeff['Coefficients'].tolist()
        bacteria_coeff_average.append(coefficients)
    return svm_coefs, bacterias, coefficients, bacteria_coeff_average

def xgb_calc_bacteria_coeff_average(data_loader, clf, xgb_coefs, bacteria_coeff_average):
    preproccessed_data = data_loader.get_preproccessed_data
    pca_obj = data_loader.get_pca_obj
    num_of_classes = clf.coef_.shape[0]
    if num_of_classes > 1:  # multi-class

        c = np.array(clf.coef_)
        xgb_coefs.append(c)
        bacterias = convert_pca_back_orig(pca_obj.components_, c[0],
                                          original_names=preproccessed_data.columns[:],
                                          visualize=False)['Taxonome'].tolist()

        coefficients = np.array([convert_pca_back_orig(pca_obj.components_, c[i],
                                                       original_names=preproccessed_data.columns[:],
                                                       visualize=False)['Coefficients'].tolist() for i in
                                 range(num_of_classes)])
        bacteria_coeff_average.append(coefficients)

    else:  # binary
        c = clf.coef_.tolist()[0]
        xgb_coefs.append(c)
        bacteria_coeff = convert_pca_back_orig(pca_obj.components_, c,
                                               original_names=preproccessed_data.columns[:],
                                               visualize=False)

        bacterias = bacteria_coeff['Taxonome'].tolist()
        coefficients = bacteria_coeff['Coefficients'].tolist()
        bacteria_coeff_average.append(coefficients)

def plot_bacteria_coeff_average(bacteria_coeff_average, num_of_classes, data_loader, title, task_name, bacterias,
                                cross_validation, algorithm, clf_folder_name, BINARY):
    bacteria_coeff_average = np.array(bacteria_coeff_average)

    if BINARY:  # binary
        avg_df = pd.DataFrame(bacteria_coeff_average)
        avg_cols = [x for x in avg_df.mean(axis=0)]
        create_coeff_plots_by_alogorithm(avg_cols, bacterias, task_name, algorithm, cross_validation,
                                         folder=clf_folder_name)
    else:
        for i in range(num_of_classes):
            avg_df = pd.DataFrame(bacteria_coeff_average[i])
            avg_cols = [x for x in avg_df.mean(axis=0)]
            names = data_loader.get_confusin_matrix_names()
            task_name_and_class = task_name.replace("_", " ") + " - " + names[i].replace("_", " ") + " class"
            create_coeff_plots_by_alogorithm(avg_cols, bacterias, task_name_and_class, algorithm, cross_validation,
                                             folder=clf_folder_name)


def save_results(task_name, train_auc, test_auc, train_rho, test_rho, confusion_matrix_average, confusion_matrix_acc,
                 train_accuracies, test_accuracies, y_score_from_all_iter, y_pred_from_all_iter,
                 y_test_from_all_iter, algorithm, clf_folder_name):
    print(algorithm + "final results: " + task_name + "\n" + "train_auc: " + str(train_auc) + "\n" + "test_auc: " +
          str(test_auc) + "\n" + "train_rho: " + str(train_rho) + "\n" + "test_rho: " + str(test_rho) + "\n" +
          "confusion_matrix_average: " + "\n")
    print(confusion_matrix_average)
    print("confusion_matrix_acc: " + "\n" + str(confusion_matrix_acc) + "\n")

    confusion_matrix_average.to_csv(os.path.join(clf_folder_name, algorithm + "_confusion_matrix_average_on_" + task_name + ".txt"))

    with open(os.path.join(clf_folder_name, task_name + "_" + algorithm + "_AUC_on_" + task_name + ".txt"), "w") as file:
        file.write("train_auc: " + str(train_auc) + "\n")
        file.write("test_auc: " + str(test_auc) + "\n")
        file.write("train_rho: " + str(train_rho) + "\n")
        file.write("test_rho: " + str(test_rho) + "\n")
        file.write("\n")
        file.write("train accuracy:\n")
        for a in train_accuracies:
            file.write(str(a) + "\n")
        file.write("\n")
        file.write("test accuracy:\n")
        for a in test_accuracies:
            file.write(str(a) + "\n")

    # save results to data frame
    if len(y_score_from_all_iter.shape) > 1:
        score_map = {"y score " + str(i): y_score_from_all_iter[:, i] for i in
                     range(y_score_from_all_iter.shape[1])}
    else:
        score_map = {"y score": y_score_from_all_iter}

    # score_map = {"y score " + str(i): svm_y_score_from_all_iter[:, i] for i in range(svm_y_score_from_all_iter.shape[1])}
    score_map["y pred"] = y_pred_from_all_iter.astype(int)
    score_map["y test"] = y_test_from_all_iter.astype(int)
    results = pd.DataFrame(score_map)

    results.append([np.mean(train_accuracies), np.mean(test_accuracies), None, None, None, None])
    # pickle.dump(results, open(os.path.join(clf_folder_name, task_name + "_" + algorithm + "_clf_results_" + task_name + ".pkl"), 'wb'))
    results.to_csv(os.path.join(clf_folder_name, task_name + "_" + algorithm + "_clf_results_" + task_name + ".csv"))
    # pickle.dump(confusion_matrix_average,
    #             open(os.path.join(clf_folder_name, task_name + "_" + algorithm + "_clf_confusion_matrix_results_" + task_name + ".pkl"), 'wb'))
    confusion_matrix_average.to_csv(os.path.join(clf_folder_name, task_name + "_" + algorithm +
                                                 "_clf_confusion_matrix_results_" + task_name + ".csv"))


def learn(title, data_loader, allow_printing, calculate_rhos, SVM, XGBOOST, NN, cross_validation,
          create_coeff_plots, check_all_parameters, svm_parameters, xgb_parameters, create_pca_plots, test_size=0.1, BINARY=True):
    # create a folder for the task
    if not os.path.exists(title):
        os.makedirs(title)
    os.chdir(os.path.join(os.path.abspath(os.path.curdir), title))

    print("learning..." + title)
    ids, tag_map, task_name = data_loader.get_learning_data(title)
    id_to_features_map = data_loader.get_id_to_features_map
    X = [id_to_features_map[id] for id in ids if id in id_to_features_map.keys()]
    y = [tag_map[id] for id in ids if id in id_to_features_map.keys()]

    # ----------------------------------------------! calculate_rhos ------------------------------------------------
    if calculate_rhos:
        print("calculating rho")
        draw_rhos_calculation_figure(tag_map, data_loader.get_preproccessed_data, title, ids_list=ids,
                                     save_folder="rhos")

    # ----------------------------------------------! PCA ------------------------------------------------
    if create_pca_plots:
        PCA_t_test(group_1=[x for x, y in zip(X, y) if y == 0], group_2=[x for x, y in zip(X, y) if y == 1],
                   title="T test for PCA dimentions on " + task_name, save=True, folder="PCA")
        plot_data_3d(X, y, data_name=task_name.capitalize(), save=True, folder="PCA")
        plot_data_2d(X, y, data_name=task_name.capitalize(), save=True, folder="PCA")

    # ----------------------------------------------! SVM ------------------------------------------------
    # Set the parameters by cross-validation
    # multi_class =”crammer_singer”
    if SVM:
        if not os.path.exists("SVM"):
            os.makedirs("SVM")
        os.chdir(os.path.join(os.path.abspath(os.path.curdir), "SVM"))
        print("SVM...")

        # update each classifier results in a mutual file
        svm_results_file = Path("all_svm_results.csv")
        if not svm_results_file.exists():
            all_svm_results = pd.DataFrame(columns=['KERNEL', 'GAMMA', 'C',
                                                    'TRAIN-AUC', 'TRAIN-ACC',
                                                    'TEST-AUC', 'TEST-ACC'])
            all_svm_results.to_csv(svm_results_file, index=False)


        optional_classifiers = []

        if check_all_parameters:
            svm_tuned_parameters = {'kernel': ['linear'],  ###### 'rbf', 'poly', 'sigmoid',  ??????????????
                                     'gamma': ['auto', 'scale'],
                                     'C': [0.01, 0.1, 1, 10, 100, 1000]}
            # create all possible classifiers
            weights = data_loader.get_weights()
            for kernel in svm_tuned_parameters['kernel']:
                for gamma in svm_tuned_parameters['gamma']:
                    for C in svm_tuned_parameters['C']:
                        clf = svm.SVC(kernel=kernel, C=C, gamma=gamma, class_weight=weights)  # class_weight='balanced')
                        optional_classifiers.append(clf)
        else:  # use the wanted classifier
            clf = svm.SVC(kernel=svm_parameters['kernel'], C=svm_parameters['C'],
                          gamma=svm_parameters['gamma'], class_weight='balanced')
            optional_classifiers.append(clf)

        for clf in optional_classifiers:
            all_svm_results = pd.read_csv(svm_results_file)
            clf_folder_name = "k=" + clf.kernel + "_" + "c=" + str(clf.C) + "_" + "g=" + clf.gamma
            if not os.path.exists(clf_folder_name):
                os.makedirs(clf_folder_name)
            # Split the data set
            X_trains, X_tests, y_trains, y_tests, svm_coefs = [], [], [], [], []
            svm_y_test_from_all_iter, svm_y_score_from_all_iter = np.array([]), np.array([])
            svm_y_pred_from_all_iter, svm_class_report_from_all_iter = np.array([]), np.array([])
            train_accuracies, test_accuracies, confusion_matrixes, y_train_preds, y_train_scores,\
            y_test_preds = [], [], [], [], [], []

            for i in range(cross_validation):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)
                X_trains.append(X_train)
                X_tests.append(X_test)
                y_trains.append(y_train)
                y_tests.append(y_test)

            bacteria_coeff_average = []

            for iter_num in range(cross_validation):
                print('------------------------------\niteration number ' + str(iter_num))
                # FIT
                clf.fit(X_trains[iter_num], y_trains[iter_num])
                # GET RESULTS
                y_score = clf.decision_function(X_tests[iter_num])  # what is this for?
                y_pred = clf.predict(X_tests[iter_num])
                y_test_preds.append(y_pred)
                svm_class_report = classification_report(y_tests[iter_num], y_pred).split("\n")
                train_pred = clf.predict(X_trains[iter_num])
                train_score = clf.decision_function(X_trains[iter_num])
                y_train_preds.append(train_pred)
                y_train_scores.append(train_score)
                # SAVE RESULTS
                train_accuracies.append(accuracy_score(y_trains[iter_num], train_pred))
                test_accuracies.append(accuracy_score(y_tests[iter_num], y_pred))
                confusion_matrixes.append(confusion_matrix(y_tests[iter_num], y_pred))
                # AUC
                if BINARY:
                    _, _, _, svm_roc_auc = roc_auc(y_tests[iter_num], y_pred, visualize=False,
                                                       graph_title='SVM\n' + str(iter_num), save=True, folder=task_name)
                # SAVE y_test AND y_score
                svm_y_test_from_all_iter = np.append(svm_y_test_from_all_iter, y_tests[iter_num])  # .values)
                svm_y_pred_from_all_iter = np.append(svm_y_pred_from_all_iter, list(y_pred))
                svm_class_report_from_all_iter = np.append(svm_class_report_from_all_iter, svm_class_report)
                if svm_y_score_from_all_iter.size > 0:
                    svm_y_score_from_all_iter = np.concatenate((svm_y_score_from_all_iter, y_score), axis=0)
                else:
                    svm_y_score_from_all_iter = y_score
                # --------------------------------------------! COEFF PLOTS -----------------------------------------
                if create_coeff_plots:
                    svm_coefs, bacterias, coefficients, bacteria_coeff_average = svm_calc_bacteria_coeff_average(data_loader, clf, svm_coefs,
                                                                            bacteria_coeff_average)



            # --------------------------------------------! AUC -----------------------------------------
            all_y_train, all_predictions_train, all_test_real_tags, all_test_pred_tags, train_auc, test_auc, train_rho, \
            test_rho = calc_auc_on_joined_results(cross_validation, y_trains, y_train_preds, y_tests, y_test_preds)

            # ----------------------------------------! CONFUSION MATRIX -------------------------------------
            print("\n------------------------------")
            names = data_loader.get_confusin_matrix_names()
            # binary = len(names) == 2
            confusion_matrix_average, confusion_matrix_acc = edit_confusion_matrix(title, confusion_matrixes, data_loader,
                                                                            "SVM", names, BINARY=BINARY)
            if BINARY:
                _, _, _, svm_roc_auc = roc_auc(svm_y_test_from_all_iter.astype(int), svm_y_score_from_all_iter,
                                               visualize=True, graph_title='SVM\n' + task_name.capitalize() +
                                                                           " AUC on all iterations", save=True, folder=clf_folder_name)
                res_path = os.path.join(clf_folder_name, str(round(svm_roc_auc, 5)))
            else:
                svm_roc_auc = 0
                res_path = clf_folder_name

            if not os.path.exists(res_path):
                os.mkdir(res_path)

            if create_coeff_plots:
                plot_bacteria_coeff_average(bacteria_coeff_average, len(names), data_loader, title, task_name,
                                            bacterias, cross_validation, "SVM", res_path, BINARY)

            # if allow_printing:
            print_confusion_matrix(confusion_matrix_average, names, confusion_matrix_acc, "SVM", task_name, res_path)

            t = np.array(y_trains).astype(int)
            t = t.flatten()
            s = np.array(y_train_scores)
            s = s.flatten()

            if BINARY:
                _, _, _, svm_train_roc_auc = roc_auc(t, s, visualize=False, graph_title="train auc", save=False, folder=res_path)
            else:
                svm_train_roc_auc = 0
                multi_class_roc_auc(svm_y_test_from_all_iter.astype(int), svm_y_score_from_all_iter, names,
                                    graph_title='SVM\n' + task_name.capitalize() + " AUC on all iterations",
                                    save=True, folder=res_path)
            # ----------------------------------------! SAVE RESULTS -------------------------------------
            save_results(task_name, train_auc, test_auc, train_rho, test_rho, confusion_matrix_average,
                         confusion_matrix_acc,
                         train_accuracies, test_accuracies, svm_y_score_from_all_iter, svm_y_pred_from_all_iter,
                         svm_y_test_from_all_iter, "SVM", res_path)

            all_svm_results.loc[len(all_svm_results)] = [clf.kernel, clf.C, clf.gamma, svm_train_roc_auc,
                                                         np.mean(train_accuracies), svm_roc_auc,
                                                         np.mean(test_accuracies)]
            if BINARY:
                all_svm_results = all_svm_results.sort_values(by=['TEST-AUC'], ascending=False)
            else:
                all_svm_results = all_svm_results.sort_values(by=['TEST-ACC'], ascending=False)

            all_svm_results.to_csv(svm_results_file, index=False)

    # ----------------------------------------------! XGBOOST ------------------------------------------------
    if XGBOOST:
        if SVM:
            os.chdir("..")
        if not os.path.exists("XGBOOST"):
            os.makedirs("XGBOOST")

        os.chdir(os.path.join(os.path.abspath(os.path.curdir), ("XGBOOST")))

        print("XGBOOST...")

        # update each classifier results in a mutual file
        xgb_results_file = Path("all_xgb_results.csv")
        if not xgb_results_file.exists():
            all_xgb_results = pd.DataFrame(columns=['LR', 'MAX-DEPTH', 'N-ESTIMATORS', 'OBJECTIVE',
                                                    'GAMMA', 'MIN-CHILD-WEIGHT', 'BOOSTER',
                                                    'TRAIN-AUC', 'TRAIN-ACC',
                                                    'TEST-AUC', 'TEST-ACC'])
            all_xgb_results.to_csv(xgb_results_file, index=False)

        optional_classifiers = []

        if check_all_parameters:
            """
            xgboost_tuned_parameters = {'learning_rate': [0.01, 0.05, 0.1],
                                         'objective': ['binary:logistic'],
                                         'n_estimators': [1000],
                                         'max_depth': range(3, 10),
                                         'min_child_weight': range(1, 12),
                                         'gamma': [0.0, 0.1, 0.2, 0.3, 1, 3, 6, 9]}
            """
            xgboost_tuned_parameters = {'learning_rate': [0.01, 0.05, 0.1],
                                         'objective': ['binary:logistic'],
                                         'n_estimators': [1000],
                                         'max_depth': [3, 5, 7, 9],
                                         'min_child_weight': [1, 5, 9],
                                         'gamma': [0.0, 0.5, 1, 5, 9]}
            # create all possible classifiers
            for max_depth in xgboost_tuned_parameters['max_depth']:
                for learning_rate in xgboost_tuned_parameters['learning_rate']:
                    for n_estimators in xgboost_tuned_parameters['n_estimators']:
                        for objective in xgboost_tuned_parameters['objective']:
                            for gamma in xgboost_tuned_parameters['gamma']:
                                for min_child_weight in xgboost_tuned_parameters['min_child_weight']:
                                    clf = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate,
                                                        n_estimators=n_estimators, objective=objective,
                                                        gamma=gamma, min_child_weight=min_child_weight,
                                                        booster='gblinear')
                                    optional_classifiers.append(clf)
        else:  # use the wanted classifier
            clf = XGBClassifier(max_depth=xgb_parameters['max_depth'],
                                learning_rate=xgb_parameters['learning_rate'],
                                n_estimators=xgb_parameters['n_estimators'],
                                objective=xgb_parameters['objective'],
                                gamma=xgb_parameters['gamma'],
                                min_child_weight=xgb_parameters['min_child_weight'],
                                booster='gblinear')
            optional_classifiers.append(clf)

        for clf in optional_classifiers:
            all_xgb_results = pd.read_csv(xgb_results_file)
            clf_folder_name = "d=" + str(clf.max_depth) + "_lr=" + str(clf.learning_rate) + "_e=" +\
                              str(clf.n_estimators) + "_o=" + clf.objective + "_g=" + str(clf.gamma) + "_m=" +\
                              str(clf.min_child_weight) + "_b=" + clf.booster
            if not os.path.exists(clf_folder_name):
                os.makedirs(clf_folder_name)

            # Split the data set
            X_trains, X_tests, y_trains, y_tests, xgb_coefs = [], [], [], [], []
            xgb_y_test_from_all_iter, xgb_y_score_from_all_iter = np.array([]), np.array([])
            xgb_y_pred_from_all_iter, xgb_class_report_from_all_iter = np.array([]), np.array([])
            xgb_coefs, bacteria_coeff_average, y_train_scores = [], [], []
            train_accuracies, test_accuracies, confusion_matrixes, y_train_preds, y_test_preds = [], [], [], [], []

            for i in range(cross_validation):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, stratify=y)
                X_trains.append(X_train)
                X_tests.append(X_test)
                y_trains.append(y_train)
                y_tests.append(y_test)

            for iter_num in range(cross_validation):
                print("------------------------------\niteration number " + str(iter_num))

                classes_sum = [np.sum(np.array(y_trains[iter_num]) == unique_class) for unique_class in
                               np.unique(np.array(y_trains[iter_num]))]
                classes_ratio = [1 - (a / sum(classes_sum)) for a in classes_sum]
                weights = [classes_ratio[a] for a in np.array(y_trains[iter_num])]  # classes_ratio

                clf.fit(np.array(X_trains[iter_num]), np.array(y_trains[iter_num]), sample_weight=weights)
                clf.predict_proba(X_tests[iter_num])
                y_score = clf.predict_proba(X_tests[iter_num])  # what is this for?
                y_pred = clf.predict(X_tests[iter_num])
                y_test_preds.append(y_pred)
                y_pred = clf.predict(X_tests[iter_num])
                xgb_class_report = classification_report(y_tests[iter_num], y_pred)
                train_pred = clf.predict(X_trains[iter_num])
                train_score = clf.predict_proba(X_trains[iter_num])
                y_train_preds.append(train_pred)
                y_train_scores.append(train_score)

                train_accuracies.append(accuracy_score(y_trains[iter_num], clf.predict(X_trains[iter_num])))
                test_accuracies.append(accuracy_score(y_tests[iter_num], y_pred))  # same as - clf.score(X_test, y_test)
                confusion_matrixes.append(confusion_matrix(y_tests[iter_num], y_pred))


                if BINARY:
                    _, _, _, xgb_roc_auc = roc_auc(y_tests[iter_num], y_pred, visualize=True,
                                               graph_title='XGB\n' + str(iter_num), folder=task_name)
                else:
                    xgb_roc_auc = 0

                # save the y_test and y_score
                xgb_y_test_from_all_iter = np.append(xgb_y_test_from_all_iter, y_tests[iter_num])
                xgb_y_pred_from_all_iter = np.append(xgb_y_pred_from_all_iter, y_pred)
                xgb_class_report_from_all_iter = np.append(xgb_class_report_from_all_iter, xgb_class_report)
                if xgb_y_score_from_all_iter.size > 0:
                    xgb_y_score_from_all_iter = np.concatenate((xgb_y_score_from_all_iter, y_score), axis=0)
                else:
                    xgb_y_score_from_all_iter = y_score
                # --------------------------------------! PLOT CORRELATION - XGBOOST -------------------------------
                # if create_coeff_plots:
                #     num_of_classes, bacterias = xgb_calc_bacteria_coeff_average(data_loader, clf, xgb_coefs,
                #                                                             bacteria_coeff_average)
            # if create_coeff_plots:
            #     plot_bacteria_coeff_average(bacteria_coeff_average, num_of_classes, data_loader, title, task_name,
            #                                 bacterias, cross_validation, "XGB")

            all_y_train, all_predictions_train, all_test_real_tags, all_test_pred_tags, train_auc, test_auc, train_rho, \
            test_rho = calc_auc_on_joined_results(cross_validation, y_trains, y_train_preds, y_tests, y_test_preds)

            names = data_loader.get_confusin_matrix_names()
            confusion_matrix_average, confusion_matrix_acc = \
                edit_confusion_matrix(title, confusion_matrixes, data_loader, "XGB", names, BINARY=BINARY)

            if BINARY:
                _, _, _, xgb_roc_auc = roc_auc(xgb_y_test_from_all_iter.astype(int), xgb_y_score_from_all_iter[:, 1],
                                               visualize=True,
                                               graph_title='XGB\n' + task_name.capitalize() + " AUC on all iterations",
                                               save=True, folder=clf_folder_name)
                res_path = os.path.join(clf_folder_name, str(round(xgb_roc_auc, 5)))

            else:
                xgb_roc_auc = 0
                res_path = clf_folder_name

            if not os.path.exists(res_path):
                os.mkdir(res_path)

            # if allow_printing:
            print_confusion_matrix(confusion_matrix_average, names, confusion_matrix_acc, "XGB", task_name, res_path)

            t = np.array(y_trains).astype(int)
            t = t.flatten()
            s = np.array(y_train_scores)
            s = s.flatten()
            c = s[::2]

            if BINARY:
                _, _, _, xgb_train_roc_auc = roc_auc(t, c, visualize=False, graph_title="", save=False,
                                                 folder=res_path)
            else:
                xgb_train_roc_auc = 0
                multi_class_roc_auc(xgb_y_test_from_all_iter.astype(int), xgb_y_score_from_all_iter, names,
                                graph_title='XGB\n' + task_name.capitalize() + " AUC on all iterations",
                                           save=True, folder=res_path)
            # ----------------------------------------! SAVE RESULTS -------------------------------------


            save_results(task_name, train_auc, test_auc, train_rho, test_rho, confusion_matrix_average,
                         confusion_matrix_acc,
                         train_accuracies, test_accuracies, xgb_y_score_from_all_iter, xgb_y_pred_from_all_iter,
                         xgb_y_test_from_all_iter, "XGB", res_path)

            all_xgb_results.loc[len(all_xgb_results)] = [clf.learning_rate, clf.max_depth, clf.n_estimators,
                                                         clf.objective, clf.gamma, clf.min_child_weight, clf.booster,
                                                         xgb_train_roc_auc, np.mean(train_accuracies), xgb_roc_auc,
                                                         np.mean(test_accuracies)]
            if BINARY:
                all_xgb_results = all_xgb_results.sort_values(by=['TEST-AUC'], ascending=False)
            else:
                all_xgb_results = all_xgb_results.sort_values(by=['TEST-ACC'], ascending=False)

            all_xgb_results.to_csv(xgb_results_file, index=False)

    # ----------------------------------------------! NN ------------------------------------------------

    if NN:
        if SVM or XGBOOST:
            os.chdir("..")
        if not os.path.exists("NN"):
            os.makedirs("NN")

        param_dict = {"lr": [0.001], "test_size": [0.2], "batch_size": [4], "shuffle": [True],
                      "num_workers": [4], "epochs": [500]}


        for lr in param_dict['lr']:
            for test_size in param_dict['test_size']:
                    for batch_size in param_dict['batch_size']:
                        for shuffle in param_dict['shuffle']:
                            for num_workers in param_dict['num_workers']:
                                for epochs in param_dict['epochs']:
                                    clf_folder_name = "lr=" + str(lr) + "_t=" + str(test_size) + "_bs=" +\
                                                      str(batch_size) + "_s=" + str(shuffle) + "_nw=" +\
                                                      str(num_workers) + "_e=" + str(epochs)
                                    if not os.path.exists(clf_folder_name):
                                        os.makedirs(clf_folder_name)
                                    nn_main(X, y, title, clf_folder_name, 20, 40, 60, 2,
                                            lr, test_size, batch_size, shuffle, 4, 500)

def run(learning_tasks, bactria_as_feature_file, samples_data_file, tax):
    """
    # Pre process - mostly not needed
    allow_printing = False  # do you want printing to screen?
    HIST = False  # drew histogram of the data after normalization
    perform_regression = False  # run normalization to drew plots
    STAGES = False  # make tags for stages task
    calculate_rhos = False  # calculate correlation between reaction to treatment to the tags
    """
    for task in learning_tasks:
        data_loader = AbstractDataLoader(title=task, bactria_as_feature_file=bactria_as_feature_file,
                                         samples_data_file=samples_data_file, taxnomy_level=tax,
                                         allow_printing=True, perform_anna_preprocess=False)

        learn(title=task, data_loader=data_loader,
              allow_printing=False, calculate_rhos=False,
              SVM=False, XGBOOST=False, NN=True,
              cross_validation=5, create_coeff_plots=True,
              check_all_parameters=False, svm_parameters=None,
              xgb_parameters=None, create_pca_plots=False)

if __name__ == "__main__":
    learning_tasks = ["first_task", "second_task"]
    bactria_as_feature_file = 'feature-table_Allergy_cleaned_taxa_290119_updated_in_140219.csv'
    samples_data_file = 'mf_merge_ok84_ok93_ok66_69_merged_by_RestoredSampleCode_as_ID_290119.csv'
    tax = 6
    # run(learning_tasks, bactria_as_feature_file, samples_data_file, tax)
    """
    # Pre process - mostly not needed
    allow_printing = False  # do you want printing to screen?
    HIST = False  # drew histogram of the data after normalization
    perform_regression = False  # run normalization to drew plots
    STAGES = False  # make tags for stages task
    calculate_rhos = False  # calculate correlation between reaction to treatment to the tags
    """

    for task in learning_tasks:
        data_loader = AbstractDataLoader(title=task, bactria_as_feature_file=bactria_as_feature_file,
                                         samples_data_file=samples_data_file, taxnomy_level=tax,
                                         allow_printing=True, perform_anna_preprocess=False)

        learn(title=task, data_loader=data_loader,
              allow_printing=False, calculate_rhos=False,
              SVM=False, XGBOOST=False, NN=True,
              cross_validation=5, create_coeff_plots=True,
              check_all_parameters=False, svm_parameters=None,
              xgb_parameters=None,
              create_pca_plots=False)
