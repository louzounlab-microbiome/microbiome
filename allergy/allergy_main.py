import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from imblearn.keras import BalancedBatchGenerator
from sklearn import svm, metrics
from scipy import stats
from sklearn.metrics import classification_report, roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from tensorflow.python.keras import regularizers
from xgboost import XGBClassifier
import tensorflow as tf
from sklearn.utils import class_weight

from dafna.nn import nn_main
from infra_functions.general import convert_pca_back_orig, draw_rhos_calculation_figure
# import keras

from allergy.allergy_data_loader import AllergyDataLoader

from dafna.plot_auc import multi_class_roc_auc, roc_auc
from dafna.plot_coef import create_coeff_plots_by_alogorithm
from dafna.plot_confusion_mat import edit_confusion_matrix, print_confusion_matrix

from infra_functions import tf_analaysis
from infra_functions.general import convert_pca_back_orig

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
n_components = 20


def generator(gen):
    for g in gen:
        yield g


# ----------------------------------------------! learning methods ------------------------------------------------


def get_learning_data(title, data_loader, only_single_allergy_type):
    if title == 'Success_task':
        W_CON = False
        ids = data_loader.get_ids_list_wo_con
        tag_map = data_loader.get_id_to_binary_success_tag_map
        task_name = 'success task'

    elif title == 'Health_task':
        W_CON = True
        ids = data_loader.get_ids_list_w_con
        tag_map = data_loader.get_id_to_binary_health_tag_map
        task_name = 'health task'

    elif title == 'Prognostic_task':
        # for only stage 0
        W_CON = False
        ids = data_loader.get_stage_0_ids
        tag_map = data_loader.get_id_to_binary_success_tag_map
        task_name = 'prognostic task'

    elif title == 'Allergy_type_task':

        W_CON = False
        # time_zero = data_loader.get_stage_0_ids
        if only_single_allergy_type:
            ids = data_loader.get_ids_list_wo_multiple
            task_name = 'allergy type task on single allergy'
        else:
            ids = data_loader.get_id_wo_non_and_egg_allergy_type_list
            task_name = 'allergy type task'
        # ids = [i for i in ids if i in time_zero]
        tag_map = data_loader.get_id_to_allergy_number_type_tag_map



    elif title == 'Milk_allergy_task':
        W_CON = False
        ids = data_loader.get_id_wo_non_and_egg_allergy_type_list
        tag_map = data_loader.get_id_to_milk_allergy_tag_map
        task_name = 'milk allergy task'

    return W_CON, list(ids), tag_map, task_name


def get_weights(title, data_loader):
    if title == "Allergy_type_task":
        allergy_type_to_weight_map = data_loader.get_allergy_type_to_weight_map
        allergy_type_weights = list(allergy_type_to_weight_map.values())
        return allergy_type_weights

    if title == 'Milk_allergy_task':
        milk_vs_other_allergy_weight_map = data_loader.get_milk_vs_other_allergy_weight_map
        milk_vs_other_allergy_weights = list(milk_vs_other_allergy_weight_map.values())
        return milk_vs_other_allergy_weights

    if title == 'Health_task':
        healthy_vs_allergic_weight_map = data_loader.get_healthy_vs_allergic_weight_map
        healthy_vs_allergic_weights = list(healthy_vs_allergic_weight_map.values())
        return healthy_vs_allergic_weights

    if title == 'Success_task':
        responding_vs_not_weight_map = data_loader.get_responding_vs_not_weight_map
        responding_vs_not_weights = list(responding_vs_not_weight_map.values())
        return responding_vs_not_weights

    if title == 'Prognostic_task':
        prognostic_responding_vs_not_weight_map = data_loader.get_prognostic_responding_vs_not_weight_map
        prognostic_responding_vs_not_weights = list(prognostic_responding_vs_not_weight_map.values())
        return prognostic_responding_vs_not_weights

    return None


def get_svm_clf(title):
    if title == 'Success_task':
        # {'C': 0.01, 'gamma': 'auto', 'kernel': 'linear'}
        # 0.676742022800625
        clf = svm.SVC(kernel='linear', C=0.01, gamma='auto', class_weight='balanced')

    if title == 'Health_task':
        # {'C': 0.01, 'gamma': 'scale', 'kernel': 'sigmoid'}
        # 0.8334190537769244
        clf = svm.SVC(kernel='linear', C=0.01, gamma='scale', class_weight='balanced')

    if title == 'Prognostic_task':
        # {'C': 0.1, 'gamma': 'scale', 'kernel': 'poly'}
        # 0.6451672886289899
        clf = svm.SVC(kernel='linear', C=0.1, gamma='scale', class_weight='balanced')

    if title == 'Allergy_type_task':
        # {'C'.........
        # 0.......
        # clf = svm.LinearSVC(dual=False, C=0.01, multi_class='ovr', class_weight='balanced', max_iter=10000000)
        clf = svm.LinearSVC(dual=False, C=1, multi_class='ovr', class_weight='balanced', max_iter=10000000)

    if title == 'Milk_allergy_task':
        # {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}
        #  0.7081696718971948
        clf = svm.SVC(kernel='linear', C=1, gamma='scale', class_weight='balanced')
    return clf


def get_xgb_clf(title, weights):
    if title == 'Success_task':
        # success
        # {'gamma': 6, 'learning_rate': 0.2, 'max_depth': 3, 'min_child_weight': 5,
        # 'n_estimators': 1000, 'objective': 'binary:logistic'}
        # 0.656803999870332
        if weights:  # sample_weight=weights
            clf = XGBClassifier(max_depth=3, learning_rate=0.2, n_estimators=1000, objective='binary:logistic',
                            gamma=6, min_child_weight=5, sample_weight='balanced', booster='gblinear')
        else:
            clf = XGBClassifier(max_depth=3, learning_rate=0.2, n_estimators=1000, objective='binary:logistic',
                                gamma=6, min_child_weight=5)

    elif title == 'Health_task':
        # health
        # {'gamma': 9, 'learning_rate': 0.2, 'max_depth': 3, 'min_child_weight': 7,
        # 'n_estimators': 1000, 'objective': 'binary:logistic'}
        # 0.7884455411996166
        if weights:
            clf = XGBClassifier(max_depth=3, learning_rate=0.2, n_estimators=1000, objective='binary:logistic',
                            gamma=9, min_child_weight=7, sample_weight='balanced', booster='gblinear')
        else:
            clf = XGBClassifier(max_depth=3, learning_rate=0.2, n_estimators=1000, objective='binary:logistic',
                                gamma=9, min_child_weight=7)

    elif title == 'Prognostic_task':
        # prognostic
        # {'gamma': 0.4, 'learning_rate': 0.2, 'max_depth': 3, 'min_child_weight': 7,
        # 'n_estimators': 1000, 'objective': 'binary:logistic'}
        # 0.6370625424793854
        if weights:
            clf = XGBClassifier(max_depth=3, learning_rate=0.2, n_estimators=1000, objective='binary:logistic',
                                gamma=3, min_child_weight=7, sample_weight='balanced', booster='gblinear')  # class_weight='balanced'

        else:
             clf = XGBClassifier(max_depth=3, learning_rate=0.2, n_estimators=1000, objective='binary:logistic',
                            gamma=3, min_child_weight=7)

    elif title == 'Allergy_type_task':
        # type
        # {'C'...
        # 0....
        if weights:
            clf = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, objective='binary:logistic',
                                gamma=0, min_child_weight=1, sample_weight='balanced', booster='gblinear') #  class_weight='balanced'

        else:
            clf = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, objective='binary:logistic',
                            gamma=0, min_child_weight=1)

    elif title == 'Milk_allergy_task':
        # type
        # {'C'...
        # 0....
        if weights:
            clf = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, objective='binary:logistic',
                            gamma=0, min_child_weight=1, sample_weight='balanced', booster='gblinear')
        else:
            clf = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, objective='binary:logistic',
                                gamma=0, min_child_weight=1)

    return clf


def calc_auc_on_joined_results(Cross_validation, y_trains, y_train_preds, y_tests, y_test_preds):
    all_y_train = []
    for i in range(Cross_validation):
        all_y_train = all_y_train + y_trains[i]

    all_predictions_train = []
    for i in range(Cross_validation):
        all_predictions_train = all_predictions_train + list(y_train_preds[i])

    all_test_real_tags = []
    for i in range(Cross_validation):
        all_test_real_tags = all_test_real_tags + y_tests[i]

    all_test_pred_tags = []
    for i in range(Cross_validation):
        all_test_pred_tags = all_test_pred_tags + list(y_test_preds[i])

    try:
        train_auc = metrics.roc_auc_score(all_y_train, all_predictions_train)
        #fpr, tpr, thresholds = metrics.roc_auc_score(all_test_real_tags, all_test_pred_tags)
        # test_auc = metrics.auc(fpr, tpr)
        test_auc = metrics.roc_auc_score(all_test_real_tags, all_test_pred_tags)
        train_rho, pval_train = stats.spearmanr(all_y_train, np.array(all_predictions_train))
        test_rho, p_value = stats.spearmanr(all_test_real_tags, np.array(all_test_pred_tags))
    except ValueError:
        # Compute ROC curve and ROC area for each class
        print("train classification_report")
        train_auc = metrics.classification_report(all_y_train, all_predictions_train)
        for row in train_auc.split("\n"):
            print(row)
        print("test classification_report")
        test_auc = metrics.classification_report(all_test_real_tags, all_test_pred_tags)
        for row in test_auc.split("\n"):
            print(row)

        train_rho, pval_train = stats.spearmanr(all_y_train, np.array(all_predictions_train))
        test_rho, p_value = stats.spearmanr(all_test_real_tags, np.array(all_test_pred_tags))

    return all_y_train, all_predictions_train, all_test_real_tags, all_test_pred_tags,\
           train_auc, test_auc, train_rho, test_rho


def get_confusin_matrix_names(data_loader, title):
    if title in ["Success_task",  "Health_task", "Prognostic_task", "Milk_allergy_task"]:
        if title == "Milk_allergy_task":
            names = ['Other', 'Milk']
        elif title == "Health_task":
            names = ['Allergic', 'Healthy']
        elif title in ["Success_task", "Prognostic_task"]:
            names = ['No', 'Yes']

    elif title in ["Allergy_type_task"]:  # MULTI CLASS
        tag_to_allergy_type_map = data_loader.get_tag_to_allergy_type_map
        allergy_type_to_instances_map = data_loader.get_allergy_type_to_instances_map
        allergy_type_to_weight_map = data_loader.get_allergy_type_to_weight_map
        allergy_type_weights = list(allergy_type_to_weight_map.values())
        names = []
        for key in range(len(tag_to_allergy_type_map.keys())):
            names.append(tag_to_allergy_type_map.get(key))

    return names


def run_learning(TITLE, PRINT, REG, RHOS, SVM, XGBOOST, NN, Cross_validation, COEFF_PLOTS, TUNED_PAREMETERS, only_single_allergy_type):

    data_loader = AllergyDataLoader(TITLE, PRINT, REG, WEIGHTS=True, ANNA_PREPROCESS=False)

    print("learning..." + TITLE)
    # Learning: x=features, y=tags
    W_CON, ids, tag_map, task_name = get_learning_data(TITLE, data_loader, only_single_allergy_type)

    if RHOS:
        print("calculating rho")
        draw_rhos_calculation_figure(tag_map, data_loader.get_preproccessed_data, TITLE, ids_list=ids, save_folder="rhos")

    id_to_features_map = data_loader.get_id_to_features_map
    X = [id_to_features_map[id] for id in ids]
    y = [tag_map[id] for id in ids]

    # ----------------------------------------------! SVM ------------------------------------------------
    # Set the parameters by cross-validation
    # multi_class =”crammer_singer”
    if SVM:
        print("SVM...")
        if TUNED_PAREMETERS:
            svm_tuned_parameters = [{'kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
                                 'gamma': ['auto', 'scale'],
                                 'C': [0.01, 0.1, 1, 10, 100, 1000]}]

            svm_clf = GridSearchCV(svm.SVC(class_weight='balanced'), svm_tuned_parameters, cv=5,
                                   scoring='roc_auc', return_train_score=True)

            svm_clf.fit(X, y)
            print(svm_clf.best_params_)
            print(svm_clf.best_score_)
            print(svm_clf.cv_results_)

            svm_results = pd.DataFrame(svm_clf.cv_results_)
            svm_results.to_csv(os.path.join(task_name, "svm_all_results_df_" + task_name + ".csv"))
            pickle.dump(svm_results, open(os.path.join(task_name, "svm_all_results_df_" + task_name + ".pkl"), 'wb'))

            # success
            # {'C': 0.01, 'gamma': 'auto', 'kernel': 'linear'}
            # 0.676742022800625
            # second run {'C': 0.1, 'gamma': 'scale', 'kernel': 'rbf'}
            # 0.6633490331056445
            # health
            # {'C': 0.01, 'gamma': 'scale', 'kernel': 'sigmoid'}
            # 0.8334190537769244
            # prognostic
            # {'C': 0.1, 'gamma': 'scale', 'kernel': 'poly'}
            # 0.6451672886289899
            # type
            #
            #
            # svm_means_test = svm_clf.cv_results_['mean_test_score']
            # svm_stds_test = svm_clf.cv_results_['std_test_score']
            # svm_means_train = svm_clf.cv_results_['mean_train_score']
            # svm_stds_train = svm_clf.cv_results_['std_train_score']
            # pickle.dump(svm_stds_test, open("svm_stds_test_" + task_name + ".pkl", 'wb'))
            # pickle.dump(svm_means_train, open("svm_means_train_" + task_name + ".pkl", 'wb'))
            # pickle.dump(svm_stds_train, open("svm_stds_train_" + task_name + ".pkl", 'wb'))
            # pickle.dump(svm_clf, open("svm_clf_" + task_name + ".pkl", 'wb'))
            # else:
                # svm_stds_test = pickle.load(open("svm_stds_test_" + task_name + ".pkl", "rb"))
                # svm_stds_train = pickle.load(open("svm_means_train_" + task_name + ".pkl", "rb"))
                # svm_stds_train = pickle.load(open("svm_stds_train_" + task_name + ".pkl", "rb"))
                # svm_clf = pickle.load(open("svm_clf_" + task_name + ".pkl", "rb"))

        # Split the data set
        X_trains = []
        X_tests = []
        y_trains = []
        y_tests = []
        svm_y_test_from_all_iter = np.array([])
        svm_y_score_from_all_iter = np.array([])
        svm_y_pred_from_all_iter = np.array([])
        svm_class_report_from_all_iter = np.array([])
        svm_coefs = []

        train_accuracies = []
        test_accuracies = []
        confusion_matrixes = []
        y_train_preds = []
        y_test_preds = []

        for i in range(Cross_validation):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
            X_trains.append(X_train)
            X_tests.append(X_test)
            y_trains.append(y_train)
            y_tests.append(y_test)

        bacteria_average = []
        bacteria_coeff_average = []

        for iter_num in range(Cross_validation):
            print(f'------------------------------\nIteration number {iter_num}')
            # SVM
            clf = get_svm_clf(TITLE)

            clf.fit(X_trains[iter_num], y_trains[iter_num])
            y_score = clf.decision_function(X_tests[iter_num])  # what is this for?
            y_pred = clf.predict(X_tests[iter_num])
            y_test_preds.append(y_pred)
            svm_class_report = classification_report(y_tests[iter_num], y_pred).split("\n")
            train_pred = clf.predict(X_trains[iter_num])
            y_train_preds.append(train_pred)

            train_accuracies.append(accuracy_score(y_trains[iter_num], train_pred))
            test_accuracies.append(accuracy_score(y_tests[iter_num], y_pred))  # same as - clf.score(X_test, y_test)

            confusion_matrixes.append(confusion_matrix(y_tests[iter_num], y_pred))

            try:  # pred or score??
                _, _, _, svm_roc_auc = roc_auc(y_tests[iter_num], y_pred, visualize=False,
                                           graph_title='SVM\n' + str(iter_num), save=True, folder=task_name)
            except ValueError:
                print(task_name + " multiclass format is not supported at roc auc calculation")


            # save the y_test and y_score
            svm_y_test_from_all_iter = np.append(svm_y_test_from_all_iter, y_tests[iter_num])  # .values)
            if svm_y_score_from_all_iter.size > 0:
                svm_y_score_from_all_iter = np.concatenate((svm_y_score_from_all_iter, y_score), axis=0)
            else:
                svm_y_score_from_all_iter = y_score
            svm_y_pred_from_all_iter = np.append(svm_y_pred_from_all_iter, list(y_pred))
            svm_class_report_from_all_iter = np.append(svm_class_report_from_all_iter, svm_class_report)

            if COEFF_PLOTS:
                preproccessed_data = data_loader.get_preproccessed_data
                pca_obj = data_loader.get_pca_obj
                num_of_classes = clf.coef_.shape[0]
                if num_of_classes > 1:                     # multi-class
                    c = clf.coef_.tolist()
                    svm_coefs.append(c)
                    bacterias = convert_pca_back_orig(pca_obj.components_, c[0],
                                                           original_names=preproccessed_data.columns[:],
                                                           visualize=False)['Taxonome'].tolist()

                    coefficients = np.array([convert_pca_back_orig(pca_obj.components_, c[i],
                                                           original_names=preproccessed_data.columns[:],
                                                           visualize=False)['Coefficients'].tolist() for i in range(num_of_classes)])
                    bacteria_coeff_average.append(coefficients)

                else:                     # binary
                    c = clf.coef_.tolist()[0]
                    svm_coefs.append(c)
                    bacteria_coeff = convert_pca_back_orig(pca_obj.components_, c,
                                                           original_names=preproccessed_data.columns[:], visualize=False)

                    bacterias = bacteria_coeff['Taxonome'].tolist()
                    coefficients = bacteria_coeff['Coefficients'].tolist()
                    bacteria_coeff_average.append(coefficients)

                """
                svm_coefs.append(c)
                entire_W = clf.coef_[0]

                preproccessed_data = data_loader.get_preproccessed_data
                pca_obj = data_loader.get_pca_obj
                bacteria_coeff = convert_pca_back_orig(pca_obj.components_, entire_W,
                                                       original_names=preproccessed_data.columns[:], visualize=False)

                coefficients = bacteria_coeff['Coefficients'].tolist()
                bacteria_coeff_average.append(coefficients)
                """

        if COEFF_PLOTS:
            bacteria_coeff_average = np.array(bacteria_coeff_average)
            if num_of_classes > 1:                      # multi-class
                for i in range(num_of_classes):
                    avg_df = pd.DataFrame(bacteria_coeff_average[i])
                    avg_cols = [x for x in avg_df.mean(axis=0)]
                    names = get_confusin_matrix_names(data_loader, TITLE)
                    task_name_and_class = task_name + " - " + names[i].replace("_", " ") + " class"
                    create_coeff_plots_by_alogorithm(avg_cols, bacterias, task_name_and_class, "SVM", Cross_validation, folder=task_name)


            else:                     # binary
                avg_df = pd.DataFrame(bacteria_coeff_average)
                avg_cols = [x for x in avg_df.mean(axis=0)]
                create_coeff_plots_by_alogorithm(avg_cols, bacterias, task_name, "SVM", Cross_validation, folder=task_name)

        all_y_train, all_predictions_train, all_test_real_tags, all_test_pred_tags, train_auc, test_auc, train_rho,\
        test_rho = calc_auc_on_joined_results(Cross_validation, y_trains, y_train_preds, y_tests, y_test_preds)

        print("\n------------------------------\n")
        names = get_confusin_matrix_names(data_loader, TITLE)
        binary = len(names) == 2
        confusion_matrix_average, confusion_matrix_acc = edit_confusion_matrix(TITLE, confusion_matrixes, data_loader,
                                                                               "SVM", names, BINARY=binary)
        print_confusion_matrix(confusion_matrix_average, names, confusion_matrix_acc, "SVM", task_name)
        if PRINT:
            print_confusion_matrix(confusion_matrix_average, names, confusion_matrix_acc, "SVM", task_name)

        try:
            _, _, _, svm_roc_auc = roc_auc(svm_y_test_from_all_iter.astype(int), svm_y_score_from_all_iter,
                                           visualize=True, graph_title='SVM\n' + task_name.capitalize() +
                                           " AUC on all iterations", save=True, folder=task_name)
        except ValueError:
            print(task_name + "multiclass format is not supported at roc auc calculation")
            multi_class_roc_auc(svm_y_test_from_all_iter.astype(int), svm_y_score_from_all_iter, names,
                                           graph_title='SVM\n' + task_name.capitalize() + " AUC on all iterations",
                                           save=True, folder=task_name)

        print("svm final results: " + task_name)
        print("train_auc: " + str(train_auc))
        print("test_auc: " + str(test_auc))
        print("train_rho: " + str(train_rho))
        print("test_rho: " + str(test_rho))
        print("confusion_matrix_average: ")
        print(confusion_matrix_average)
        print("confusion_matrix_acc: ")
        print(confusion_matrix_acc)
        confusion_matrix_average.to_csv(os.path.join(task_name, "svm_confusion_matrix_average_on_" + task_name + ".txt"))

        with open(os.path.join(task_name, "svm_AUC_on_" + task_name + ".txt"), "w") as file:
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
        if len(svm_y_score_from_all_iter.shape) > 1:
            score_map = {"y score " + str(i): svm_y_score_from_all_iter[:, i] for i in range(svm_y_score_from_all_iter.shape[1])}
        else:
            score_map = {"y score": svm_y_score_from_all_iter}

        # score_map = {"y score " + str(i): svm_y_score_from_all_iter[:, i] for i in range(svm_y_score_from_all_iter.shape[1])}
        score_map["y pred"] = svm_y_pred_from_all_iter.astype(int)
        score_map["y test"] = svm_y_test_from_all_iter.astype(int)
        results = pd.DataFrame(score_map)

        results.append([np.mean(train_accuracies), np.mean(test_accuracies), None, None, None, None])
        pickle.dump(results, open(os.path.join(task_name, "svm_clf_results_" + task_name + ".pkl"), 'wb'))
        results.to_csv(os.path.join(task_name, "svm_clf_results_" + task_name + ".csv"))
        pickle.dump(confusion_matrix_average, open(os.path.join(task_name, "svm_clf_confusion_matrix_results_" + task_name + ".pkl"), 'wb'))
        confusion_matrix_average.to_csv(os.path.join(task_name, "svm_clf_confusion_matrix_results_" + task_name + ".csv"))

    # ----------------------------------------------! XGBOOST ------------------------------------------------

    if XGBOOST:
        print("XGBOOST...")
        if TUNED_PAREMETERS:
            xgboost_tuned_parameters = [{'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.2],
                                         'objective': ['binary:logistic'],
                                         'n_estimators': [1000],
                                         'max_depth': range(3, 10),
                                         'min_child_weight': range(1, 12),
                                         'gamma': [0.0, 0.1, 0.2, 0.3, 0.4, 1, 3, 6, 9]}]

            xgb_clf = GridSearchCV(XGBClassifier(class_weight='balanced'), xgboost_tuned_parameters, cv=5,
                                   scoring='roc_auc', return_train_score=True)

            xgb_clf.fit(np.asarray(X), np.asarray(y))  # xgb_clf.fit(X, y)
            print(xgb_clf.best_params_)
            print(xgb_clf.best_score_)
            print(xgb_clf.cv_results_)
            xgb_results = pd.DataFrame(xgb_clf.cv_results_)
            xgb_results.to_csv(os.path.join(task_name, "xgb_all_results_df_" + task_name + ".csv"))
            pickle.dump(xgb_results, open(os.path.join(task_name, "xgb_all_results_df_" + task_name + ".pkl"), 'wb'))
            # xgb_means_test = xgb_clf.cv_results_['mean_test_score']
            # xgb_stds_test = xgb_clf.cv_results_['std_test_score']
            # xgb_means_train = xgb_clf.cv_results_['mean_train_score']
            # xgb_stds_train = xgb_clf.cv_results_['std_train_score']
            # pickle.dump(xgb_stds_test, open("xgb_stds_test_" + task_name + ".pkl", 'wb'))
            # pickle.dump(xgb_means_train, open("xgb_means_train_" + task_name + ".pkl", 'wb'))
            # pickle.dump(xgb_stds_train, open("xgb_stds_train_" + task_name + ".pkl", 'wb'))
            # pickle.dump(xgb_clf, open("xgb_clf_" + task_name + ".pkl", 'wb'))
            # else:
                # xgb_stds_test = pickle.load(open("xgb_stds_test_" + task_name + ".pkl", "rb"))
                # xgb_stds_train = pickle.load(open("xgb_means_train_" + task_name + ".pkl", "rb"))
                # xgb_stds_train = pickle.load(open("xgb_stds_train_" + task_name + ".pkl", "rb"))
                # xgb_clf = pickle.load(open("svm_clf_" + task_name + ".pkl", "rb"))
        # Split the data set
        X_trains = []
        X_tests = []
        y_trains = []
        y_tests = []
        xgb_y_test_from_all_iter = np.array([])
        xgb_y_score_from_all_iter = np.array([])
        xgb_y_pred_from_all_iter = np.array([])
        xgb_class_report_from_all_iter = np.array([])
        xgb_coefs = []
        bacteria_average = []
        bacteria_coeff_average = []

        train_accuracies = []
        test_accuracies = []
        confusion_matrixes = []
        y_train_preds = []
        y_test_preds = []

        for i in range(Cross_validation):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)
            X_trains.append(X_train)
            X_tests.append(X_test)
            y_trains.append(y_train)
            y_tests.append(y_test)

        for iter_num in range(Cross_validation):
            print(f'------------------------------\nIteration number {iter_num}')
            # xgb - determine classifier according to task

            weights = get_weights(TITLE, data_loader)
            clf = get_xgb_clf(TITLE, weights)

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
            y_train_preds.append(train_pred)

            train_accuracies.append(accuracy_score(y_trains[iter_num], clf.predict(X_trains[iter_num])))
            test_accuracies.append(accuracy_score(y_tests[iter_num], y_pred))  # same as - clf.score(X_test, y_test)
            confusion_matrixes.append(confusion_matrix(y_tests[iter_num], y_pred))

            try:
                _, _, _, xgb_roc_auc = roc_auc(y_tests[iter_num], y_pred, visualize=True,
                                               graph_title='XGB\n' + str(iter_num), folder=task_name)
            except ValueError:
                print(task_name + " multiclass format is not supported at roc auc calculation")

            # save the y_test and y_score
            xgb_y_test_from_all_iter = np.append(xgb_y_test_from_all_iter, y_tests[iter_num])
            if xgb_y_score_from_all_iter.size > 0:
                xgb_y_score_from_all_iter = np.concatenate((xgb_y_score_from_all_iter, y_score), axis=0)
            else:
                xgb_y_score_from_all_iter = y_score
            xgb_y_pred_from_all_iter = np.append(xgb_y_pred_from_all_iter, y_pred)
            xgb_class_report_from_all_iter = np.append(xgb_class_report_from_all_iter, xgb_class_report)
        # --------------------------------------! PLOT CORRELATION - XGBOOST -------------------------------
            if COEFF_PLOTS:
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
                    svm_coefs.append(c)
                    bacteria_coeff = convert_pca_back_orig(pca_obj.components_, c,
                                                           original_names=preproccessed_data.columns[:],
                                                           visualize=False)

                    bacterias = bacteria_coeff['Taxonome'].tolist()
                    coefficients = bacteria_coeff['Coefficients'].tolist()
                    bacteria_coeff_average.append(coefficients)

                """
                svm_coefs.append(c)
                entire_W = clf.coef_[0]

                preproccessed_data = data_loader.get_preproccessed_data
                pca_obj = data_loader.get_pca_obj
                bacteria_coeff = convert_pca_back_orig(pca_obj.components_, entire_W,
                                                       original_names=preproccessed_data.columns[:], visualize=False)

                coefficients = bacteria_coeff['Coefficients'].tolist()
                bacteria_coeff_average.append(coefficients)
                """

        if COEFF_PLOTS:
            bacteria_coeff_average = np.array(bacteria_coeff_average)
            if num_of_classes > 1:  # multi-class
                for i in range(num_of_classes):
                    avg_df = pd.DataFrame(bacteria_coeff_average[i])
                    avg_cols = [x for x in avg_df.mean(axis=0)]
                    names = get_confusin_matrix_names(data_loader, TITLE)
                    task_name_and_class = task_name + " - " + names[i].replace("_", " ") + " class"
                    create_coeff_plots_by_alogorithm(avg_cols, bacterias, task_name_and_class, "XGB", Cross_validation,
                                                     folder=task_name)


            else:  # binary
                avg_df = pd.DataFrame(bacteria_coeff_average)
                avg_cols = [x for x in avg_df.mean(axis=0)]
                create_coeff_plots_by_alogorithm(avg_cols, bacterias, task_name, "XGB", Cross_validation,
                                                 folder=task_name)
                """
            if COEFF_PLOTS:
                c = np.array(clf.coef_)
                xgb_coefs.append(c)
                preproccessed_data = data_loader.get_preproccessed_data
                pca_obj = data_loader.get_pca_obj
                bacteria_coeff = convert_pca_back_orig(pca_obj.components_, c, original_names=preproccessed_data.columns[:], visualize=False)
                bacterias = bacteria_coeff['Taxonome'].tolist()
                coefficients = bacteria_coeff['Coefficients'].tolist()
                bacteria_coeff_average.append(coefficients)

                # draw_horizontal_bar_chart(entire_W[0:starting_col], interesting_cols, title='Feature Coeff',
                #                         ylabel='Feature', xlabel='Coeff Value', left_padding=0.3)

        if COEFF_PLOTS:
            bacterias = bacteria_coeff['Taxonome'].tolist()
            avg_df = pd.DataFrame(bacteria_coeff_average)
            avg_cols = [x for x in avg_df.mean(axis=0)]
            create_coeff_plots_by_alogorithm(avg_cols, bacterias, task_name, "XGB", Cross_validation,
                                             folder=task_name)
        """
        all_y_train, all_predictions_train, all_test_real_tags, all_test_pred_tags, train_auc, test_auc, train_rho, \
        test_rho = calc_auc_on_joined_results(Cross_validation, y_trains, y_train_preds, y_tests, y_test_preds)

        names = get_confusin_matrix_names(data_loader, TITLE)
        binary = len(names) == 2
        confusion_matrix_average, confusion_matrix_acc =\
            edit_confusion_matrix(TITLE, confusion_matrixes, data_loader, "XGB", names, BINARY=binary)
        print_confusion_matrix(confusion_matrix_average, names, confusion_matrix_acc, "XGB", task_name)

        if PRINT:
            print_confusion_matrix(confusion_matrix_average, names, confusion_matrix_acc, "XGB", task_name)

        try:  # pred or score ?
            _, _, _, xgb_roc_auc = roc_auc(xgb_y_test_from_all_iter.astype(int), xgb_y_score_from_all_iter[:, 1], visualize=True,
                                           graph_title='XGB\n' + task_name.capitalize() + " AUC on all iterations",
                                           save=True, folder=task_name)
        except ValueError:
            print(task_name + " multiclass format is not supported at roc auc calculation")
            multi_class_roc_auc(xgb_y_test_from_all_iter.astype(int), xgb_y_score_from_all_iter, names,
                                graph_title='XGB\n' + task_name.capitalize() + " AUC on all iterations",
                                           save=True, folder=task_name)  # xgb_y_pred_from_all_iter.astype(int)



        print("\n------------------------------\n")
        print("xgb final results: " + task_name)
        print("train_auc: " + str(train_auc))
        print("test_auc: " + str(test_auc))
        print("train_rho: " + str(train_rho))
        print("test_rho: " + str(test_rho))
        print("confusion_matrix_average: ")
        print(confusion_matrix_average)
        print("confusion_matrix_acc: ")
        print(confusion_matrix_acc)
        confusion_matrix_average.to_csv(os.path.join(task_name, "xgb_confusion_matrix_average_on_" + task_name + ".txt"))

        with open(os.path.join(task_name, "xgb_AUC_on_" + task_name + ".txt"), "w") as file:
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
        if len(xgb_y_score_from_all_iter.shape) > 1:
            score_map = {"y score " + str(i): xgb_y_score_from_all_iter[:, i] for i in range(xgb_y_score_from_all_iter.shape[1])}
        else:
            score_map = {"y score": xgb_y_score_from_all_iter}

        score_map["y pred"] = xgb_y_pred_from_all_iter.astype(int)
        score_map["y test"] = xgb_y_test_from_all_iter.astype(int)
        results = pd.DataFrame(score_map)

        results.append([np.mean(train_accuracies), np.mean(test_accuracies), None, None, None, None])
        pickle.dump(results, open(os.path.join(task_name, "xgb_clf_results_" + task_name + ".pkl"), 'wb'))
        results.to_csv(os.path.join(task_name, "xgb_clf_results_" + task_name + ".csv"))
        pickle.dump(confusion_matrix_average, open(os.path.join(task_name, "xgb_clf_confusion_matrix_results_" + task_name + ".pkl"), 'wb'))
        confusion_matrix_average.to_csv(os.path.join(task_name, "xgb_clf_confusion_matrix_results_" + task_name + ".csv"))

    # ----------------------------------------------! NN ------------------------------------------------

    if NN:
        nn_main(X, y, TITLE, task_name, 20, 40, 60, 2)
        """
        X_trains = []
        X_tests = []
        y_trains = []
        y_tests = []
        for i in range(Cross_validation):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)
            X_trains.append(np.array(X_train))
            X_tests.append(np.array(X_test))
            y_trains.append(np.array(y_train))
            y_tests.append(np.array(y_test))

        nn_y_test_from_all_iter = None
        nn_y_score_from_all_iter = None
        nn_y_pred_from_all_iter = None

        for i in range(Cross_validation):
            print(TITLE + ": round " + str(i+1) + "/" + str(Cross_validation))

            test_model = tf_analaysis.nn_model()
            regularizer = regularizers.l2(1e-5)
            test_model.build_nn_model(hidden_layer_structure=[{'units': n_components},
                                                              {'units': n_components*2, 'activation': tf.nn.relu, 'kernel_regularizer': regularizer},
                                                              ({'rate': 0.5}, 'dropout'),
                                                              {'units': n_components*2, 'activation': tf.nn.relu,'kernel_regularizer': regularizer},
                                                              ({'rate': 0.5}, 'dropout'),
                                                              {'units': n_components * 2, 'activation': tf.nn.relu, 'kernel_regularizer': regularizer},
                                                              ({'rate': 0.5}, 'dropout'),
                                                              {'units': 1, 'activation': 'sigmoid'}])

            
            # test_model.compile_nn_model(loss='binary_crossentropy', metrics=['accuracy'])
                # hist = test_model.train_model(X_train.values, y_train.values.astype(np.float), epochs=50, verbose=0, class_weight=class_weights)
                # print('Train evaluation')
                # test_model.evaluate_model(X_train.values, y_train.values.astype(np.float))
                # print('\n\nTest evaluation')
                # test_model.evaluate_model(X_test.values, y_test.values.astype(np.float))
                #
            
            test_model.compile_nn_model(loss='binary_crossentropy', metrics=['accuracy'])


            training_generator = BalancedBatchGenerator(X_tests[i], y_tests[i], sampler=NearMiss(), batch_size=10, random_state=42)
            gen = generator(training_generator)
            l_gen = len([i for i in generator(training_generator)])
            # hist = test_model.model.fit_generator(generator=gen, steps_per_epoch=l_gen, epochs=10, verbose=1)

            #w = get_weights(TITLE, data_loader)
            w = class_weight.compute_class_weight('balanced',
                                                              np.unique(y_trains[i]),
                                                              y_trains[i])
            weights_dict = {}
            for i in range(len(w)):
                weights_dict[i] = w[i]

            hist = test_model.train_model(X_trains[i], y_trains[i].astype(np.float), epochs=20, verbose=1, class_weight=weights_dict)
            # hist = test_model.train_model(X_trains[i], y_trains[i], epochs=10, verbose=1, class_weight=weights_dict)
            print('Train evaluation')
            test_model.evaluate_model(X_tests[i], y_tests[i])
            print('\n\nTest evaluation')
            test_model.evaluate_model(X_tests[i], y_tests[i].astype(np.float))


            y_score = test_model.model.predict_proba(X_tests[i])
            y_pred = (test_model.model.predict(X_tests[i]) > 0.5).astype(int)

            # save the y_test and y_score
            if nn_y_test_from_all_iter is None:
                nn_y_test_from_all_iter = y_tests[i]
                nn_y_score_from_all_iter = y_score
                nn_y_pred_from_all_iter = y_pred

            else:
                nn_y_test_from_all_iter = np.append(nn_y_test_from_all_iter, y_tests[i])
                nn_y_score_from_all_iter = np.append(nn_y_score_from_all_iter, y_score)
                nn_y_pred_from_all_iter = np.append(nn_y_pred_from_all_iter, y_pred)


        print('\n *** NN **** \n')
        c_mat = np.array(confusion_matrix(nn_y_test_from_all_iter, nn_y_pred_from_all_iter))
        print(c_mat)
        print(classification_report(nn_y_test_from_all_iter, nn_y_pred_from_all_iter))
        if TITLE != "Allergy_type_task":  # not multi-class
            fpr, tpr, thresholds, nn_roc_auc = roc_auc(nn_y_test_from_all_iter, nn_y_score_from_all_iter, visualize=True,
                                                   graph_title='NN\n' + TITLE.replace("_", " "))
            # plt.show()
            plt.savefig('NN_AUC_' + TITLE + ".png")

        names = get_confusin_matrix_names(data_loader, TITLE)
        confusion_matrix_average, confusion_matrix_acc = edit_confusion_matrix(TITLE, c_mat, data_loader, "NN", names)
        print_confusion_matrix(confusion_matrix_average, names, confusion_matrix_acc, "NN", TITLE)

        print('******************* \n')
        """

def get_X_y_for_nni( TITLE = "Health_task", PRINT = False, REG = False, RHOS = False,only_single_allergy_type = False):
    os.chdir(os.path.join("..", "..", "..", "allergy"))
    data_loader = AllergyDataLoader(TITLE, PRINT, REG, WEIGHTS=True, ANNA_PREPROCESS=False)
    W_CON, ids, tag_map, task_name = get_learning_data(TITLE, data_loader, only_single_allergy_type)
    id_to_features_map = data_loader.get_id_to_features_map
    X = [id_to_features_map[id] for id in ids]
    y = [tag_map[id] for id in ids]

    return np.array(X), np.array(y)

if __name__ == "__main__":
    """
    # Pre process - mostly not needed
    PRINT = False  # do you want printing to screen?
    HIST = False  # drew histogram of the data after normalization
    REG = False  # run normalization to drew plots
    STAGES = False  # make tags for stages task
    RHOS = False  # calculate correlation between reaction to treatment to the tags
    """

    TITLES = ["Health_task"] #"Allergy_type_task"] #  "Milk_allergy_task"] # , "Success_task","Prognostic_task","Milk_allergy_task", ,

    # only_single_allergy_type = remove patients with more then one allergy
    for t in TITLES:
        run_learning(TITLE=t, PRINT=False, REG=False, RHOS=False, SVM=False, XGBOOST=False, NN=True, COEFF_PLOTS=True,
                     Cross_validation=5, TUNED_PAREMETERS=False, only_single_allergy_type=False)

    # create a deep neural network for better accuracy percentages

