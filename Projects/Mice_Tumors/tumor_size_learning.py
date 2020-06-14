from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
import os
import pickle
import numpy as np
from xgboost import XGBRegressor

from allergy.allergy_main import calc_auc_on_joined_results
from Plot.plot_auc import roc_auc, multi_class_roc_auc


def get_svm_clf():
    pass


def get_xgb_clf():
    pass


def dafi():
    return 5


def tumor_learning(X, y, task_name, SVM, XGBOOST, Cross_validation, TUNED_PAREMETERS):
    print("learning..." + task_name)
    # ----------------------------------------------! SVM ------------------------------------------------
    # Set the parameters by cross-validation
    # multi_class =”crammer_singer”
    if SVM:
        print("SVM...")
        if TUNED_PAREMETERS:
            svm_tuned_parameters = [{'kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
                                 'gamma': ['auto', 'scale'],
                                 'C': [0.01, 0.1, 1, 10, 100, 1000]}]

            svm_clf = GridSearchCV(svm.SVR(class_weight='balanced'), svm_tuned_parameters, cv=5,
                                   scoring='roc_auc', return_train_score=True)

            svm_clf.fit(X, y)
            print(svm_clf.best_params_)
            print(svm_clf.best_score_)
            print(svm_clf.cv_results_)

            svm_results = pd.DataFrame(svm_clf.cv_results_)
            svm_results.to_csv(os.path.join(task_name, "svm_all_results_df_" + task_name + ".csv"))
            pickle.dump(svm_results, open(os.path.join(task_name, "svm_all_results_df_" + task_name + ".pkl"), 'wb'))
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
            clf = get_svm_clf(task_name)

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

        all_y_train, all_predictions_train, all_test_real_tags, all_test_pred_tags, train_auc, test_auc, train_rho,\
        test_rho = calc_auc_on_joined_results(Cross_validation, y_trains, y_train_preds, y_tests, y_test_preds)

        print("\n------------------------------\n")
        try:
            _, _, _, svm_roc_auc = roc_auc(svm_y_test_from_all_iter.astype(int), svm_y_score_from_all_iter,
                                           visualize=True, graph_title='SVM\n' + task_name.capitalize() +
                                           " AUC on all iterations", save=True, folder=task_name)
        except ValueError:
            print(task_name + "multiclass format is not supported at roc auc calculation")
            names = []
            multi_class_roc_auc(svm_y_test_from_all_iter.astype(int), svm_y_score_from_all_iter, names,
                                           graph_title='SVM\n' + task_name.capitalize() + " AUC on all iterations",
                                           save=True, folder=task_name)

        print("svm final results: " + task_name)
        print("train_auc: " + str(train_auc))
        print("test_auc: " + str(test_auc))
        print("train_rho: " + str(train_rho))
        print("test_rho: " + str(test_rho))
        print("confusion_matrix_average: ")

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

    # ----------------------------------------------! XGBOOST ------------------------------------------------

    if XGBOOST:
        print("XGBOOST...")
        if TUNED_PAREMETERS:
            xgboost_tuned_parameters = [{'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.2],
                                         'objective': ['binary:logistic'],
                                         'n_estimators': [1000],
                                         'max_depth': range(3, 10),
                                         'min_child_weight': range(1, 12),
                                         'gamma': [0.0, 0.1, 0.2, 0.3, 0.4, 1, 3, 6, 9],
                                         'colsample_bytree': [0.4],
                                         'reg_alpha': 0.75,
                                         'reg_lambda': 0.45,
                                         'subsample': 0.6,
                                         'seed': 42}]

            xgb_clf = GridSearchCV(XGBRegressor(), xgboost_tuned_parameters, cv=5,
                                   scoring='roc_auc', return_train_score=True)

            xgb_clf.fit(np.asarray(X), np.asarray(y))  # xgb_clf.fit(X, y)
            print(xgb_clf.best_params_)
            print(xgb_clf.best_score_)
            print(xgb_clf.cv_results_)
            xgb_results = pd.DataFrame(xgb_clf.cv_results_)
            xgb_results.to_csv(os.path.join(task_name, "xgb_all_results_df_" + task_name + ".csv"))
            pickle.dump(xgb_results, open(os.path.join(task_name, "xgb_all_results_df_" + task_name + ".pkl"), 'wb'))

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

            clf = get_xgb_clf(task_name)

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
