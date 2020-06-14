import os
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score
from xgboost import XGBClassifier
import numpy as np
from LearningMethods.simple_learning_model import SimpleLearningModel
from Plot import roc_auc, calc_auc_on_flat_results, edit_confusion_matrix, print_confusion_matrix, multi_class_roc_auc


class XGBLearningModel(SimpleLearningModel):
    def __init__(self):
        super().__init__()

    def get_model_coeff(self, clf, pca_obj, pca_flag, binary_flag):  # suited for svm only
        if pca_flag:  # preformed PCA -> convert_pca_back_orig
            if binary_flag:
                c = clf.coef_.tolist()
                coefficients = c[:pca_obj.n_components]
            else:  # multi-class
                c = clf.coef_.tolist()
                coefficients = [c_[:pca_obj.n_components] for c_ in c]

        else:  # didn't preformed PCA -> no need to convert_pca_back_orig, use original coefficients
            if binary_flag:
                c = clf.coef_.tolist()
            else:  # multi-class
                c = clf.coef_.tolist()
            coefficients = [c_[:pca_obj.n_components] for c_ in c]
        return coefficients

    def create_classifiers(self, params):  # suited for xgb only
        optional_classifiers = []
        # create all possible classifiers
        for max_depth in params['max_depth']:
            for learning_rate in params['learning_rate']:
                for n_estimators in params['n_estimators']:
                    for objective in params['objective']:
                        for gamma in params['gamma']:
                            for min_child_weight in params['min_child_weight']:
                                clf = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate,
                                                    n_estimators=n_estimators, objective=objective,
                                                    gamma=gamma, min_child_weight=min_child_weight,
                                                    booster='gblinear')
                                optional_classifiers.append(clf)
        return optional_classifiers

    def fit(self, X, y, X_train_ids, X_test_ids, y_train_ids, y_test_ids, params, bacteria, task_name_title, relative_path_to_save_results, pca_obj=None):
        if not os.path.exists(os.path.join(relative_path_to_save_results, "XGBOOST")):
            os.makedirs(os.path.join(relative_path_to_save_results, "XGBOOST"))
        os.chdir(os.path.join(os.path.abspath(os.path.curdir), relative_path_to_save_results, "XGBOOST"))

        print("XGBOOST...")

        # update each classifier results in a mutual file
        xgb_results_file = Path("all_xgb_results.csv")
        if not xgb_results_file.exists():
            all_xgb_results = pd.DataFrame(columns=['LR', 'MAX-DEPTH', 'N-ESTIMATORS', 'OBJECTIVE',
                                                    'GAMMA', 'MIN-CHILD-WEIGHT', 'BOOSTER',
                                                    'TRAIN-AUC', 'TRAIN-ACC',
                                                    'TEST-AUC', 'TEST-ACC',
                                                    'PRECISION', 'RECALL'])
            all_xgb_results.to_csv(xgb_results_file, index=False)

        num_of_classes = len(set(y))
        BINARY = True if num_of_classes == 2 else False
        optional_classifiers = self.create_classifiers(params)

        for clf in optional_classifiers:
            all_xgb_results = pd.read_csv(xgb_results_file)
            clf_folder_name = "d=" + str(clf.max_depth) + "_lr=" + str(clf.learning_rate) + "_e=" + \
                              str(clf.n_estimators) + "_o=" + clf.objective + "_g=" + str(clf.gamma) + "_m=" + \
                              str(clf.min_child_weight) + "_b=" + clf.booster
            if not os.path.exists(clf_folder_name):
                os.makedirs(clf_folder_name)

            # Split the data set
            X_trains, X_tests, y_trains, y_tests, xgb_coefs = [], [], [], [], []
            xgb_y_test_from_all_iter, xgb_y_score_from_all_iter = np.array([]), np.array([])
            xgb_y_pred_from_all_iter, xgb_class_report_from_all_iter = np.array([]), np.array([])
            xgb_coefs, bacteria_coeff_average, y_train_scores, y_test_scores = [], [], [], []
            train_accuracies, test_accuracies, confusion_matrixes, y_train_preds, y_test_preds = [], [], [], [], []

            for i in range(params["K_FOLD"]):
                print('------------------------------\niteration number ' + str(i))
                X_train, X_test, y_train, y_test = np.array(X.loc[X_train_ids[i]]), np.array(X.loc[X_test_ids[i]]), np.array(y[y_train_ids[i]]), np.array(y[y_test_ids[i]])
                X_trains.append(X_train)
                X_tests.append(X_test)
                y_trains.append(y_train)
                y_tests.append(y_test)

                clf.fit(X_train, y_train)
                clf.predict_proba(X_test)
                y_score = clf.predict_proba(X_test)
                y_pred = clf.predict(X_test)
                y_test_preds.append(y_pred)
                y_test_scores.append(y_score[:, 0])
                xgb_class_report = classification_report(y_test, y_pred)
                train_pred = clf.predict(X_train)
                train_score = clf.predict_proba(X_train)
                y_train_preds.append(train_pred)
                y_train_scores.append(train_score[:, 0])

                train_accuracies.append(accuracy_score(y_train, clf.predict(X_train)))
                test_accuracies.append(accuracy_score(y_test, y_pred))  # same as - clf.score(X_test, y_test)
                confusion_matrixes.append(confusion_matrix(y_test, y_pred))

                if BINARY:
                    self.print_auc_for_iter(np.array(y_test), np.array(y_score).T[0])

                self.save_y_test_and_score(y_test, y_pred, y_score, xgb_class_report)
                # --------------------------------------------! COEFF PLOTS -----------------------------------------
                if params["create_coeff_plots"]:
                    svm_coefs, coefficients, bacteria_coeff_average = \
                        self.calc_bacteria_coeff_average(num_of_classes, pca_obj, bacteria, clf, xgb_coefs, bacteria_coeff_average)

            # --------------------------------------------! AUC -----------------------------------------
            all_y_train = np.array(y_trains).flatten()
            all_predictions_train = np.array(y_train_preds).flatten()
            y_train_scores = np.array(y_train_scores).flatten()
            all_test_real_tags = np.array(y_tests).flatten()
            all_test_pred_tags = np.array(y_test_preds).flatten()
            y_test_scores = np.array(y_test_scores).flatten()

            train_auc, test_auc, train_rho, test_rho = \
                calc_auc_on_flat_results(all_y_train, y_train_scores,
                                         all_test_real_tags, y_test_scores)

            # ----------------------------------------! CONFUSION MATRIX -------------------------------------
            print("------------------------------")
            names = params["CLASSES_NAMES"]
            confusion_matrix_average, confusion_matrix_acc = edit_confusion_matrix(confusion_matrixes,
                                                                                   "XGB", names, BINARY=BINARY)
            if BINARY:
                _, _, _, xgb_roc_auc = roc_auc(all_test_real_tags.astype(int), y_test_scores,
                                               visualize=True, graph_title='XGB\n' + task_name_title.capitalize() +
                                                                           " AUC on all iterations", save=True,
                                               folder=clf_folder_name)
                res_path = os.path.join(clf_folder_name, str(round(xgb_roc_auc, 5)))
            else:
                xgb_roc_auc = 0
                res_path = clf_folder_name

            if not os.path.exists(res_path):
                os.mkdir(res_path)

            if params["create_coeff_plots"]:
                self.plot_bacteria_coeff_average(bacteria_coeff_average, len(set(y)), params["TASK_TITLE"],
                                                 clf_folder_name,
                                                 bacteria, params["K_FOLD"], "XGB", res_path, BINARY, names)

            print_confusion_matrix(confusion_matrix_average, names, confusion_matrix_acc, "XGB", task_name_title,
                                   res_path)

            if BINARY:
                _, _, _, xgb_train_roc_auc = roc_auc(all_y_train, y_train_scores, visualize=False,
                                                     graph_title="train auc", save=False, folder=res_path)
            else:
                xgb_train_roc_auc = 0
                multi_class_roc_auc(all_y_train.astype(int), y_train_scores, names,
                                    graph_title='XGB\n' + task_name_title.capitalize() + " AUC on all iterations",
                                    save=True, folder=res_path)


            # ----------------------------------------! SAVE RESULTS -------------------------------------

            self.save_results(task_name_title, train_auc, test_auc, train_rho, test_rho, confusion_matrix_average,
                         confusion_matrix_acc,
                         train_accuracies, test_accuracies, xgb_y_score_from_all_iter, xgb_y_pred_from_all_iter,
                         xgb_y_test_from_all_iter, "XGB", res_path)

            all_xgb_results.loc[len(all_xgb_results)] = [clf.learning_rate, clf.max_depth, clf.n_estimators,
                                                         clf.objective, clf.gamma, clf.min_child_weight, clf.booster,
                                                         xgb_train_roc_auc, np.mean(train_accuracies), xgb_roc_auc,
                                                         np.mean(test_accuracies),
                                                         precision_score(all_test_real_tags.astype(int), all_test_pred_tags,  average='micro'),
                                                         recall_score(all_test_real_tags.astype(int),  all_test_pred_tags, average='micro')]
            if BINARY:
                all_xgb_results = all_xgb_results.sort_values(by=['TEST-AUC'], ascending=False)
            else:
                all_xgb_results = all_xgb_results.sort_values(by=['TEST-ACC'], ascending=False)

            all_xgb_results.to_csv(xgb_results_file, index=False)
