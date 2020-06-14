from Plot.plot_rho import draw_rhos_calculation_figure
from Projects.GVHD_BAR.load_merge_otu_mf import OtuMfHandler
from Preprocess.preprocess import preprocess_data
from os.path import join
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, explained_variance_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
import os
import pickle
import numpy as np
from xgboost import XGBRegressor, DMatrix, train
from sklearn.metrics import mean_squared_error
import math
import warnings
import matplotlib.pyplot as plt
import datetime
import pprint
import xgboost as xgb
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

from allergy.allergy_main import calc_auc_on_joined_results
from Plot.plot_auc import roc_auc, multi_class_roc_auc
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def prepare_data(tax_file, map_file, preform_z_scoring=True, taxnomy_level=6, n_components=20):
    OtuMf = OtuMfHandler(join(SCRIPT_DIR, tax_file), join(SCRIPT_DIR, map_file), from_QIIME=False,
                         id_col='#OTU ID')

    preproccessed_data = preprocess_data(OtuMf.otu_file, preform_z_scoring=preform_z_scoring, visualize_data=True,
                                         taxnomy_level=taxnomy_level,
                                         preform_taxnomy_group=True)

    # otu_after_pca_wo_taxonomy, _, _ = apply_pca(preproccessed_data, n_components=n_components, visualize=True)

    return OtuMf, preproccessed_data


def create_data_maps(mapping_file):
    # look at exp1 and exp14
    samples_ids = mapping_file.index
    experiment = mapping_file['Experiment']
    id_to_experiment_map = {id: p for id, p in zip(samples_ids, experiment)}

    experiments = set(id_to_experiment_map.values())
    experiments_to_ids_map = {e: [] for e in experiments}
    for i, experiment in id_to_experiment_map.items():
        experiments_to_ids_map[experiment].append(i)

    time_point = mapping_file['TimePointNum']
    id_to_time_point_map = {id: p for id, p in zip(samples_ids, time_point)}

    time_points = set(id_to_time_point_map.values())
    time_points_to_ids_map = {p: [] for p in time_points}
    for i, point in id_to_time_point_map.items():
        time_points_to_ids_map[point].append(i)

    exp1 = experiments_to_ids_map[1]
    exp14 = experiments_to_ids_map[14]
    time_point_5 = time_points_to_ids_map[5]
    ids = [i for i in exp1 + exp14 if i in time_point_5]

    # ------------------------------------- tumour load -------------------------------------
    tumor_load = mapping_file['tumor_load']
    id_to_tumor_load_map = {id: t_l for id, t_l in zip(samples_ids, tumor_load)}
    # sub set ids
    id_to_tumor_load_map = {key: val for key, val in id_to_tumor_load_map.items() if key in ids}
    tumer_ids = [i for i in ids if str(id_to_tumor_load_map[i]) != "nan"]
    # len([1 for i in id_to_tumor_load_map.values() if str(i) == "nan"]) / len(id_to_tumor_load_map)

    # ------------------------------------- immunology -------------------------------------
    # cell_spleen, MDSC_GR1_spleen, MFI_zeta_spleen, cell_BM, MDSC_GR1_bm
    cell_spleen = mapping_file['cell_spleen']
    id_to_cell_spleen_map = {id: c for id, c in zip(samples_ids, cell_spleen)}

    MDSC_GR1_spleen = mapping_file['MDSC_GR1_spleen']
    id_to_MDSC_GR1_spleen_map = {id: c for id, c in zip(samples_ids, MDSC_GR1_spleen)}

    MFI_zeta_spleen = mapping_file['MFI_zeta_spleen']
    id_to_MFI_zeta_spleen_map = {id: c for id, c in zip(samples_ids, MFI_zeta_spleen)}

    cell_BM = mapping_file['cell_BM']
    id_to_cell_BM_map = {id: c for id, c in zip(samples_ids, cell_BM)}

    MDSC_GR1_bm = mapping_file['MDSC_GR1_bm']
    id_to_MDSC_GR1_bm_map = {id: c for id, c in zip(samples_ids, MDSC_GR1_bm)}

    # ------------------------------------- mice id -------------------------------------
    all_ids = [i for i in exp1 + exp14]
    cage_num = mapping_file['CageNum']
    mouse = mapping_file['Mouse']
    id_to_mouse_id_map = {id: (str(cage) + "_" + str(mos)).replace(".0", "") for id, cage, mos in
                          zip(all_ids, cage_num, mouse)}
    mouse_ids = set(id_to_mouse_id_map.values())
    mouse_ids_to_ids_map = {p: [] for p in mouse_ids}
    for i, point in id_to_mouse_id_map.items():
        mouse_ids_to_ids_map[point].append(i)
    mouse_ids_to_ids_map = {key: val for key, val in mouse_ids_to_ids_map.items() if "nan" not in key}

    # ------------------------------------- tumour load learning data-------------------------------------
    # add samples from other times, tag them with time 5 tumor size
    all_ids_to_tumor_load_map = {i: id_to_tumor_load_map[i] for i in tumer_ids if str(id_to_tumor_load_map[i]) != "nan"}
    # for each mouse, tag all entries as time 5 tag
    for mouse, mouse_entries in mouse_ids_to_ids_map.items():
        mouse_tag = None
        for entry in mouse_entries:
            if entry in tumer_ids:
                mouse_tag = id_to_tumor_load_map[entry]
                break
        if mouse_tag:
            for entry in mouse_entries:
                if entry not in all_ids_to_tumor_load_map.keys():
                    all_ids_to_tumor_load_map[entry] = mouse_tag

    return tumer_ids, id_to_tumor_load_map, ids, id_to_cell_spleen_map, id_to_MDSC_GR1_spleen_map, \
           id_to_MFI_zeta_spleen_map, id_to_cell_BM_map, id_to_MDSC_GR1_bm_map, all_ids_to_tumor_load_map



def calc_correlations(ids, maps_list, titles_list, preproccessed_data, taxnomy_level, folder="correlations"):
    for att_map, att_title in zip(maps_list, titles_list):
        draw_rhos_calculation_figure(att_map, preproccessed_data, att_title, taxnomy_level, num_of_mixtures=10,
                                     ids_list=ids, save_folder=join(folder, str(taxnomy_level)))


def get_svm_clf():
    pass


def get_xgb_clf():
    xgb_clf = XGBRegressor(colsample_bytree=0.4,
                 gamma=0,
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=10000,
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42)
    return xgb_clf


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
                                         'reg_alpha': [0.75],
                                         'reg_lambda': [0.45],
                                         'subsample': [0.6],
                                         'seed': [42]}]

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
            xgdmat = DMatrix(X_trains[iter_num], y_trains[iter_num])
            our_params = {'eta': 0.1, 'seed': 0, 'subsample': 0.8, 'colsample_bytree': 0.8, 'objective': 'reg:linear',
                          'max_depth': 3, 'min_child_weight': 1}
            final_gb = train(our_params, xgdmat)
            tesdmat = DMatrix(X_tests[iter_num])
            y_pred = final_gb.predict(tesdmat)
            print(y_pred)

            testScore = math.sqrt(mean_squared_error(y_tests[iter_num], y_pred))
            print(testScore)



            # -------------------------------------------------------------------------------- #

            clf = get_xgb_clf()

            classes_sum = [np.sum(np.array(y_trains[iter_num]) == unique_class) for unique_class in
                           np.unique(np.array(y_trains[iter_num]))]
            classes_ratio = [1 - (a / sum(classes_sum)) for a in classes_sum]
            tag_val_to_ratio_map = {tag: classes_ratio[i] for i, tag in enumerate(np.unique(np.array(y_trains[iter_num])))}
            weights = [tag_val_to_ratio_map[a] for a in np.array(y_trains[iter_num])]  # classes_ratio

            clf.fit(np.array(X_trains[iter_num]), np.array(y_trains[iter_num]), sample_weight=weights)
            # clf.predict_proba(X_tests[iter_num])
            # y_score = clf.predict_proba(X_tests[iter_num])  # what is this for?
            y_pred = clf.predict(X_tests[iter_num])
            y_test_preds.append(y_pred)
            s = explained_variance_score(y_pred, y_tests[iter_num])
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


def plot_y_diff_plot(title, y, y_score):
    plt.figure(figsize=(5, 5))
    m = max(y.max(), y_score.max())
    data = {'a': np.array(y),
            'b': np.array(y_score)}
    plt.scatter('a', 'b', data=data)  # , c='c', s='d'
    plt.axis([-0.05, m + 0.05, -0.05, m + 0.05])
    plt.xlabel('real size')
    plt.ylabel('predicted size')
    plt.title(title)
    # plt.show()
    plt.savefig(title.replace(" ", "_") + ".svg", bbox_inches="tight", format='svg')


def find_xgb_best_model(X, y):
    print('Select Model...')
    start_time = datetime.datetime.now()
    xgb_clf = xgb.XGBRegressor()
    parameters = {'n_estimators': [1000, 2000, 3000, 4000], 'max_depth': [3, 5, 7, 9, 11],
                  'learning_rate': [0.1, 0.05, 0.01]}  #, 'objective': 'reg:squarederror'}
    grid_search = GridSearchCV(estimator=xgb_clf, param_grid=parameters, cv=10)  #, n_jobs=-1)
    print("parameters:")
    pprint.pprint(parameters)
    grid_search.fit(X, y)
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
    end_time = datetime.datetime.now()
    print('Select Done..., Time Cost: %d' % ((end_time - start_time).seconds))
    print("!")


if __name__ == "__main__":
    if False:
        with open("exported_feature-table_for_YoramL.txt", "r") as feature_table:
            data = feature_table.readlines()
            sep_data = []
            for entry in data:
                entry = entry.strip("\n")
                info = entry.split("\t")
                sep_data.append(info)
            cols = sep_data[0]
            sep_data.pop(0)
            df = pd.DataFrame(columns=cols, data=sep_data).set_index("#OTU ID")
        taxonomy = pd.read_csv("taxonomy.csv").set_index("#SampleID")
        matching_tax = [taxonomy.loc[i]["taxonomy"] for i in df.index]
        df['taxonomy'] = matching_tax
        df.to_csv("feature-table.csv")
    feature_file = "feature-table.csv"
    map_file = 'mapping file with data Baniyahs Merge.csv'
    tax_level = 5

    OtuMf, preproccessed_data = prepare_data(feature_file, map_file, preform_z_scoring=True, taxnomy_level=tax_level)
    # preproccessed_data = preproccessed_data.drop(["Unassigned"], axis=1)
    bacteria = preproccessed_data.columns
    otu_file = OtuMf.otu_file_wo_taxonomy
    mapping_file = OtuMf.mapping_file
    # get [antibiotics, chemo] one hot for each sample, bacteria data, time and weight gain data
    tumer_ids, id_to_tumor_load_map, exp_1_14_time_5_ids, id_to_cell_spleen_map, id_to_MDSC_GR1_spleen_map, \
    id_to_MFI_zeta_spleen_map, id_to_cell_BM_map, id_to_MDSC_GR1_bm_map, all_ids_to_tumor_load_map = create_data_maps(
        mapping_file)

    correlations = True
    if correlations:
        # immunology correlations
        calc_correlations(exp_1_14_time_5_ids,
                          [id_to_cell_spleen_map, id_to_MDSC_GR1_spleen_map, id_to_MFI_zeta_spleen_map,
                           id_to_cell_BM_map, id_to_MDSC_GR1_bm_map],
                          ["Cell_spleen", "MDSC_GR1_spleen", "MFI_zeta_spleen",
                           "Cell_Bone_Marrow", "MDSC_GR1_BM"], preproccessed_data, tax_level)

        # tumor size correlations
        draw_rhos_calculation_figure(id_to_tumor_load_map, preproccessed_data, "Tumor_Size", tax_level,
                                     num_of_mixtures=10,
                                     ids_list=tumer_ids, save_folder=join("correlations", str(tax_level)))

    # predict tumor load from microbiome day 5 in column (R)
    tumor_pred_ids = list(all_ids_to_tumor_load_map.keys())
    y = list(all_ids_to_tumor_load_map.values())
    X = preproccessed_data.loc[tumor_pred_ids].as_matrix()
    nan_idx = []
    for i, x in enumerate(X):
        if str(x[0]) == "nan":
            nan_idx.append(i)
    nan_idx.reverse()
    for i in nan_idx:
        y.pop(i)
        X = np.delete(X, i, axis=0)


    # 37 -> nan_id = "5.1.D44.Exp14"
    # nn_main(X, y, "tumor_load_prediction_task", "tumor_load", X.shape[1], 40, 20, 1)
    # find_xgb_best_model(X, y)
    # learning_rate: 0.1
    # max_depth: 3
    # n_estimators: 1000

    X_train, X_test, y_train, y_test = train_test_split(X, np.array(y), test_size=0.2, random_state=42)
    xgdmat = xgb.DMatrix(X_train, y_train)
    our_params = {'objective': 'reg:squarederror', 'n_estimators': 1000,
                  'max_depth': 3, 'min_child_weight': 1, 'learning_rate': 0.1}
    final_gb = xgb.train(our_params, xgdmat)
    tesdmat = xgb.DMatrix(X_test)
    y_pred = final_gb.predict(tesdmat)
    print(y_pred)
    plot_y_diff_plot("XGB\nReal Tumer Sizes vs. Predicted Sizes (test set)", y_test, y_pred)


    """
    # remove '2' tag
    idx = y.index(2)
    y.pop(idx)
    y = [int(i) for i in y]
    X = np.delete(X, 29, axis=0)
    print(len(y))
    print(len(X))
    """
    # tumor_learning(X, y, "Tumor Size Prediction", SVM=False, XGBOOST=True, Cross_validation=10, TUNED_PAREMETERS=False)


