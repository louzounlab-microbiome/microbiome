import os
import pandas as pd
from LearningMethods.create_otu_and_mapping_files import CreateOtuAndMappingFiles
from Plot import plot_heat_map_from_df
from Plot.plot_preproccess_evaluation import plot_task_comparision
from infra_functions import apply_pca
import numpy as np
from sklearn import svm, metrics, preprocessing
from sklearn.model_selection import train_test_split, LeaveOneOut
from LearningMethods.leave_two_out import LeaveTwoOut
from LearningMethods.multi_model_learning import get_weights, read_otu_and_mapping_files


def get_train_test_auc_from_svm(otu_path, mapping_path, method="fold"):
    X, y = read_otu_and_mapping_files(otu_path, mapping_path)
    # weights = get_weights(y)
    clf = svm.SVC(kernel='linear', C=1, class_weight='balanced')
    y_trains, y_tests = [], []
    y_train_scores, y_test_scores, train_auc_list = [], [], []

    if method == "loo":
        loo = LeaveOneOut()
        for train_index, test_index in loo.split(y):
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
            y_tests.append(y_test)

            # FIT
            clf.fit(X_train, y_train)
            test_score = clf.decision_function(X_test)
            train_score = clf.decision_function(X_train)
            y_test_scores.append(test_score)

            train_auc = metrics.roc_auc_score(y_train, train_score)
            train_auc_list.append(train_auc)

        # --------------------------------------------! AUC -----------------------------------------
        all_test_real_tags = np.array(y_tests).flatten()
        y_test_scores = np.array(y_test_scores).flatten()

        test_auc = metrics.roc_auc_score(all_test_real_tags, y_test_scores)
        train_auc = np.average(train_auc_list)

    elif method == "lto":
        lto = LeaveTwoOut()
        for train_index, test_index in lto.split(y):
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
            y_tests.append(y_test)

            # FIT
            clf.fit(X_train, y_train)
            test_score = clf.decision_function(X_test)
            train_score = clf.decision_function(X_train)
            y_test_scores.append(test_score)

            train_auc = metrics.roc_auc_score(y_train, train_score)
            train_auc_list.append(train_auc)

        # --------------------------------------------! AUC -----------------------------------------
        all_test_real_tags = np.array(y_tests).flatten()
        y_test_scores = np.array(y_test_scores).flatten()

        test_auc = metrics.roc_auc_score(all_test_real_tags, y_test_scores)
        train_auc = np.average(train_auc_list)

    else:  # method == "fold"
        for i in range(10):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            y_trains.append(y_train)
            y_tests.append(y_test)

            # FIT
            clf.fit(X_train, y_train)
            test_score = clf.decision_function(X_test)
            train_score = clf.decision_function(X_train)
            y_train_scores.append(train_score)
            y_test_scores.append(test_score)

        # --------------------------------------------! AUC -----------------------------------------
        all_y_train = np.array(y_trains).flatten()
        y_train_scores = np.array(y_train_scores).flatten()
        all_test_real_tags = np.array(y_tests).flatten()
        y_test_scores = np.array(y_test_scores).flatten()

        test_auc = metrics.roc_auc_score(all_test_real_tags, y_test_scores)
        train_auc = metrics.roc_auc_score(all_y_train, y_train_scores)

    return train_auc, test_auc


def microbiome_preprocess(pca_list, tax_list, tag_list, rho_pca_plots=False, evaluate=False):
    for pca_n in pca_list:
        for tax in tax_list:
            for tag in tag_list:
                # parameters for preprocess
                preprocess_prms = {'taxonomy_level': tax, 'taxnomy_group': 'mean', 'epsilon': 0.1, 'normalization': 'log',
                                   'z_scoring': 'row', 'norm_after_rel': '', 'std_to_delete': 0, 'pca': pca_n}

                mapping_file = CreateOtuAndMappingFiles("otu.csv", tag + "_tag.csv")
                mapping_file.preprocess(preprocess_params=preprocess_prms, visualize=False)

                if rho_pca_plots:
                    folder = "preprocess_plots_" + tag + "_tag_tax_" + str(tax) + "_pca_" + str(pca_n)
                    mapping_file.rhos_and_pca_calculation(tag, preprocess_prms['taxonomy_level'], preprocess_prms['pca'],
                                                          os.path.join(folder, "rhos"), os.path.join(folder, "pca"))

                otu_path, tag_path, pca_path = mapping_file.csv_to_learn(tag + '_task', tag + "_tax_" + str(tax) + "_csv_files",
                                                                         tax, pca_n)
                print(otu_path)

    # compere tax level and number of pca component using certain svm model and compere results
    if evaluate:
        microbiome_preprocess_evaluation(pca_options=pca_list,
                                         tax_options=tax_list,
                                         tag_options=tag_list)


def extra_features_preprocess(pca_list, tag_list, id_col_name, folder, df_path, results_path, evaluate=False):
    if not os.path.exists(folder):
        os.mkdir(folder)
    df = pd.read_csv(df_path)
    df = df.rename(columns={id_col_name: "ID"})
    df = df.set_index("ID")
    for pca_n in pca_list:
        for tag in tag_list:
            df_after_pca, pca_obj, _ = apply_pca(df, pca_n)
            file_name = "Extra_features_" + tag + "_task_pca_" + str(pca_n) + ".csv"
            df_after_pca.to_csv(os.path.join(folder, file_name))
            print("done extra features for " + tag + "- pca=" + str(pca_n))

    # compere number of pca component using certain svm model and compere results
    if evaluate:
        extra_features_preprocess_evaluation(folder,
                                             tag_options=tag_list,
                                             pca_options=pca_list,
                                             results_path=results_path)


def microbiome_preprocess_evaluation(tag_options, pca_options, tax_options):
    results_path = "preprocess_evaluation_plots"
    for tag in tag_options:
        task_results = {}
        for tax in tax_options:
            for pca_n in pca_options:
                folder = tag + "_tax_" + str(tax) + "_csv_files"
                otu_path = os.path.join(folder, 'OTU_merged_' + tag + "_task_tax_level_" + str(tax) + '_pca_' + str(
                                            pca_n) + '.csv')
                tag_path = os.path.join(folder, 'Tag_file_' + tag + '_task.csv')
                train_auc, test_auc = get_train_test_auc_from_svm(otu_path, tag_path, method="lto")
                task_results[(tax, pca_n)] = (train_auc, test_auc)
        plot_task_comparision(task_results, results_path, tag + "_preprocess_evaluation_plots", pca_options, tax_options)


def extra_features_preprocess_evaluation(folder, tag_options, pca_options, results_path):
    for tag in tag_options:
        task_results = {}
        for pca_n in pca_options:
            features_path = os.path.join(folder, "Extra_features_" + tag + "_task_pca_" + str(pca_n) + ".csv")
            tag_path = os.path.join(tag + '_tag.csv')
            train_auc, test_auc = get_train_test_auc_from_svm(features_path, tag_path, method="fold")
            task_results[pca_n] = (train_auc, test_auc)
        plot_task_comparision(task_results, results_path, tag + "_preprocess_evaluation_plots", pca_options)


def fill_and_normalize_extra_features(extra_features_df):
    for col in extra_features_df.columns:
        extra_features_df[col] = extra_features_df[col].replace(" ", np.nan)
        average = np.average(extra_features_df[col].dropna().astype(float))
        extra_features_df[col] = extra_features_df[col].replace(np.nan, average)
        # z-score on columns

    extra_features_df[:] = preprocessing.scale(extra_features_df, axis=1)
    return extra_features_df


def create_na_distribution_csv(df, sub_df_list, col_names, title, plot=True):
    folder = "na"

    results_df = pd.DataFrame(columns=["column_name"] + col_names)
    for col in df.columns:
        na_values_number = []
        na_values_percent = []
        for sub_data_df in sub_df_list:
            na_values = sub_data_df[col].isna().sum()
            na_values_number.append(na_values)
            na_values_percent.append(round(na_values / len(sub_data_df), 4))
        # results_df.loc[len(results_df)] = [col + ";na_number"] + na_values_number
        results_df.loc[len(results_df)] = [col] + na_values_percent

    results_df = results_df.set_index("column_name")
    results_df.to_csv(os.path.join(folder, title + "_feature_na_distribution.csv"))

    if plot:
        start = 0
        margin = 18
        for i in range(int(len(results_df) / margin) + 1):
            plot_heat_map_from_df(results_df[start:min(start + margin, len(results_df))],
                                  title.replace("_", " ") + " Feature NA Distribution Part " + str(i),
                                  "groups", "features", folder, pos_neg=False)
            start += margin


def create_csv_from_column(df, col_name, id_col_name, title):
    tag_df = pd.DataFrame(columns=["ID", "Tag"])
    tag_df["ID"] = df[id_col_name]
    tag_df["Tag"] = df[col_name]
    tag_df = tag_df.set_index("ID").replace(" ", "nan")
    tag_df.to_csv(title)


def create_tags_csv(df, id_col_name, tag_col_and_name_list):
    for tag, name in tag_col_and_name_list:
        create_csv_from_column(df, tag, name + "_tag.csv", id_col_name)

