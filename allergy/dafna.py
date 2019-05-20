import copy
import pickle
import random

from scipy.stats import spearmanr
from sklearn import svm, metrics
from scipy import stats
from sklearn.metrics import classification_report, roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from tensorflow.python.keras import regularizers
from xgboost import XGBClassifier
from allergy.data_loader import AllergyDataLoader
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from infra_functions import tf_analaysis
from infra_functions.general import convert_pca_back_orig
import tensorflow as tf
import seaborn as sns
from infra_functions.general import apply_pca, use_spearmanr, use_pearsonr, roc_auc, convert_pca_back_orig, draw_horizontal_bar_chart, draw_rhos_calculation_figure


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
n_components = 20


def roc_auc(y_test, y_score, verbose=False, visualize=False, graph_title='ROC Curve', save=False):
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    print(f'ROC AUC = {roc_auc}')
    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(fpr, tpr)
        plt.title(f'{graph_title}\nroc={roc_auc}')
        plt.xlabel('fpr')
        plt.ylabel('tpr')
        plt.show()
        if save:
            plt.savefig(graph_title.replace(" ", "_").replace("\n", "_") + ".png")
            plt.close()
    return fpr, tpr, thresholds, roc_auc

# ----------------------------------------------! learning methods ------------------------------------------------


def get_learning_data(title, data_loader):
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
        ids = data_loader.get_id_wo_non_and_egg_allergy_type_list
        # ids = [i for i in ids if i in time_zero]
        tag_map = data_loader.get_id_to_allergy_number_type_tag_map
        task_name = 'allergy type task'


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
        clf = svm.LinearSVC(dual=False, C=1.0, multi_class='ovr', class_weight='balanced', max_iter=10000000)

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


def create_coeff_plots_by_alogorithm(bacteria_coeff_average, bacterias, task_name, algorithm):
    averages = [0] * len(bacteria_coeff_average[0])
    for iter_num, l in enumerate(bacteria_coeff_average):
        for i, num in enumerate(l):
            averages[i] = averages[i] + num
    averages = [avg/len(bacteria_coeff_average) for avg in averages]
    min_rho = min(averages)
    max_rho = max(averages)
    rho_range = max_rho - min_rho
    # we want to take those who are located on the sides of most (center 98%) of the mixed tags entries
    # there for the bound isn't fixed, and is dependent on the distribution of the mixed tags

    lower_bound = min_rho + (rho_range * 0.3)
    upper_bound = max_rho - (rho_range * 0.3)
    significant_bacteria_and_rhos = []

    for i, bact in enumerate(bacterias):
        if averages[i] < lower_bound or averages[i] > upper_bound:  # significant
            significant_bacteria_and_rhos.append([bact, averages[i]])

    significant_bacteria_and_rhos.sort(key=lambda s: s[1])
    with open(algorithm + "_" + task_name + "_significant_bacteria_coeff_avarage_10_runs.txt", "w") as file:
        for s in significant_bacteria_and_rhos:
            file.write(str(s[1]) + "," + str(s[0]) + "\n")

    # get the significant bacteria full names
    features = [s[0] for s in significant_bacteria_and_rhos]
    # short_feature_names = [f.split(";")[-1] if len(f.split(";")[-1]) > 4 else f.split(";")[-2] for f in features]

    # extract the last meaningful name - long multi level names to the lowest level definition
    short_feature_names = []
    for f in features:
        i = 1
        while len(f.split(";")[-i]) < 5:  # meaningless name
            i += 1
        short_feature_names.append(f.split(";")[-i])

    left_padding = 0.4
    fig, ax = plt.subplots()
    y_pos = np.arange(len(significant_bacteria_and_rhos))
    c = [s[1] for s in significant_bacteria_and_rhos]
    coeff_color = []
    for x in c:
        if x >= 0:
            coeff_color.append('green')
        else:
            coeff_color.append('red')
    # coeff_color = ['blue' for x in data >= 0]
    ax.barh(y_pos, c, color=coeff_color)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_feature_names)
    plt.yticks(fontsize=10)
    plt.title(algorithm + " - " + task_name + " average coef for " + str(iter_num) + "/10 run")
    # ax.set_ylabel(ylabel)
    ax.set_xlabel("Coeff value")
    fig.subplots_adjust(left=left_padding)
    plt.show()
    plt.savefig("xgb_bacteria_pos_neg_correlation_at_ " + task_name + "_avarage_" + str(iter_num) + "_run.png")


"""
def create_general_coeff_plots(entire_W, bacteria_coeff, task_name, iter_num, algorithm):
    # select significance bactria by coefficients
    min_rho = min(entire_W)
    max_rho = max(entire_W)
    rho_range = max_rho - min_rho
    lower_bound = min_rho + (rho_range * 0.3)
    upper_bound = max_rho - (rho_range * 0.3)
    significant_bacteria_and_rhos = []

    bacterias = bacteria_coeff['Taxonome'].tolist()
    coefficients = bacteria_coeff['Coefficients'].tolist()

    for i, bact in enumerate(bacterias):
        if coefficients[i] < lower_bound or coefficients[i] > upper_bound:  # significant
            significant_bacteria_and_rhos.append([bact, coefficients[i]])

    significant_bacteria_and_rhos.sort(key=lambda s: s[1])
    with open(algorithm + "_" + task_name + "_significant_bacteria_coeff.txt", "w") as file:
        for s in significant_bacteria_and_rhos:
            file.write(str(s[1]) + "," + str(s[0]) + "\n")
    # draw the distribution of real rhos vs. mixed rhos

    # plot
    features = [s[0] for s in significant_bacteria_and_rhos]
    left_padding = 0.4
    fig, ax = plt.subplots()
    y_pos = np.arange(len(significant_bacteria_and_rhos))
    c = [s[1] for s in significant_bacteria_and_rhos]
    coeff_color = []
    for x in c:
        if x >= 0:
            coeff_color.append('green')
        else:
            coeff_color.append('red')
    ax.barh(y_pos, c, color=coeff_color)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    plt.yticks(fontsize=10)
    plt.title(task_name + " coef " + str(iter_num))
    # ax.set_ylabel(ylabel)
    ax.set_xlabel("Coeff value")
    fig.subplots_adjust(left=left_padding)
    # set_size(5, 5, ax)
    plt.show()
    # plt.savefig(algorithm + _bacteria_pos_neg_correlation_at_ " + task_name + "_" + str(iter_num) + ".png")

"""
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
        """
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot of a ROC curve for a specific class
        for i in range(n_classes):
            plt.figure()
            plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic example')
            plt.legend(loc="lower right")
            plt.show()
        """



    return all_y_train, all_predictions_train, all_test_real_tags, all_test_pred_tags,\
           train_auc, test_auc, train_rho, test_rho


def print_confusion_matrix(confusion_matrix, class_names, acc, algorithm, figsize=(10, 7), fontsize=14):
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True)  # , fmt="d"
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.title(TITLE + " Confusion Matrix Heat Map - Accuracy = " + str(acc))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.show()
    plt.savefig(TITLE + "_" + algorithm + "confusion_matrix_heat_map.png")
    return fig


def edit_confusion_matrix(title, confusion_matrixes, data_loader):
    if title in ["Success_task",  "Health_task", "Prognostic_task", "Milk_allergy_task"]:
        if title == "Milk_allergy_task":
            names = ['Other', 'Milk']
        elif title == "Health_task":
            names = ['Allergic', 'Healthy']
        elif title in ["Success_task", "Prognostic_task"]:
            names = ['No', 'Yes']

        x1 = np.mean([c[0][0] for c in list(confusion_matrixes)])
        x2 = np.mean([c[0][1] for c in list(confusion_matrixes)])
        x3 = np.mean([c[1][0] for c in list(confusion_matrixes)])
        x4 = np.mean([c[1][1] for c in list(confusion_matrixes)])

        # calc_acc
        sum = x1 + x2 + x3 + x4
        x1 = x1 / sum
        x2 = x2 / sum
        x3 = x3 / sum
        x4 = x4 / sum
        acc = x1 + x4
        print("acc = " + str(acc))
        mat = [[x1, x2], [x3, x4]]
        mat.append([acc])
        df = pd.DataFrame(data=mat)
        df.columns = names
        df.index = names + ["acc"]
        confusion_matrix_average = df  # "[[" + str(x1) + ", " + str(x2) + "], [" + str(x3) + ", " + str(x4) + "]]"

        # random classification and acc calculation in order to validate the results.
        #TODO

        return confusion_matrix_average, acc, names

    elif title in ["Allergy_type_task"]:
        tag_to_allergy_type_map = data_loader.get_tag_to_allergy_type_map
        allergy_type_to_instances_map = data_loader.get_allergy_type_to_instances_map
        allergy_type_to_weight_map = data_loader.get_allergy_type_to_weight_map
        allergy_type_weights = list(allergy_type_to_weight_map.values())
        types = []
        for key in range(len(tag_to_allergy_type_map.keys())):
            types.append(tag_to_allergy_type_map.get(key))

        final_matrix = []
        matrix = list(copy.deepcopy(confusion_matrixes[0]))
        for l in matrix:
            final_matrix.append(list(l))

        # set a final empty matrix
        for i in range(len(final_matrix)):
            for j in range(len(final_matrix)):
                final_matrix[i][j] = 0
        # fill it with the sum of all matrixes
        for mat in confusion_matrixes:
            for i in range(len(final_matrix)):
                for j in range(len(final_matrix)):
                    final_matrix[i][j] = final_matrix[i][j] + mat[i][j]
        # divide to get the avarege
        for i in range(len(final_matrix)):
            for j in range(len(final_matrix)):
                final_matrix[i][j] = float(final_matrix[i][j]) / float(len(confusion_matrixes))

        # calc_acc
        sum = 0
        for i in range(len(final_matrix)):
            for j in range(len(final_matrix)):
                sum = sum + final_matrix[i][j]

        reg_final_matrix = copy.deepcopy(final_matrix)
        for i in range(len(final_matrix)):
            for j in range(len(final_matrix)):
                reg_final_matrix[i][j] = reg_final_matrix[i][j] / sum
        acc = 0
        for i in range(len(final_matrix)):
            acc = acc + reg_final_matrix[i][i]

        types_w_acc = types + ["acc"]
        final_matrix.append([str(acc)])
        df = pd.DataFrame(final_matrix)
        df.columns = types
        df.index = types_w_acc

        # reg_final_matrix.append([str(acc)])
        reg_df = pd.DataFrame(reg_final_matrix)
        reg_df.columns = types
        reg_df.index = types
        print("acc = " + str(acc))
        return reg_df, acc, types
    return None


def run_learning(TITLE, PRINT, REG, RHOS, SVM, XGBOOST, Cross_validation, TUNED_PAREMETERS, DUMP_TO_PICKLE):

    data_loader = AllergyDataLoader(TITLE, PRINT, REG)

    print("learning..." + TITLE)
    # Learning: x=features, y=tags
    W_CON, ids, tag_map, task_name = get_learning_data(TITLE, data_loader)

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
            svm_results.to_csv("xgb_all_results_df_" + task_name + ".csv")
            pickle.dump(svm_results, open("xgb_all_results_df_" + task_name + ".pkl", 'wb'))

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
        svm_y_test_from_all_iter = []
        svm_y_score_from_all_iter = []
        svm_y_pred_from_all_iter = []
        svm_class_report_from_all_iter = []
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

            try:
                _, _, _, svm_roc_auc = roc_auc(y_tests[iter_num], y_pred, verbose=True, visualize=False,
                                           graph_title='SVM\n' + str(iter_num), save=True)
            except ValueError:
                print(task_name + " multiclass format is not supported at roc auc calculation")


            # save the y_test and y_score
            svm_y_test_from_all_iter.append(y_tests[iter_num])  # .values)
            svm_y_score_from_all_iter.append(list(y_score))
            svm_y_pred_from_all_iter.append(list(y_pred))
            svm_class_report_from_all_iter.append(svm_class_report)

            c = clf.coef_.tolist()[0]
            svm_coefs.append(c)
            entire_W = clf.coef_[0]
            
            preproccessed_data = data_loader.get_preproccessed_data
            pca_obj = data_loader.get_pca_obj
            bacteria_coeff = convert_pca_back_orig(pca_obj.components_, entire_W,
                                                   original_names=preproccessed_data.columns[:], visualize=False)

            bacterias = bacteria_coeff['Taxonome'].tolist()
            coefficients = bacteria_coeff['Coefficients'].tolist()

            bacteria_average.append(bacterias)
            bacteria_coeff_average.append(coefficients)

        if COEFF_PLOTS:
            create_coeff_plots_by_alogorithm(bacteria_coeff_average, bacterias, task_name, "SVM")

        all_y_train, all_predictions_train, all_test_real_tags, all_test_pred_tags, train_auc, test_auc, train_rho,\
        test_rho = calc_auc_on_joined_results(Cross_validation, y_trains, y_train_preds, y_tests, y_test_preds)


        print("\n------------------------------\n")
        try:
            _, _, _, svm_roc_auc = roc_auc(all_test_real_tags, all_test_pred_tags, verbose=True, visualize=True,
                                           graph_title='SVM\n' + task_name + " AUC on all iterations", save=True)
        except ValueError:
            print(task_name + "multiclass format is not supported at roc auc calculation")

        confusion_matrix_average, confusion_matrix_acc, confusion_matrix_indexes = edit_confusion_matrix(TITLE, confusion_matrixes, data_loader)
        print_confusion_matrix(confusion_matrix_average, confusion_matrix_indexes, confusion_matrix_acc, "SVM")

        if PRINT:
            print_confusion_matrix(confusion_matrix_average, confusion_matrix_indexes, confusion_matrix_acc, "SVM")

        print("svm final results: " + task_name)
        print("train_auc: " + str(train_auc))
        print("test_auc: " + str(test_auc))
        print("train_rho: " + str(train_rho))
        print("test_rho: " + str(test_rho))
        print("confusion_matrix_average: ")
        print(confusion_matrix_average)
        print("confusion_matrix_acc: ")
        print(confusion_matrix_acc)
        confusion_matrix_average.to_csv("svm_confusion_matrix_average_on_" + task_name + ".txt")

        with open("svm_AUC_on_" + task_name + ".txt", "w") as file:
            file.write("train_auc: " + str(train_auc) + "\n")
            file.write("test_auc: " + str(test_auc) + "\n")
            file.write("train_rho: " + str(train_rho) + "\n")
            file.write("test_rho: " + str(test_rho) + "\n")


        # save results to data frame
        results = pd.DataFrame(
            {
                "train accuracy": train_accuracies,
                "test accuracy": test_accuracies,
                "y score": svm_y_score_from_all_iter,
                "class report": svm_class_report_from_all_iter,
                "y test": svm_y_test_from_all_iter,
                "y pred": svm_y_pred_from_all_iter,
            }
        )



        results.append([np.mean(train_accuracies), np.mean(test_accuracies), None, None, None, None])
        pickle.dump(results, open("svm_clf_results_" + task_name + ".pkl", 'wb'))
        results.to_csv("svm_clf_results_" + task_name + ".csv")
        pickle.dump(confusion_matrix_average, open("svm_clf_confusion_matrix_results_" + task_name + ".pkl", 'wb'))
        confusion_matrix_average.to_csv("svm_clf_confusion_matrix_results_" + task_name + ".csv")

        """
        # Compute ROC curve and ROC area for each class
        n_classes = 2
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
    
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        """

        """
        svm_conf_stats = ''
        for train_mean, train_std, test_mean, test_std, params in zip(means_train, stds_train, means_test, stds_test,
                                                                      svm_clf.cv_results_['params']):
            svm_conf_stats += ("Train: %0.3f (+/-%0.03f) , Test: %0.3f (+/-%0.03f) for %r \n" % (
            train_mean, train_std * 2, test_mean, test_std * 2, params))
        
        entire_W = svm_clf.best_estimator_.coef_[0]
        W_pca = entire_W[starting_col:starting_col + n_components]
        bacteria_coeff = convert_pca_back_orig(pca_obj.components_, W_pca, original_names=preproccessed_data.columns[:],
                                               visualize=True)
        draw_horizontal_bar_chart(entire_W[0:starting_col], interesting_cols, title='Feature Coeff', ylabel='Feature',
                                  xlabel='Coeff Value', left_padding=0.3)
        # y_true, y_pred = y_test, svm_clf.predict(X_test)
        # # svm_class_report = classification_report(y_true, y_pred)
        # _, _, _, svm_roc_auc = roc_auc(y_true, y_pred, verbose=True, visualize=False,
        #         graph_title='SVM\n' + permutation_str)
        """
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
            xgb_results.to_csv("xgb_all_results_df_" + task_name + ".csv")
            pickle.dump(xgb_results, open("xgb_all_results_df_" + task_name + ".pkl", 'wb'))
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
        xgb_y_test_from_all_iter = []
        xgb_y_score_from_all_iter = []
        xgb_y_pred_from_all_iter = []
        xgb_class_report_from_all_iter = []
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
                _, _, _, xgb_roc_auc = roc_auc(y_tests[iter_num], y_pred, verbose=True, visualize=False,
                                               graph_title='xgb\n' + str(iter_num))
            except ValueError:
                print(task_name + " multiclass format is not supported at roc auc calculation")

            # save the y_test and y_score
            xgb_y_test_from_all_iter.append(y_tests[iter_num])  # .values)
            xgb_y_score_from_all_iter.append(y_score)
            xgb_y_pred_from_all_iter.append(y_pred)
            xgb_class_report_from_all_iter.append(xgb_class_report)
        # --------------------------------------! PLOT CORRELATION - XGBOOST -------------------------------

            c = np.array(clf.coef_)  #clf.coef_.tolist()[0]  numpy.ndarray
            xgb_coefs.append(c)
            entire_W = c  # clf.coef_[0]
            preproccessed_data = data_loader.get_preproccessed_data
            pca_obj = data_loader.get_pca_obj
            bacteria_coeff = convert_pca_back_orig(pca_obj.components_, entire_W,
                                                original_names=preproccessed_data.columns[:], visualize=False)

            bacterias = bacteria_coeff['Taxonome'].tolist()
            coefficients = bacteria_coeff['Coefficients'].tolist()

            bacteria_average.append(bacterias)
            bacteria_coeff_average.append(coefficients)
            # draw_horizontal_bar_chart(entire_W[0:starting_col], interesting_cols, title='Feature Coeff',
            #                         ylabel='Feature', xlabel='Coeff Value', left_padding=0.3)

        if COEFF_PLOTS:
            create_coeff_plots_by_alogorithm(bacteria_coeff_average, bacterias, task_name, "XGB")
            # data, names=None, title=None, ylabel=None, xlabel=None, use_pos_neg_colors=True, left_padding=0.4
            #draw_horizontal_bar_chart(entire_W[0:starting_col], interesting_cols, title='Feature Coeff',
             #                                                  ylabel='Feature', xlabel='Coeff Value', left_padding=0.3)


        all_y_train, all_predictions_train, all_test_real_tags, all_test_pred_tags, train_auc, test_auc, train_rho, \
        test_rho = calc_auc_on_joined_results(Cross_validation, y_trains, y_train_preds, y_tests, y_test_preds)

        confusion_matrix_average, confusion_matrix_acc, confusion_matrix_indexes = edit_confusion_matrix(TITLE, confusion_matrixes, data_loader)
        print_confusion_matrix(confusion_matrix_average, confusion_matrix_indexes, confusion_matrix_acc, "XGB")

        if PRINT:
            print_confusion_matrix(confusion_matrix_average, confusion_matrix_indexes, confusion_matrix_acc, "XGB")

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
        confusion_matrix_average.to_csv("xgb_confusion_matrix_average_on_" + task_name + ".txt")

        with open("xgb_AUC_on_" + task_name + ".txt", "w") as file:
            file.write("train_auc: " + str(train_auc) + "\n")
            file.write("test_auc: " + str(test_auc) + "\n")
            file.write("train_rho: " + str(train_rho) + "\n")
            file.write("test_rho: " + str(test_rho) + "\n")

        # save results to data frame
        results = pd.DataFrame(
            {
                "train accuracy": train_accuracies,
                "test accuracy": test_accuracies,
                 "y score": xgb_y_score_from_all_iter,
                "class report": xgb_class_report_from_all_iter,
                "y test": xgb_y_test_from_all_iter,
                "y pred": xgb_y_pred_from_all_iter,
            }
        )

        results.append([np.mean(train_accuracies), np.mean(test_accuracies), None, None, None, None])
        pickle.dump(results, open("xgb_clf_results_" + task_name + ".pkl", 'wb'))
        results.to_csv("xgb_clf_results_" + task_name + ".csv")
        pickle.dump(confusion_matrix_average, open("xgb_clf_confusion_matrix_results_" + task_name + ".pkl", 'wb'))
        confusion_matrix_average.to_csv("xgb_clf_confusion_matrix_results_" + task_name + ".csv")

    # ----------------------------------------------! NN ------------------------------------------------
    if RNN:
        X_trains = []
        X_tests = []
        y_trains = []
        y_tests = []
        for i in range(Cross_validation):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)
            X_trains.append(X_train)
            X_tests.append(X_test)
            y_trains.append(y_train)
            y_tests.append(y_test)

        nn_y_test_from_all_iter = None
        nn_y_score_from_all_iter = None
        nn_y_pred_from_all_iter = None
        test_model = tf_analaysis.nn_model()
        regularizer = regularizers.l2(0.1)
        test_model.build_nn_model(hidden_layer_structure=[{'units': n_components},
                                                          {'units': n_components*2, 'activation': tf.nn.relu, 'kernel_regularizer': regularizer},
                                                          ({'rate': 0.5}, 'dropout'),
                                                          {'units': n_components*2, 'activation': tf.nn.relu,'kernel_regularizer': regularizer},
                                                          ({'rate': 0.5}, 'dropout'),
                                                          {'units': n_components * 2, 'activation': tf.nn.relu, 'kernel_regularizer': regularizer},
                                                          ({'rate': 0.5}, 'dropout'),
                                                          {'units': 1, 'activation': 'sigmoid'}])

        test_model.compile_nn_model(loss='binary_crossentropy', metrics=['AUC'])
        hist = test_model.train_model(np.array(X_trains[i]), np.array(y_trains[i]), epochs=50, verbose=1, class_weight=get_weights(TITLE, data_loader))
        print('Train evaluation')
        test_model.evaluate_model(np.array(X_tests[i]), np.array(y_tests[i]))
        print('\n\nTest evaluation')
        test_model.evaluate_model(X_test.values, y_test.values.astype(np.float))


        y_score = test_model.model.predict_proba(X_test.values)
        y_pred = (test_model.model.predict(X_test.values)>0.5).astype(int)

        # save the y_test and y_score
        if nn_y_test_from_all_iter is None:
            nn_y_test_from_all_iter = y_test.values
            nn_y_score_from_all_iter = y_score
            nn_y_pred_from_all_iter = y_pred

        else:
            nn_y_test_from_all_iter = np.append(nn_y_test_from_all_iter, y_test.values)
            nn_y_score_from_all_iter = np.append(nn_y_score_from_all_iter, y_score)
            nn_y_pred_from_all_iter = np.append(nn_y_pred_from_all_iter, y_pred)


        print('\n *** NN **** \n')
        print(confusion_matrixes(nn_y_test_from_all_iter, nn_y_pred_from_all_iter))
        print(classification_report(nn_y_test_from_all_iter, nn_y_pred_from_all_iter))
        fpr, tpr, thresholds, nn_roc_auc = roc_auc(nn_y_test_from_all_iter, nn_y_score_from_all_iter, visualize=True,
                                                   graph_title='NN\n' + str(i))
        plt.show()
        print('******************* \n')


if __name__ == "__main__":
    # Pre process - mostly not needed
    PRINT = False  # do you want printing to screen?
    HIST = False  # drew histogram of the data after normalization
    REG = False  # run normalization to drew plots
    STAGES = False  # make tags for stages task
    RHOS = True  # calculate correlation between reaction to treatment to the tags

    TITLES = ["Success_task"]  #"Health_task", "Prognostic_task", "Milk_allergy_task",  , "Allergy_type_task"]
    """
    # what task to run ?
    Health_task = True     # is patient healthy or allergic ?
    if Health_task:
        TITLE = "health task"
    Prognostic_task = False  # is patient in day 0 of treatment is seem to react to the treatment ?
    if Prognostic_task:
        TITLE = "prognostic task"
    Success_task = False  # diagnostic: is patient in day X of treatment is seem to react to the treatment ?
    if Success_task:
        TITLE = "success task"
    Allergy_type_task = False  # multi-class classification to type of allergy given allergic
    if Allergy_type_task:
        TITLE = "allergy type task"
    Milk_allergy_task = False  # binary-class classification to type of allergy given allergic between milk and other
    if Milk_allergy_task:
        TITLE = "milk allergy task"
    """

    # learning method
    SVM = False
    XGBOOST = False
    RNN = False
    COEFF_PLOTS = False
    Cross_validation = 5

    # calculate parameters the first time running the task, and saving them to pickles
    TUNED_PAREMETERS = False
    DUMP_TO_PICKLE = True

    for TITLE in TITLES:
        run_learning(TITLE, PRINT, REG, RHOS, SVM, XGBOOST, Cross_validation, TUNED_PAREMETERS, DUMP_TO_PICKLE)

    # create a deep neural network for better accuracy percentages
    # questions to ask:
    # recognize milk allergy
