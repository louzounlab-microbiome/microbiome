from sklearn.metrics import roc_curve, auc
from LearningMethods.abstract_learning_model import AbstractLearningModel
import pandas as pd
import numpy as np
import os
from Plot import create_coeff_plots_by_alogorithm, make_class_coef_plots_from_multiclass_model_binary_sub_models
from infra_functions import convert_pca_back_orig


class SimpleLearningModel(AbstractLearningModel):
    def __init__(self):
        self.clf_y_test_from_all_iter = np.array([])
        self.clf_y_pred_from_all_iter = np.array([])
        self.clf_class_report_from_all_iter = np.array([])
        self.clf_y_score_from_all_iter = np.array([])
        super().__init__()

    def get_model_coeff(self, clf, pca_obj, pca_flag, binary_flag):
        raise NotImplemented  # implement for each simple model according to it structure...

    def calc_bacteria_coeff_average(self, num_of_classes, pca_obj, bacteria_names, clf, coefs_list, bacteria_coeff_average):
        BINARY = True if num_of_classes == 2 else False
        if pca_obj:  # preformed PCA -> convert_pca_back_orig
            c = self.get_model_coeff(clf, pca_obj, pca_flag=pca_obj, binary_flag=BINARY)
            bacteria_coeff = convert_pca_back_orig(pca_obj.components_, c,
                                                   original_names=bacteria_names,
                                                   visualize=False)
            coefficients = bacteria_coeff['Coefficients'].tolist()
            # coefs_list.append(c)
            # bacteria_coeff_average.append(coefficients)

        else:  # didn't preformed PCA -> no need to convert_pca_back_orig, use original coefficients
            coefficients = self.get_model_coeff(clf, pca_obj, pca_flag=pca_obj, binary_flag=BINARY)

        coefs_list.append(coefficients)
        bacteria_coeff_average.append(coefficients)

        return coefs_list, coefficients, bacteria_coeff_average

    def plot_bacteria_coeff_average(self, bacteria_coeff_average, num_of_classes, title, task_name, bacteria,
                                    cross_validation, algorithm, clf_folder_name, BINARY, names, edge_percent=1):
        bacteria_coeff_average = np.array(bacteria_coeff_average)

        if BINARY:  # binary
            avg_df = pd.DataFrame(bacteria_coeff_average)
            avg_cols = [x for x in avg_df.mean(axis=0)]

            create_coeff_plots_by_alogorithm(avg_cols, bacteria, task_name, algorithm, cross_validation,
                                             folder=clf_folder_name, edge_percent=edge_percent)

        else:  # for each cross of n classes - number of crosses is n*(n-1)/2
            task_name_and_class_list = []
            for i in range(num_of_classes):  # create crosses names
                for j in range(i + 1, num_of_classes):
                    task_name_and_class_list.append(
                        task_name.replace("_", " ") + " -\n" + names[i].replace("_", " ") + " class vs. " + \
                        names[j].replace("_", " ") + " class")

            # mean calculation of the cross validation results
            # 1) sum
            avg_df = pd.DataFrame(bacteria_coeff_average[0])
            for i in range(1, len(bacteria_coeff_average)):  # for number of cross validations
                avg_df = avg_df + pd.DataFrame(bacteria_coeff_average[i])
            # 2) divide
            avg_df = avg_df / len(bacteria_coeff_average)

            for i in range(len(avg_df)):  # create plots
                avg_cols = avg_df.loc[i]
                create_coeff_plots_by_alogorithm(avg_cols, bacteria, task_name_and_class_list[i], algorithm,
                                                 cross_validation,
                                                 folder=clf_folder_name, edge_percent=edge_percent)

            # create combined coeff plots for each group
            # get files names, using 'coeff'
            all_files_in_folder = os.listdir(clf_folder_name)
            coeff_files = []
            for file in all_files_in_folder:
                if 'coeff' in file:
                    coeff_files.append(file)
            pair_names = [[] for i in range(len(coeff_files))]
            for i, file in enumerate(coeff_files):
                for name in names:  # make sure the names are in the right order
                    name_ = name.replace(" ", "_")
                    if name in file:
                        pair_names[i].append(name)
                    elif name_ in file:
                        pair_names[i].append(name_)

                if len(pair_names[i]) < 2:
                    raise Exception  # problem with the naming of the files
                else:
                    idx_1 = file.index(pair_names[i][0])
                    idx_2 = file.index(pair_names[i][1])
                    if idx_2 < idx_1:
                        pair_names[i].reverse()

            make_class_coef_plots_from_multiclass_model_binary_sub_models(coeff_files, pair_names, names,
                                                                          task_name=task_name,
                                                                          algorithm=algorithm, num_of_iters=5,
                                                                          folder=clf_folder_name)

    def print_auc_for_iter(self, y_test, y_score):
        fpr, tpr, thresholds = roc_curve(np.array(y_test), np.array(y_score))
        roc_auc = auc(fpr, tpr)
        print('ROC AUC = ' + str(round(roc_auc, 4)))

    def save_y_test_and_score(self, y_test, y_pred, y_score, class_report):
        self.clf_y_test_from_all_iter = np.append(self.clf_y_test_from_all_iter, y_test)
        self.clf_y_pred_from_all_iter = np.append(self.clf_y_pred_from_all_iter, y_pred)
        self.clf_class_report_from_all_iter = np.append(self.clf_class_report_from_all_iter, class_report)
        self.clf_y_score_from_all_iter = np.append(self.clf_y_score_from_all_iter, y_score)


        """
        if self.clf_y_score_from_all_iter.size > 0:
            self.clf_y_score_from_all_iter = np.concatenate((self.clf_y_score_from_all_iter, y_score), axis=0)
        else:
            self.clf_y_score_from_all_iter = y_score
        """

    def save_results(self, task_name, train_auc, test_auc, train_rho, test_rho, confusion_matrix_average,
                     confusion_matrix_acc, train_accuracies, test_accuracies, y_score_from_all_iter,
                     y_pred_from_all_iter, y_test_from_all_iter, algorithm, clf_folder_name):

        print(algorithm + "final results: " + task_name + "\n" + "train_auc: " + str(train_auc) + "\n" + "test_auc: " +
              str(test_auc) + "\n" + "train_rho: " + str(train_rho) + "\n" + "test_rho: " + str(test_rho) + "\n" +
              "confusion_matrix_average: " + "\n")
        print(confusion_matrix_average)
        print("confusion_matrix_acc: " + "\n" + str(confusion_matrix_acc) + "\n")

        confusion_matrix_average.to_csv(
            os.path.join(clf_folder_name, algorithm + "_confusion_matrix_average_on_" + task_name + ".txt"))

        with open(os.path.join(clf_folder_name, task_name + "_" + algorithm + "_AUC_on_" + task_name + ".txt"),
                  "w") as file:
            file.write("train_auc: " + str(train_auc) + "\n")
            file.write("test_auc: " + str(test_auc) + "\n")
            file.write("train_rho: " + str(train_rho) + "\n")
            file.write("test_rho: " + str(test_rho) + "\n")
            file.write("\n")
            file.write("train accuracy: average " + str(np.mean(train_accuracies)) + "\n")
            for i, a in enumerate(train_accuracies):
                file.write(str(i) + ") " + str(a) + "\n")
            file.write("\n")
            file.write("test accuracy: average " + str(np.mean(test_accuracies)) + "\n")
            for i, a in enumerate(test_accuracies):
                file.write(str(i) + ") " + str(a) + "\n")

        # save results to data frame
        if len(y_score_from_all_iter.shape) > 1:
            score_map = {"y score " + str(i): y_score_from_all_iter[:, i] for i in
                         range(y_score_from_all_iter.shape[1])}
        else:
            score_map = {"y score": y_score_from_all_iter}

        score_map["y pred"] = y_pred_from_all_iter.astype(int)
        score_map["y test"] = y_test_from_all_iter.astype(int)
        results = pd.DataFrame(score_map)

        results.append([np.mean(train_accuracies), np.mean(test_accuracies), None, None, None, None])
        results.to_csv(
            os.path.join(clf_folder_name, task_name + "_" + algorithm + "_clf_results_" + task_name + ".csv"))
        confusion_matrix_average.to_csv(os.path.join(clf_folder_name, task_name + "_" + algorithm +
                                                     "_clf_confusion_matrix_results_" + task_name + ".csv"))



