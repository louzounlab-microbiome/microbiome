import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# https://matplotlib.org/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
from sklearn.model_selection import LeaveOneOut


def get_values_min_and_max(values_list):
    min_list = []
    max_list = []
    for values in values_list:
        min_list.append(min(values))
        max_list.append(max(values))

    return min(min_list), max(max_list)



def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = round(rect.get_height(), 3)
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def plot_multi_grouped_results_2(names, values_list, std_list, labels_list, y_label, title):
    width = 1 / len(values_list)
    x = np.arange(len(names))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    min_, max_ = get_values_min_and_max(values_list)

    ax.set_ylim(min_, max_ + 0.2)
    rects_list = []
    addition = 0
    for values, label, stds in zip(values_list, labels_list, std_list):
        ax.bar(x + addition, values, width, label=label, ecolor='k', yerr=stds)
        #ax.errorbar(x + addition, values, yerr=stds)
        #rects_list.append(rect)
        addition += width


    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45)
    ax.legend()
    fig.tight_layout()
    plt.show()

    for rect in rects_list:
        autolabel(ax, rect)
    return plt



def plot_multi_grouped_results(names, values_1, values_2, label_1, label_2, y_label, title):
    x = np.arange(len(names))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    ax.set_ylim(min((min(values_1), (min(values_2)))), max((max(values_1), (max(values_2)))) + 0.1)
    rects1 = ax.bar(x - width/2, values_1, width, label=label_1)
    rects2 = ax.bar(x + width/2, values_2, width, label=label_2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45)
    ax.legend()

    autolabel(ax, rects1)
    autolabel(ax, rects2)
    fig.tight_layout()
    return plt


if __name__ == "__main__":
    plot_multi_grouped_results_2(
        ["random forest regression", "lasso regression", "svr regression", "ard regression",
         "bayesian ridge regression", "decision tree regression", "ridge regression", "linear regression"],
        [[0.304, 0.20938, 0.19519, 0.16231, 0.14425, 0.10522, 0.10441, 0.0407],
        [0.41508, 0.37548, 0.37471, 0.37, 0.35517, 0.18034, 0.27948, 0.15971]],
        [[0.13209, 0.14774, 0.1192, 0.14339, 0.16292, 0.13792, 0.15903, 0.14152],
         [0.11186, 0.12769, 0.11982, 0.1172, 0.14562, 0.1549, 0.12174, 0.10202]],
        ["vitamin A", "GDM"], "Correlation", "Regression Models Comparison").savefig("Regression_MSE.png")

    """
    plot_multi_grouped_results(
   ["ARD-regression", "NN-single bacteria", "NN-multi bacteria", "RNN-single bacteria",  "RNN-multi bacteria"],
        [0.56574, 0.4542, 0.4294, 0.6704, 0.4002], [0.58482, 0.3405, 0.2662, 0.6827, 0.2110],
   "GDM", "vitamin A", "RMSE", "GDM and vitamin A data set - result summary").savefig("this_4.png")

    best_models_conclusion_file_name = "interaction_network_structure_20_fold_test_size_0.5_conclusions.csv"
    plots_folder = "plots"
    data_sets_names = ['MDSINE_data_cdiff', 'MDSINE_data_diet', 'Diet_study',
                       'MITRE_data_bokulich', 'MITRE_data_david', 'VitamineA']
  
    # tax=5 vs. tax=6
    for algo in ["random forest", "ard regression"]:
        data_tax_5_real_rhos = []
        data_tax_6_real_rhos = []
        for data in data_sets_names:
            df = pd.read_csv(os.path.join("..", "Microbiome_Intervention", data, "tax=5", best_models_conclusion_file_name))
            df = df.set_index("algorithm")
            data_tax_5_real_rhos.append(df.loc[algo + " - real"]["rhos mean"])
            df = pd.read_csv(os.path.join("..", "Microbiome_Intervention", data, "tax=6", best_models_conclusion_file_name))
            df = df.set_index("algorithm")
            data_tax_6_real_rhos.append(df.loc[algo + " - real"]["rhos mean"])

        title = 'Rhos by data set\n' + algo + ' algorithm'
        plt = plot_multi_grouped_results(data_sets_names, data_tax_5_real_rhos, data_tax_6_real_rhos, 'Tax=5', 'Tax=6', 'Rhos',
                                   'Rhos by data set\n' + algo + ' algorithm' + '\n')
        plt.savefig(
            os.path.join("..", "Microbiome_Intervention", plots_folder, title.replace("\n", "_").replace(" ", "_") + ".svg")
            , bbox_inches='tight', format='svg')

    # real vs. random
    for algo in ["ard regression"]:
        for tax in ["5"]:
            data_real_rhos = []
            data_random_rhos = []
            for data in data_sets_names:
                if data in ['MDSINE_data_cdiff', 'MDSINE_data_diet']:
                    df = pd.read_csv(os.path.join("..", "Microbiome_Intervention", data, "tax=7",
                                                  best_models_conclusion_file_name))
                else:
                    df = pd.read_csv(os.path.join("..", "Microbiome_Intervention", data, "tax=" + str(tax),
                                                  best_models_conclusion_file_name))
                df = df.set_index("algorithm")
                data_real_rhos.append(df.loc[algo + " - real"]["rhos mean"])
                data_random_rhos.append(df.loc[algo + " - random"]["rhos mean"])

            title = 'Rhos by data set\n' + algo + ' algorithm' + '\n' + 'taxonomy level ' + tax
            plt = plot_multi_grouped_results(data_sets_names, data_real_rhos, data_random_rhos, 'Real Rhos',
                                             'Random Rhos', 'Rhos',
                                             title)
            plt.savefig(os.path.join("..", "Microbiome_Intervention", plots_folder,
                                     title.replace("\n", "_").replace(" ", "_") + ".svg")
                        , bbox_inches='tight', format='svg')
            plt.show()

    # real vs. random
    for algo in ["ard regression"]:
        for tax in ["6"]:
            data_real_rmse = []
            data_random_rmse = []
            for data in data_sets_names:
                if data in ['MDSINE_data_cdiff', 'MDSINE_data_diet']:
                    df = pd.read_csv(os.path.join("..", "Microbiome_Intervention", data, "tax=7",
                                                  best_models_conclusion_file_name))
                else:
                    df = pd.read_csv(os.path.join("..", "Microbiome_Intervention", data, "tax=" + str(tax),
                                                  best_models_conclusion_file_name))
                df = df.set_index("algorithm")
                data_real_rmse.append(df.loc[algo + " - real"]["rmse mean"])
                data_random_rmse.append(df.loc[algo + " - random"]["rmse mean"])

            title = 'RMSE by data set\n' + algo + ' algorithm' + '\n' + 'taxonomy level ' + tax
            plt = plot_multi_grouped_results(data_sets_names, data_real_rmse, data_random_rmse, 'Real RMSE',
                                             'Random RMSE', 'RMSE',
                                             title)
            plt.savefig(os.path.join("..", "Microbiome_Intervention", plots_folder,
                                     title.replace("\n", "_").replace(" ", "_") + ".svg")
                        , bbox_inches='tight', format='svg')
            plt.show()


    # paper results vs. us(tax=5)
    data_our_rhos = []
    data_paper_auc = [0.93, 0.85, 0, 0, 0, 0]  # fillllllllllllllllllllllllllllllllllllllllllllllllllllllll
    data_paper_rmse = [0.56, 0.85, 0, 0, 0, 0]  # fillllllllllllllllllllllllllllllllllllllllllllllllllllllll
    for algo in ["ard regression"]:
        for tax in ["6"]:
            for data in data_sets_names:
                if data in ['MDSINE_data_cdiff', 'MDSINE_data_diet']:
                    df = pd.read_csv(os.path.join("..", "Microbiome_Intervention", data, "tax=7",
                                                  best_models_conclusion_file_name))
                else:
                    df = pd.read_csv(os.path.join("..", "Microbiome_Intervention", data, "tax=" + str(tax),
                                                  best_models_conclusion_file_name))
                df = df.set_index("algorithm")
                data_our_rhos.append(df.loc[algo + " - real"]["auc"])

            title = 'AUC by data set compered to previous results\n' + algo + ' algorithm\n' + 'taxonomy level ' + tax
            plt = plot_multi_grouped_results(data_sets_names, data_our_rhos, data_paper_auc, 'Our', 'Them', 'AUC',
                                       title)
            plt.savefig(os.path.join("..", "Microbiome_Intervention", plots_folder,
                                     title.replace("\n", "_").replace(" ", "_") + ".svg")
                        , bbox_inches='tight', format='svg')
            plt.show()
    """
    data_sets_names = ['MDSINE_data_cdiff_AUC', 'MDSINE_data_cdiff_RMSE']
    data_paper_results = [0.93, 0.56]  # AUC, RMSE
    data_our_results = [0.79284, 0.04134]
    for algo in ["ard regression"]:
        for tax in ["7"]:
            for data in ['MDSINE_data_cdiff']:
                df = pd.read_csv(os.path.join("..", "Microbiome_Intervention", data, "tax=" + str(tax),
                                                  best_models_conclusion_file_name))
                df = df.set_index("algorithm")
                data_our_results.append(df.loc[algo + " - real"]["auc"])
                data_our_results.append(df.loc[algo + " - real"]["rmse mean"])

            title = 'AUC and RMSE by data set compered to MDSINE results\n' + algo + ' algorithm\n' + 'taxonomy level ' + tax
            plt = plot_multi_grouped_results(data_sets_names, data_our_results, data_paper_results, 'Our', 'Them', 'SCORE',
                                             title)
            plt.savefig(os.path.join("..", "Microbiome_Intervention", plots_folder,
                                     title.replace("\n", "_").replace(" ", "_") + ".svg")
                        , bbox_inches='tight', format='svg')
            plt.show()
