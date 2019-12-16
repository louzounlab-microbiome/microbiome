import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# https://matplotlib.org/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py


def autolabel(ax, rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = round(rect.get_height(), 3)
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


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
    ax.set_xticklabels(data_sets_names, rotation=45)
    ax.legend()

    autolabel(ax, rects1)
    autolabel(ax, rects2)
    fig.tight_layout()
    return plt


if __name__ == "__main__":
    data_sets_names = ['Diet_study', 'VitamineA', 'MITRE_data_bokulich', 'MITRE_data_david'] # 'MDSINE_data_cdiff']
    # tax=5 vs. tax=6
    for algo in ["random forest", "ard regression"]:
        data_tax_5_real_rhos = []
        data_tax_6_real_rhos = []
        for data in data_sets_names:
            df = pd.read_csv(os.path.join("..", "Microbiome_Intervention", data, "tax=5", "best_models_conclusion.csv"))
            df = df.set_index("algorithm")
            data_tax_5_real_rhos.append(df.loc[algo + " - real"]["rhos mean"])
            df = pd.read_csv(os.path.join("..", "Microbiome_Intervention", data, "tax=6", "best_models_conclusion.csv"))
            df = df.set_index("algorithm")
            data_tax_6_real_rhos.append(df.loc[algo + " - real"]["rhos mean"])

        title = 'Rhos by data set\n' + algo + ' algorithm'
        plt = plot_multi_grouped_results(data_sets_names, data_tax_5_real_rhos, data_tax_6_real_rhos, 'Tax=5', 'Tax=6', 'Rhos',
                                   'Rhos by data set\n' + algo + ' algorithm' + '\n')
        plt.savefig(
            os.path.join("..", "Microbiome_Intervention", "plots", title.replace("\n", "_").replace(" ", "_") + ".svg")
            , bbox_inches='tight', format='svg')


    data_sets_names = ['Diet_study', 'VitamineA', 'MDSINE_data_cdiff', 'MDSINE_data_diet', 'MITRE_data_bokulich', 'MITRE_data_david'] # 'MDSINE_data_cdiff']
    # real vs. random
    for algo in ["ard regression", "random forest"]:
        for tax in ["5", "6"]:
            data_real_rhos = []
            data_random_rhos = []
            for data in data_sets_names:
                df = pd.read_csv(os.path.join("..", "Microbiome_Intervention", data, "tax=" + str(tax),
                                              "best_models_conclusion.csv"))
                df = df.set_index("algorithm")
                data_real_rhos.append(df.loc[algo + " - real"]["rhos mean"])
                data_random_rhos.append(df.loc[algo + " - random"]["rhos mean"])

            title = 'Rhos by data set\n' + algo + ' algorithm' + '\n' + 'taxonomy level ' + tax
            plt = plot_multi_grouped_results(data_sets_names, data_real_rhos, data_random_rhos, 'Real Rhos',
                                             'Random Rhos', 'Rhos',
                                             title)
            plt.savefig(os.path.join("..", "Microbiome_Intervention", "plots",
                                     title.replace("\n", "_").replace(" ", "_") + ".svg")
                        , bbox_inches='tight', format='svg')
            plt.show()

    # real vs. random
    for algo in ["ard regression", "random forest"]:
        for tax in ["5", "6"]:
            data_real_rmse = []
            data_random_rmse = []
            for data in data_sets_names:
                df = pd.read_csv(os.path.join("..", "Microbiome_Intervention", data, "tax=" + str(tax),
                                              "best_models_conclusion.csv"))
                df = df.set_index("algorithm")
                data_real_rmse.append(df.loc[algo + " - real"]["rmse mean"])
                data_random_rmse.append(df.loc[algo + " - random"]["rmse mean"])

            title = 'RMSE by data set\n' + algo + ' algorithm' + '\n' + 'taxonomy level ' + tax
            plt = plot_multi_grouped_results(data_sets_names, data_real_rmse, data_random_rmse, 'Real RMSE',
                                             'Random RMSE', 'RMSE',
                                             title)
            plt.savefig(os.path.join("..", "Microbiome_Intervention", "plots",
                                     title.replace("\n", "_").replace(" ", "_") + ".svg")
                        , bbox_inches='tight', format='svg')
            plt.show()

    """
    # paper results vs. us(tax=5)

    data_our_rhos = []
    data_paper_rhos = [0.3, 0.3, 0.3, 0.3, 0.3]  # fillllllllllllllllllllllllllllllllllllllllllllllllllllllll
    for algo in ["ard regression"]:  # , "random forest"
        for data in data_sets_names:
            df = pd.read_csv(os.path.join("..", "Microbiome_Intervention", data, "tax=5", "best_models_conclusion.csv"))
            df = df.set_index("algorithm")
            data_our_rhos.append(df.loc[algo + " - real"]["rhos mean"])

        title = 'Rhos by data set compered to previous results\n' + algo + ' algorithm\n'
        plt = plot_multi_grouped_results(data_sets_names, data_our_rhos, data_paper_rhos, 'Our', 'Them', 'Rhos',
                                   title)
        plt.savefig(os.path.join("..", "Microbiome_Intervention", "plots",
                                 title.replace("\n", "_").replace(" ", "_") + ".svg")
                    , bbox_inches='tight', format='svg')
        plt.show()

"""
