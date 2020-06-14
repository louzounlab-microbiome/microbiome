import pickle
from random import shuffle

import pandas as pd
import numpy as np
import os
from os.path import join
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, balanced_accuracy_score

from Plot import plot_roc_auc, multi_class_roc_auc
from Plot.plot_bacteria_intraction_network import plot_bacteria_intraction_network


def create_G(title, df_path, folder, p_value=0.001, simulation=False):
    """
    To determine which bacteria have interactions and create a visual graph, we will take the file which retained the
    prediction values for a basic case and for positive and negative change created by "bacteria_network_nni_runner".
    Comparing the original prediction and the modified data’s prediction distributions, if the change between the two is
    significant according to U test, we conclude that there is interaction between the bacterial pair.
    The type of interaction will be determined by the obtained change- increasing or decreasing the count of the
    bacterium at a fixed size, and its effect, increase or decrease in the prediction of the count of bacteria.

    :param title: (string) graph name.
    :param df_path: (string) csv file path created by "bacteria_network_nni_runner"
    :param folder: (string) main dataset folder "DATASET/tax=x"
    :param p_value: (float) the p value cutoff for concluding that there is interaction between the bacterial pair.
    :param simulation: (Bool) True / False - there are different parameters for the graph in case of simulation -
    not real bacteria
    :return: Does not return an object, create files and display HTML of the graph and save them in the a folder.
    """
    g = pd.DataFrame(columns=["FROM", "TO", "RELATION"])
    df = pd.read_csv(df_path)
    with open(os.path.join(folder, "bacteria.txt"), "r") as b_file:
        bacteria = b_file.readlines()
        bacteria = [b.rstrip() for b in bacteria]

    number_of_bacteria = len(bacteria)

    folder = os.path.join(folder, 'interaction_network')
    if not os.path.exists(folder):
        os.mkdir(folder)

    edges = []
    color_map = {"-+ -> -+": '#FF4233',  # red
                 "-+ -> +-": '#3E89C0',  # blue
                 "-+ -> ++": '#2d7826',  # green
                 "-+ -> --": '#FFA500',  # orange
                 "+ -> +": '#00FFFF',  # aqua
                 "+ -> -": '5E5E08',  # olive
                 "- -> +": '#F8F414',  # yellow
                 "- -> -": '#1F0202'}  # brown
    y_score_pos = np.array([[0.0 for i in range(number_of_bacteria)] for i in range(number_of_bacteria)])
    y_p_val_pos = np.array([[0.0 for i in range(number_of_bacteria)] for i in range(number_of_bacteria)])
    y_score_neg = np.array([[0.0 for i in range(number_of_bacteria)] for i in range(number_of_bacteria)])
    y_p_val_neg = np.array([[0.0 for i in range(number_of_bacteria)] for i in range(number_of_bacteria)])

    no_change_df = df.loc[df['CHANGE'] == "no change"]
    no_change_df = no_change_df.sort_values(by="BACTERIA")
    for main_bact in range(0, number_of_bacteria):
        main_bact_df = df[df["BACTERIA"] == main_bact]
        no_change_y_pred = np.array([float(val) for val in no_change_df[no_change_df["BACTERIA"] == main_bact].values[0][3].split(" ")[:-1]])
        for sub_bact in range(0, number_of_bacteria):
            sub_bact_df = main_bact_df[main_bact_df["CHANGED_BACTERIA"] == sub_bact]
            pos_y_pred = np.array([float(val) for val in sub_bact_df[sub_bact_df["CHANGE"] == "plus 0.5"].values[0][3].split(" ")[:-1]])
            neg_y_pred = np.array([float(val) for val in sub_bact_df[sub_bact_df["CHANGE"] == "minus 0.5"].values[0][3].split(" ")[:-1]])
            pos_diff_array = no_change_y_pred - pos_y_pred
            pos_diff_mean = np.mean(pos_diff_array)
            neg_diff_array = no_change_y_pred - neg_y_pred
            neg_diff_mean = np.mean(neg_diff_array)

            pos_u, pos_u_test_p_val = mannwhitneyu(no_change_y_pred, pos_y_pred)
            neg_u, neg_u_test_p_val = mannwhitneyu(no_change_y_pred, neg_y_pred)

            y_score_pos[main_bact][sub_bact] = pos_u
            y_p_val_pos[main_bact][sub_bact] = pos_u_test_p_val
            y_score_neg[main_bact][sub_bact] = neg_u
            y_p_val_neg[main_bact][sub_bact] = neg_u_test_p_val

            if main_bact == sub_bact:
                continue

            if pos_u_test_p_val < p_value:
                if neg_u_test_p_val < p_value:
                    #print("pos_p_values", str(pos_u_test_p_val), "neg_p_values", str(neg_u_test_p_val),
                    #      "pos_diff_mean", str(pos_diff_mean), "neg_diff_mean", str(neg_diff_mean), sep=" ")
                    # both significant
                    if pos_diff_mean > 0 and neg_diff_mean < 0:
                        #print("-+ -> -+")
                        g.loc[len(g)] = [sub_bact, main_bact, "-+ -> -+"]
                        edges.append((sub_bact, main_bact, {'color': color_map["-+ -> -+"]}))

                    elif pos_diff_mean < 0 and neg_diff_mean > 0:
                        #print("-+ -> +-")
                        g.loc[len(g)] = [sub_bact, main_bact, "-+ -> +-"]
                        edges.append((sub_bact, main_bact, {'color': color_map["-+ -> +-"]}))


                    elif pos_diff_mean > 0 and neg_diff_mean > 0:
                        #print("-+ -> ++")
                        g.loc[len(g)] = [sub_bact, main_bact, "-+ -> ++"]
                        edges.append((sub_bact, main_bact, {'color': color_map["-+ -> ++"]}))

                    elif pos_diff_mean < 0 and neg_diff_mean < 0:
                        #print("-+ -> --")
                        g.loc[len(g)] = [sub_bact, main_bact, "-+ -> --"]
                        edges.append((sub_bact, main_bact, {'color': color_map["-+ -> --"]}))
                # only one way change is significant
                else:  # neg_u_test_p_val > p_value
                    print("neg_u_test_p_val > p_value")
                    print("pos_p_values", str(pos_u_test_p_val), "neg_p_values", str(neg_u_test_p_val),
                          "pos_diff_mean", str(pos_diff_mean), "neg_diff_mean", str(neg_diff_mean), sep=" ")
                    if pos_diff_mean > 0:
                        print([sub_bact, main_bact, "+ -> +"])
                        g.loc[len(g)] = [sub_bact, main_bact, "+ -> +"]
                        edges.append((sub_bact, main_bact, {'color': color_map["+ -> +"]}))

                    elif pos_diff_mean < 0:
                        print( [sub_bact, main_bact, "+ -> -"])
                        g.loc[len(g)] = [sub_bact, main_bact, "+ -> -"]
                        edges.append((sub_bact, main_bact, {'color': color_map["+ -> -"]}))
            elif neg_u_test_p_val < p_value:
                print("pos_u_test_p_val > p_value")
                print("pos_p_values", str(pos_u_test_p_val), "neg_p_values", str(neg_u_test_p_val),
                      "pos_diff_mean", str(pos_diff_mean), "neg_diff_mean", str(neg_diff_mean), sep=" ")
                if neg_diff_mean > 0:
                    print([sub_bact, main_bact, "- -> +"])
                    g.loc[len(g)] = [sub_bact, main_bact, "- -> +"]
                    edges.append((sub_bact, main_bact, {'color': color_map["- -> +"]}))

                elif neg_diff_mean < 0:
                    print([sub_bact, main_bact, "- -> -"])
                    g.loc[len(g)] = [sub_bact, main_bact, "- -> -"]
                    edges.append((sub_bact, main_bact, {'color': color_map["- -> -"]}))

        g.to_csv(os.path.join(folder, "graph_u_test_p_value_" + str(p_value) + "_" + title + ".csv"), index=False)
        pd.DataFrame(columns=range(number_of_bacteria), index=range(number_of_bacteria), data=y_score_pos)\
            .to_csv(os.path.join(folder, title + "_pos_scores_u_test_p_value_" + str(p_value) + ".csv"), index=False)
        pd.DataFrame(columns=range(number_of_bacteria), index=range(number_of_bacteria), data=y_p_val_pos)\
            .to_csv(os.path.join(folder, title + "_pos_p_values_u_test_p_value_" + str(p_value) + ".csv"), index=False)
        pd.DataFrame(columns=range(number_of_bacteria), index=range(number_of_bacteria), data=y_score_neg)\
            .to_csv(os.path.join(folder, title + "_neg_scores_u_test_p_value_" + str(p_value) + ".csv"), index=False)
        pd.DataFrame(columns=range(number_of_bacteria), index=range(number_of_bacteria), data=y_p_val_neg)\
            .to_csv(os.path.join(folder, title + "_neg_p_values_u_test_p_value_" + str(p_value) + ".csv"), index=False)

        node_list = [i for i in range(number_of_bacteria)]
        edge_list = []
        color_list = []

        num_of_edges_list = [0 for i in range(number_of_bacteria)]

        for e_ in edges:
            if e_[0] != e_[1]:
                num_of_edges_list[e_[0]] += 1
                num_of_edges_list[e_[1]] += 1

        for e_ in edges:
            if (num_of_edges_list[e_[0]] > 0) or (num_of_edges_list[e_[1]] > 0):
                edge_list.append((e_[0], e_[1]))
                color_list.append((e_[2]["color"]))

        shuffle(edge_list)

        pickle.dump(node_list, open(os.path.join(folder, title + "_node_list" + "_pvalue_" + str(p_value) + ".pkl"), "wb"))
        pickle.dump(edge_list, open(os.path.join(folder, title + "_edge_list" + "_pvalue_" + str(p_value) + ".pkl"), "wb"))
        pickle.dump(color_list, open(os.path.join(folder, title + "color_list" + "_pvalue_" + str(p_value) + ".pkl"), "wb"))
        v = [100] * number_of_bacteria

        if simulation:
            plot_bacteria_intraction_network([str(i) for i in range(number_of_bacteria)], node_list, v, edge_list,
                                             color_list, title + "_pvalue_" + str(p_value), folder,
                                             control_color_and_shape=False)

        else:
            plot_bacteria_intraction_network(bacteria, node_list, v, edge_list, color_list, title, folder)


def plot_net(folder, title, p_value, simulation=False):
    """
    use the output of "create_G" 3 pickles : node_list, edge_list and color_list to plot the network
    :param folder: (string) main dataset folder "DATASET/tax=x"
    :param title: (string) graph name.
    :param p_value: (float) the p value cutoff for concluding that there is interaction between the bacterial pair.
    :param simulation: (Bool) True / False - there are different parameters for the graph in case of simulation -
    :return: Does not return an object, create files and display HTML of the graph and save them in the a folder.

    """
    folder = os.path.join(folder, 'interaction_network')
    node_list = pickle.load(open(os.path.join(folder, title + "_node_list" + "_pvalue_" + str(p_value) + ".pkl")))
    edge_list = pickle.load(open(os.path.join(folder, title + "_edge_list" + "_pvalue_" + str(p_value) + ".pkl")))
    color_list = pickle.load(open(os.path.join(folder, title + "_color_list" + "_pvalue_" + str(p_value) + ".pkl")))

    with open(os.path.join(os.getcwd(), 'GDM_data', 'tax=5', "bacteria.txt"), "r") as b_file:
        bacteria = b_file.readlines()
        bacteria = [b.rstrip() for b in bacteria]

    v = [100] * len(bacteria)

    if simulation:
        plot_bacteria_intraction_network([str(i) for i in range(len(bacteria))], node_list, v, edge_list,
                                         color_list, title + "_pvalue_" + str(p_value), folder,
                                         control_color_and_shape=False)

    else:
        plot_bacteria_intraction_network(bacteria, node_list, v, edge_list, color_list, title, folder)


def simulation_auc(true_csv, pos_score_csv, pos_p_values_csv, neg_score_csv, neg_p_values_csv, folder, model, plot_distribution=True):
    """
    There is no way to truly know whether the network of interactions are correct for the actual data being tested
    because detailed information on bacterial interactions is not yet available.
    Therefore, in order to examine the nature of the network's prediction, we will use simulated data of bacteria
    counts (OTUs) based on previously selected interactions and examine whether these interactions are identified by
     his method.
     Predictability can be scored by calculating the AUC between the discrete value of the true known interactions,
     and the continuous value of 1 / U test score- which is found to be highly correlated to the interaction type.
    
    :param true_csv: (DataFrame) simulated data csv of true interactions, created by "Create_otu_population.py".
    :param pos_score_csv: (DataFrame) Score of the U test for positive change, created by "create_G" function.
    :param pos_p_values_csv: (DataFrame) p value of the U test for positive change, created by "create_G" function.
    :param neg_score_csv: (DataFrame) Score of the U test for negative change, created by "create_G" function.
    :param neg_p_values_csv: (DataFrame) p value of the U test for negative change, created by "create_G" function.
    :param folder: (string) folder to save results in.
    :param model: (string) model used to predict the bacteria values.
    :param plot_distribution: (Bool) True / False - plot distribution of the U test score and p value.

    :return: Does not return an object, create plots for AUC and if wanted distribution.
    """
    if not os.path.exists(folder):
        os.mkdir(folder)
    original_y_true = true_csv
    y_true = true_csv.values.flatten()
    y_true_1 = original_y_true.replace(-1, 0).values.flatten()
    y_true_2 = original_y_true.replace(1, 0).replace(-1, 1).values.flatten()
    y_true_3 = original_y_true.replace(-1, 1).values.flatten()

    pos_y_score = pos_score_csv.values.flatten()
    pos_y_score_ = [1 / val for val in pos_y_score.flatten()]
    pos_y_p_val = pos_p_values_csv.values.flatten()
    pos_y_p_val_ = [1 / val for val in pos_y_p_val.flatten()]

    neg_y_score = neg_score_csv.values.flatten()
    neg_y_score_ = [1 / val for val in neg_y_score.flatten()]
    neg_y_p_val = neg_p_values_csv.values.flatten()
    neg_y_p_val_ = [1 / val for val in neg_y_p_val.flatten()]

    old_plot_distribition = False
    if old_plot_distribition:
        for target, target_name in [(pos_y_score_ + neg_y_score_, "all changes U test score"),
                                    (pos_y_p_val_ + neg_y_p_val_, "all changes U test p value")]:
            _1 = []
            _0 = []
            _m1 = []
            for true, tar in zip(y_true, target):
                if true == 1:
                    _1.append(tar)
                elif true == 0:
                    _0.append(tar)
                elif true == -1:
                    _m1.append(tar)

            title = model + " Interaction " + target_name + " Distribution"
            [count, bins] = np.histogram(_1, 50)
            plt.bar(bins[:-1], count, width=0.8 * (bins[1] - bins[0]), alpha=0.5, label="1",
                    color="#FF4233")
            [count, bins] = np.histogram(_m1, 50)
            plt.bar(bins[:-1], count, width=0.8 * (bins[1] - bins[0]), alpha=0.5, label="-1",
                    color="#48C03E")
            plt.title(title)
            plt.xlabel(target_name)
            plt.ylabel('Number of samples')
            plt.legend()
            plt.savefig(os.path.join(folder, title.replace(" ", "_") + "_1_-1.png"))
            plt.show()
            plt.close()

            [count, bins] = np.histogram(_0, 50)
            plt.bar(bins[:-1], count, width=0.8 * (bins[1] - bins[0]), alpha=0.5, label="0",
                    color="#3E89C0")

            [count, bins] = np.histogram(_1, 50)
            plt.bar(bins[:-1], count, width=0.8 * (bins[1] - bins[0]), alpha=0.5, label="1",
                    color="#FF4233")

            [count, bins] = np.histogram(_m1, 50)
            plt.bar(bins[:-1], count, width=0.8 * (bins[1] - bins[0]), alpha=0.5, label="-1",
                    color="#48C03E")
            plt.title(title)
            plt.xlabel(target_name)
            plt.ylabel('Number of samples')
            plt.legend()
            plt.savefig(os.path.join(folder, title.replace(" ", "_") + "_0_1_-1.png"))
            plt.show()
            plt.close()

    if plot_distribution:
        for target, target_name in [(pos_y_score_ + neg_y_score_, "all changes U test score"),
                                    (pos_y_p_val + neg_y_p_val, "all changes U test p value")]:
            _1 = []
            _0 = []
            _m1 = []
            for true, tar in zip(y_true, target):
                if true == 1:
                    _1.append(tar)
                elif true == 0:
                    _0.append(tar)
                elif true == -1:
                    _m1.append(tar)

            title = model + " Interaction " + target_name + " Distribution"

            [count, bins] = np.histogram(_1, 50)
            plt.bar(bins[:-1], count, width=0.8 * (bins[1] - bins[0]), alpha=0.5, label="1",
                    color="#FF4233")
            [count, bins] = np.histogram(_m1, 50)
            plt.bar(bins[:-1], count, width=0.8 * (bins[1] - bins[0]), alpha=0.5, label="-1",
                    color="#48C03E")
            plt.title(title)
            plt.xlabel(target_name)
            plt.ylabel('Number of samples')
            plt.legend()
            plt.savefig(os.path.join(folder, title.replace(" ", "_") + "_1_-1.png"))
            plt.show()
            plt.close()

            [count, bins] = np.histogram(_0, 50)
            plt.bar(bins[:-1], count, width=0.8 * (bins[1] - bins[0]), alpha=0.5, label="0",
                    color="#3E89C0")

            [count, bins] = np.histogram(_1, 50)
            plt.bar(bins[:-1], count, width=0.8 * (bins[1] - bins[0]), alpha=0.5, label="1",
                    color="#FF4233")

            [count, bins] = np.histogram(_m1, 50)
            plt.bar(bins[:-1], count, width=0.8 * (bins[1] - bins[0]), alpha=0.5, label="-1",
                    color="#48C03E")
            plt.title(title)
            plt.xlabel(target_name)
            plt.ylabel('Number of samples')
            plt.legend()
            plt.savefig(os.path.join(folder, title.replace(" ", "_") + "_0_1_-1.png"))
            plt.show()
            plt.close()
    # interaction vs. no interaction
    plot_roc_auc(list(y_true_3) + list(y_true_3), neg_y_p_val_ + pos_y_p_val_, visualize=True,
                 graph_title=model + '\nROC curve - U test p value correlation\n all interactions', save=True,
                 folder=folder, fontsize=17)

    plot_roc_auc(list(y_true_3) + list(y_true_3), neg_y_score_ + pos_y_score_, visualize=True,
                 graph_title=model + '\nROC curve - U test score correlation\nall interactions', save=True,
                 folder=folder, fontsize=17)


"""
def simulation_acc(true_csv, pred_csv):
    original_y_true = true_csv
    y_true = original_y_true.values.flatten()
    y_true_not_0_idx = [i for i, val in enumerate(y_true) if val != 0]
    y_true_1 = original_y_true.replace(-1, 0).values.flatten()
    y_true_2 = original_y_true.replace(1, 0).replace(-1, 1).values.flatten()

    y_pred = pred_csv.values.flatten()
    y_pred_not_0_idx = [i for i, val in enumerate(y_pred) if val != 0]

    _0_0 = 0
    _1_1 = 0
    _m1_m1 = 0
    _1_m1 = 0
    _0_1 = 0
    _0_m1 = 0
    for t, p in zip(y_true, y_pred):
        if t == 0 and p == 0:
            _0_0 += 1
        elif t == 1 and p == 1:
            _1_1 += 1
        elif t == -1 and p == -1:
            _m1_m1 += 1
        elif t == -1 and p == 1:
            _1_m1 += 1
        elif t == -1 and p == 0:
            _0_m1 += 1
        elif t == 0 and p == 1:
            _0_1 += 1

    print("_0_0 " + str(_0_0 / len(y_true) * 100))
    print("_1_1 " + str(_1_1 / len(y_true) * 100))
    print("_m1_m1 " + str(_m1_m1 / len(y_true) * 100))
    print("_1_m1 " + str(_1_m1 / len(y_true) * 100))
    print("_0_1 " + str(_0_1 / len(y_true) * 100))
    print("_0_m1 " + str(_0_m1 / len(y_true) * 100))
    print("all samples")
    acc = accuracy_score(y_true, y_pred)
    print("accuracy " + str(acc))
    ba_acc = balanced_accuracy_score(y_true, y_pred)
    print("balanced accuracy " + str(ba_acc))
    recall = recall_score(y_true, y_pred, average="micro")
    print("micro - recall " + str(recall))
    recall = recall_score(y_true, y_pred, average="macro")
    print("macro - recall " + str(recall))
    precision = precision_score(y_true, y_pred, average="micro")
    print("micro - precision " + str(precision))
    precision = precision_score(y_true, y_pred, average="macro")
    print("macro - precision " + str(precision))
    f1 = f1_score(y_true, y_pred, average="micro")
    print("micro - f1 " + str(f1))
    f1 = f1_score(y_true, y_pred, average="macro")
    print("macro - f1 " + str(f1))


    not_0_idx = list(set(y_pred_not_0_idx + y_true_not_0_idx))
    y_true = [y for i, y in enumerate(y_true) if i in not_0_idx]
    y_pred = [y for i, y in enumerate(y_pred) if i in not_0_idx]
    print("only non zero samples")

    acc = accuracy_score(y_true, y_pred)
    print("accuracy " + str(acc))
    ba_acc = balanced_accuracy_score(y_true, y_pred)
    print("balanced accuracy " + str(ba_acc))
    recall = recall_score(y_true, y_pred, average="micro")
    print("micro - recall " + str(recall))
    recall = recall_score(y_true, y_pred, average="macro")
    print("macro - recall " + str(recall))
    precision = precision_score(y_true, y_pred, average="micro")
    print("micro - precision " + str(precision))
    precision = precision_score(y_true, y_pred, average="macro")
    print("macro - precision " + str(precision))
    f1 = f1_score(y_true, y_pred, average="micro")
    print("micro - f1 " + str(f1))
    f1 = f1_score(y_true, y_pred, average="macro")
    print("macro - f1 " + str(f1))

    print("2")
"""


def interactions_tuple_to_interaction_df(interactions, number_of_bacteria):
    """
    Transforming the tuple of interactions into a matrix held in a data frame.
    :param interactions: (tuple) Tuple of interactions.
    :param number_of_bacteria: (int) Number of bacteria.
    :return: DataFrame of interactions.
    """
    data = [[0 for i in range(number_of_bacteria)]for i in range(number_of_bacteria)]
    for e in interactions:
        data[e[0]][e[1]] = e[2][0]  # !!!!!!! double check
    df = pd.DataFrame(index=range(number_of_bacteria), columns=range(number_of_bacteria), data=data)
    return df


def interactions_G_to_interaction_df(interactions, number_of_bacteria):
    """
    Transforming the csv file of interactions into a matrix held in a data frame.
    :param interactions: (DataFrame) Tuple of interactions.
    :param number_of_bacteria: (int) Number of bacteria.
    :return: DataFrame of interactions.
    """
    data = [[0 for i in range(number_of_bacteria)]for i in range(number_of_bacteria)]
    relation_to_number_map ={"-+ -> -+": 1,  # red
                 "-+ -> +-": -1,  # blue
                 "-+ -> ++": 2,  # green
                 "-+ -> --":2,  # orange
                 "+ -> +": 1,  # aqua
                 "+ -> -": -1,  # olive
                 "- -> +": 1,  # yellow
                 "- -> -": -1}  # brown
    for i, e in interactions.iterrows():
        data[e["TO"]][e["FROM"]] = relation_to_number_map[e["RELATION"]]
    df = pd.DataFrame(index=range(number_of_bacteria), columns=range(number_of_bacteria), data=data)
    return df


if __name__ == "__main__":
    run_sim_1 = True
    run_vitamineA = False
    run_GDM = False
    if run_sim_1:
        task_title = "simulation_1"
        folder = os.path.join(os.getcwd(), 'Simulations', '1')
        model_list = ["ard regression", "lasso regression", "linear regression", "ridge regression",
        "bayesian ridge regression", "svr regression", "decision tree regression", "random forest regression"]
        for m in model_list:
            model_name = m.replace(" ", "_") + "_"
            df_title = join(folder, model_name + "interaction_network_change_in_data_df.csv")
            pvalue = 0.001
            create_G(task_title, df_title, folder, number_of_bacteria=100, p_value=pvalue, simulation=True)
            true = pd.read_csv(os.path.join('Simulations', '1', "simulation_interactions.csv"))

            pos_score = pd.read_csv(os.path.join(folder, model_name + "simulation_1_pos_scores_u_test_p_value_" + str(pvalue) + ".csv"))
            pos_p_values = pd.read_csv(os.path.join(folder, model_name + "simulation_1_pos_p_values_u_test_p_value_" + str(pvalue) + ".csv"))
            neg_score = pd.read_csv(os.path.join(folder, model_name + "simulation_1_neg_scores_u_test_p_value_" + str(pvalue) + ".csv"))
            neg_p_values = pd.read_csv(os.path.join(folder, model_name + "simulation_1_neg_p_values_u_test_p_value_" + str(pvalue) + ".csv"))
            simulation_auc(true, pos_score, pos_p_values, neg_score, neg_p_values, os.path.join(folder, "plots"), model=model_name)

    if run_vitamineA:
        title = "VITAMINA"
        folder = os.path.join(os.getcwd(), 'VitamineA', 'tax=5')
        df_path = join(folder, "interaction_network_{STRUCTURE:001L200H,TRAIN_TEST_SPLIT:0.7,EPOCHS:70,LEARNING_RATE:0.001,"
                       "OPTIMIZER:Adam,REGULARIZATION:0.01,DROPOUT:0.1}_df.csv")
        create_G(title, df_path, folder, simulation=False)

    if run_GDM:
        title = "GDM"
        folder = os.path.join(os.getcwd(), 'GDM_data', 'tax=5')
        df_path = join(folder,
                       "interaction_network_{STRUCTURE:001L200H,TRAIN_TEST_SPLIT:0.7,EPOCHS:70,"
                       "LEARNING_RATE:0.001,OPTIMIZER:Adam,REGULARIZATION:0.01,DROPOUT:0.1}_df.csv")
        create_G(title, df_path,  folder, simulation=False)
